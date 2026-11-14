"""Unit and integration tests for real-time inference, temporal smoothing, and async TTS."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sign_translator.architecture import LSTMClassifier
from sign_translator.live_inference import (
    AsyncTTSWorker,
    DummyTTSEngine,
    PredictionSmoother,
    Pyttsx3Engine,
    RealtimeGesturePipeline,
)
from sign_translator.schemas import PipelineConfig


@pytest.mark.unit
class TestPredictionSmoother:
    """Unit tests for M-of-K temporal voting, confidence filtering, and cooldown timers."""

    def test_single_frame_false_spike_suppressed(self) -> None:
        """TC-01: Single high-confidence anomaly frame is suppressed by M-of-K voting."""
        smoother = PredictionSmoother(
            window_size=10,
            min_matches=6,
            confidence_threshold=0.85,
            cooldown_seconds=2.0,
        )

        # Frame 1: Outlier spike
        spike_pred = smoother.step(raw_label="help", confidence=0.99, current_time=0.0)
        assert not spike_pred.is_stable
        assert not spike_pred.should_trigger_tts

        # Frames 2..5: Background or different class (not reaching 6)
        for i in range(1, 5):
            res = smoother.step(raw_label="hello", confidence=0.90, current_time=i * 0.033)
            assert not res.is_stable
            assert not res.should_trigger_tts

    def test_m_of_k_stabilization(self) -> None:
        """TC-02: Stabilizes and triggers TTS exactly when M matches are reached."""
        smoother = PredictionSmoother(
            window_size=10,
            min_matches=6,
            confidence_threshold=0.85,
            cooldown_seconds=2.0,
        )

        # Frames 1 through 5: "hello" at confidence 0.90
        for i in range(5):
            res = smoother.step(raw_label="hello", confidence=0.90, current_time=i * 0.033)
            assert not res.is_stable
            assert not res.should_trigger_tts

        # Frame 6: Reaches M=6
        res6 = smoother.step(raw_label="hello", confidence=0.90, current_time=6 * 0.033)
        assert res6.is_stable
        assert res6.should_trigger_tts
        assert res6.stable_label == "hello"
        assert res6.match_count == 6

    def test_confidence_threshold_gating(self) -> None:
        """TC-03: Consistent frames below threshold are rejected."""
        smoother = PredictionSmoother(
            window_size=10,
            min_matches=6,
            confidence_threshold=0.85,
        )

        for i in range(12):
            res = smoother.step(raw_label="hello", confidence=0.84, current_time=i * 0.033)
            assert not res.is_stable
            assert not res.should_trigger_tts
            assert res.match_count == 0

    def test_cooldown_duplicate_suppression(self) -> None:
        """TC-04: Same gesture is suppressed during cooldown and re-triggers after elapsed."""
        smoother = PredictionSmoother(
            window_size=10,
            min_matches=6,
            confidence_threshold=0.85,
            cooldown_seconds=2.0,
        )

        # Establish stable trigger at t=0.0
        for _ in range(6):
            smoother.step(raw_label="hello", confidence=0.90, current_time=0.0)

        # During cooldown (t = 0.5s, 1.0s, 1.9s)
        for t in (0.5, 1.0, 1.9):
            res = smoother.step(raw_label="hello", confidence=0.90, current_time=t)
            assert res.is_stable
            assert not res.should_trigger_tts, f"Triggered unexpectedly at t={t}"

        # After cooldown elapsed (t = 2.1s)
        res_after = smoother.step(raw_label="hello", confidence=0.90, current_time=2.1)
        assert res_after.is_stable
        assert res_after.should_trigger_tts

    def test_cooldown_different_class_immediate_trigger(self) -> None:
        """TC-05: Transitioning to a different gesture triggers immediately."""
        smoother = PredictionSmoother(
            window_size=10,
            min_matches=6,
            confidence_threshold=0.85,
            cooldown_seconds=2.0,
        )

        # Trigger "hello" at t=0.0
        for _ in range(6):
            res = smoother.step(raw_label="hello", confidence=0.90, current_time=0.0)
        assert res.should_trigger_tts

        # Transition to "yes" at t=0.5
        for _ in range(6):
            res_yes = smoother.step(raw_label="yes", confidence=0.95, current_time=0.5)

        assert res_yes.is_stable
        assert res_yes.stable_label == "yes"
        assert res_yes.should_trigger_tts

    def test_sliding_window_eviction(self) -> None:
        """TC-06: 10 new frames completely purge older history."""
        smoother = PredictionSmoother(window_size=10, min_matches=6)

        for _ in range(10):
            smoother.step(raw_label="hello", confidence=0.90)

        for _ in range(10):
            smoother.step(raw_label="no", confidence=0.90)

        counts = [lbl for lbl, _ in smoother._window if lbl == "hello"]
        assert len(counts) == 0

    def test_smoother_reset(self) -> None:
        """Reset flushes sliding deque and debounce memory."""
        smoother = PredictionSmoother(window_size=10)
        smoother.step(raw_label="hello", confidence=0.95)
        assert len(smoother._window) == 1

        smoother.reset()
        assert len(smoother._window) == 0
        assert len(smoother._last_trigger_times) == 0

    def test_smoother_invalid_init(self) -> None:
        """min_matches > window_size raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            PredictionSmoother(window_size=5, min_matches=6)


@pytest.mark.unit
class TestAsyncTTSWorker:
    """Unit tests for background speech synthesis worker and non-blocking timing."""

    def test_tts_worker_lifecycle(self) -> None:
        """Start, state check, and clean shutdown of worker thread."""
        dummy = DummyTTSEngine()
        worker = AsyncTTSWorker(engine=dummy)

        assert not worker.is_alive()
        worker.start()
        assert worker.is_alive()

        worker.say("hello")
        time.sleep(0.15)  # allow worker loop to consume
        assert "hello" in dummy.spoken_texts

        worker.stop()
        assert not worker.is_alive()

    def test_tts_worker_non_blocking_dispatch(self) -> None:
        """TC-07: say() returns in <1ms without blocking main thread."""
        dummy = DummyTTSEngine()
        worker = AsyncTTSWorker(engine=dummy)
        worker.start()

        t0 = time.perf_counter()
        success = worker.say("gesture detected")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert success
        assert elapsed_ms < 1.0, f"say() took too long: {elapsed_ms}ms"

        worker.stop()

    def test_tts_worker_queue_overflow(self) -> None:
        """TC-08: Saturated queue drops oldest messages without raising exceptions."""
        dummy = DummyTTSEngine()
        # Small queue size = 3, worker not started so queue stays full
        worker = AsyncTTSWorker(engine=dummy, max_queue_size=3)

        for i in range(10):
            worker.say(f"msg_{i}")

        assert worker._queue.qsize() == 3

        # Start worker and verify it processes without errors
        worker.start()
        time.sleep(0.2)
        worker.stop()

        assert len(dummy.spoken_texts) == 3
        # Should have newest messages (msg_7, msg_8, msg_9)
        assert dummy.spoken_texts == ["msg_7", "msg_8", "msg_9"]

    def test_tts_worker_empty_text_ignored(self) -> None:
        """Empty or whitespace text is rejected."""
        dummy = DummyTTSEngine()
        worker = AsyncTTSWorker(engine=dummy)
        worker.start()

        assert not worker.say("")
        assert not worker.say("   ")

        worker.stop()

    def test_pyttsx3_engine_fallback(self) -> None:
        """Pyttsx3Engine initializes or falls back to Dummy without error."""
        engine = Pyttsx3Engine()
        engine.say("test")
        engine.runAndWait()
        engine.stop()


@pytest.mark.unit
class TestRealtimeGesturePipeline:
    """Integration tests for end-to-end real-time inference pipeline."""

    def test_pipeline_buffering_phase(self) -> None:
        """TC-11: Frames 1..29 return buffering placeholder."""
        config = PipelineConfig(sequence_length=30)
        pipeline = RealtimeGesturePipeline(config=config, enable_tts=False)

        dummy_vector = [0.0] * 126

        for _ in range(29):
            pred = pipeline.process_landmarks(dummy_vector)
            assert pred.label == "buffering"
            assert not pred.is_stable
            assert pred.confidence == 0.0
            assert not pred.triggered_tts

        pipeline.close()

    def test_pipeline_full_window_inference(self) -> None:
        """TC-12: Frame 30 triggers neural model inference and produces valid latency."""
        config = PipelineConfig(sequence_length=30, vocabulary=["hello", "yes", "no"])
        model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=3)
        dummy_tts = DummyTTSEngine()
        tts_worker = AsyncTTSWorker(engine=dummy_tts)

        pipeline = RealtimeGesturePipeline(
            config=config,
            model=model,
            tts_worker=tts_worker,
            enable_tts=True,
        )

        dummy_vector = [0.1] * 126

        # Feed 29 frames
        for _ in range(29):
            pipeline.process_landmarks(dummy_vector)

        # Feed 30th frame -> Activates model inference
        pred30 = pipeline.process_landmarks(dummy_vector)
        assert pred30.label in config.vocabulary
        assert pred30.latency_ms > 0.0
        assert 0.0 <= pred30.confidence <= 1.0

        pipeline.close()

    def test_pipeline_synthetic_stream_integration(self) -> None:
        """TC-14: Process 40 synthetic video frames through entire pipeline end-to-end."""
        config = PipelineConfig(sequence_length=15, smoothing_window=5, smoothing_min_matches=3)
        model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=5)
        dummy_tts = DummyTTSEngine()
        tts_worker = AsyncTTSWorker(engine=dummy_tts)

        pipeline = RealtimeGesturePipeline(
            config=config,
            model=model,
            tts_worker=tts_worker,
            enable_tts=True,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        predictions = []
        for i in range(40):
            pred = pipeline.process_frame(frame, timestamp=i * 0.033)
            predictions.append(pred)

        assert len(predictions) == 40
        # Frames after 15 should have real predictions
        assert all(p.label != "buffering" for p in predictions[15:])

        pipeline.close()

    def test_pipeline_empty_frame_resilience(self) -> None:
        """TC-13: Empty or zero-sized frames do not crash pipeline."""
        pipeline = RealtimeGesturePipeline(enable_tts=False)
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)

        pred = pipeline.process_frame(empty_frame)
        assert pred.label == "buffering"

        pipeline.close()

    def test_pipeline_reset_and_context_manager(self) -> None:
        """TC-15: Verify context manager lifecycle and reset clears frame buffer."""
        config = PipelineConfig(sequence_length=10)
        dummy_vector = np.zeros(126, dtype=np.float32)

        with RealtimeGesturePipeline(config=config, enable_tts=False) as pipeline:
            for _ in range(10):
                pipeline.process_landmarks(dummy_vector)
            # After 10 frames, no longer buffering
            pred = pipeline.process_landmarks(dummy_vector)
            assert pred.label != "buffering"

            pipeline.reset()
            # After reset, buffer is empty and should return buffering
            pred_reset = pipeline.process_landmarks(dummy_vector)
            assert pred_reset.label == "buffering"

    def test_pipeline_run_with_capture_source(self) -> None:
        """TC-16: Verify pipeline.run() over a synthetic capture source stream."""
        from sign_translator.collect_data import SyntheticCaptureSource

        config = PipelineConfig(sequence_length=10)
        source = SyntheticCaptureSource(max_frames=15)
        pipeline = RealtimeGesturePipeline(
            config=config,
            capture_source=source,
            gesture_classes=["hello", "thank_you", "yes", "no"],
            enable_tts=False,
        )

        results = pipeline.run(max_frames=15)
        assert len(results) == 15
        assert results[0].label == "buffering"
        assert results[-1].label in ["hello", "thank_you", "yes", "no"]
        pipeline.close()

    def test_pipeline_run_without_source_raises(self) -> None:
        """TC-17: Calling run() without a configured capture source raises ValueError."""
        pipeline = RealtimeGesturePipeline(enable_tts=False)
        with pytest.raises(ValueError, match="No capture_source provided"):
            pipeline.run(max_frames=5)
        pipeline.close()
