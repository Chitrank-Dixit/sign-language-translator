"""Real-time gesture recognition, temporal smoothing, and non-blocking speech synthesis.

Integrates sliding-window inference, M-of-K majority voting, per-label cooldown
debouncing, and asynchronous background TTS dispatch.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from sign_translator.collect_data import FrameCaptureSource, HandLandmarkExtractor
from sign_translator.device import resolve_ort_providers, resolve_torch_device
from sign_translator.schemas import (
    GesturePrediction,
    PipelineConfig,
    SmoothedPrediction,
    TTSMessage,
)


@runtime_checkable
class TTSEngine(Protocol):
    """Protocol defining interface for text-to-speech audio engines."""

    def say(self, text: str) -> None:
        """Queue text for vocalization."""
        ...

    def runAndWait(self) -> None:
        """Run speech loop until speech queue is empty."""
        ...

    def stop(self) -> None:
        """Halt speech playback."""
        ...


class DummyTTSEngine:
    """Headless mock TTS engine recording utterances in-memory without audio hardware."""

    def __init__(self) -> None:
        self.spoken_texts: list[str] = []

    def say(self, text: str) -> None:
        self.spoken_texts.append(text)

    def runAndWait(self) -> None:
        pass

    def stop(self) -> None:
        pass


class Pyttsx3Engine:
    """Standard pyttsx3 offline TTS engine wrapper with headless fallback."""

    def __init__(self, rate: int = 150) -> None:
        try:
            import pyttsx3

            self._engine: Any = pyttsx3.init()
            self._engine.setProperty("rate", rate)
        except Exception:
            self._engine = DummyTTSEngine()

    def say(self, text: str) -> None:
        self._engine.say(text)

    def runAndWait(self) -> None:
        self._engine.runAndWait()

    def stop(self) -> None:
        self._engine.stop()


class AsyncTTSWorker:
    """Decoupled background worker for non-blocking speech synthesis."""

    def __init__(
        self,
        engine: TTSEngine | None = None,
        max_queue_size: int = 10,
    ) -> None:
        self.engine: TTSEngine = engine if engine is not None else DummyTTSEngine()
        self._queue: queue.Queue[TTSMessage] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> AsyncTTSWorker:
        """Start background speech dispatch thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        return self

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.engine.say(msg.text)
                self.engine.runAndWait()
            except Exception:
                pass

    def say(self, text: str) -> bool:
        """Enqueue utterance string without blocking video frame loop."""
        if not text or not text.strip():
            return False
        msg = TTSMessage(text=text.strip())
        if self._queue.full():
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(msg)
        return True

    def is_alive(self) -> bool:
        """Return True if background worker thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> None:
        """Terminate background worker thread and close engine."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        with contextlib.suppress(Exception):
            self.engine.stop()

    def __enter__(self) -> AsyncTTSWorker:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class PredictionSmoother:
    """Sliding-window M-of-K temporal voting, confidence gating, and cooldown debouncer."""

    def __init__(
        self,
        window_size: int = 10,
        min_matches: int = 6,
        confidence_threshold: float = 0.85,
        cooldown_seconds: float = 2.0,
    ) -> None:
        if min_matches > window_size:
            raise ValueError(
                f"min_matches ({min_matches}) cannot exceed window_size ({window_size})"
            )

        self.window_size = window_size
        self.min_matches = min_matches
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds

        self._window: deque[tuple[str | None, float]] = deque(maxlen=window_size)
        self._last_trigger_times: dict[str, float] = {}

    def step(
        self,
        raw_label: str,
        confidence: float,
        current_time: float | None = None,
    ) -> SmoothedPrediction:
        """Process incoming raw frame prediction through M-of-K voting filter."""
        now = time.monotonic() if current_time is None else current_time

        if confidence >= self.confidence_threshold:
            self._window.append((raw_label, confidence))
        else:
            self._window.append((None, confidence))

        valid_entries = [(lbl, conf) for lbl, conf in self._window if lbl is not None]
        counts = Counter(lbl for lbl, _ in valid_entries)

        if counts:
            candidate_label, match_count = counts.most_common(1)[0]
            if match_count >= self.min_matches:
                is_stable = True
                matching_confs = [conf for lbl, conf in valid_entries if lbl == candidate_label]
                stable_conf = float(sum(matching_confs) / len(matching_confs))

                last_trigger = self._last_trigger_times.get(candidate_label, -float("inf"))
                if (now - last_trigger) >= self.cooldown_seconds:
                    should_trigger_tts = True
                    self._last_trigger_times[candidate_label] = now
                else:
                    should_trigger_tts = False

                return SmoothedPrediction(
                    raw_label=raw_label,
                    raw_confidence=confidence,
                    stable_label=candidate_label,
                    stable_confidence=stable_conf,
                    is_stable=is_stable,
                    should_trigger_tts=should_trigger_tts,
                    match_count=match_count,
                )

        return SmoothedPrediction(
            raw_label=raw_label,
            raw_confidence=confidence,
            stable_label=raw_label,
            stable_confidence=confidence,
            is_stable=False,
            should_trigger_tts=False,
            match_count=0,
        )

    def reset(self) -> None:
        """Clear temporal window buffer and reset debounce timers."""
        self._window.clear()
        self._last_trigger_times.clear()


@runtime_checkable
class SequenceModelRunner(Protocol):
    """Protocol for sequence model inference backends."""

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """Predict class probability distribution for temporal sequence.

        Args:
            sequence: Array of shape (T, feature_dim).

        Returns:
            Probability array of shape (num_classes,).
        """
        ...


class PyTorchModelRunner:
    """PyTorch execution backend for sequence classification models."""

    def __init__(self, model: nn.Module, device: str = "auto") -> None:
        self.model = model
        self.device = resolve_torch_device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(sequence).unsqueeze(0).float().to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs


class ONNXModelRunner:
    """High-performance ONNX Runtime execution engine for sequence classification models."""

    def __init__(
        self,
        model_path: Path | str,
        num_threads: int = 2,
        inter_op_threads: int = 1,
        execution_provider: str = "auto",
        graph_optimization_level: str = "all",
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = inter_op_threads

        opt_levels = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }
        sess_options.graph_optimization_level = opt_levels.get(
            graph_optimization_level.lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        target_provider = device if device is not None else execution_provider
        providers = resolve_ort_providers(target_provider)

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options, providers=providers
            )
        except Exception:
            # Fallback to CPU if provider initialization failed
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options, providers=providers
            )

        self.active_provider: str = (
            self.session.get_providers()[0]
            if self.session.get_providers()
            else "CPUExecutionProvider"
        )
        self.device: str = "cuda:0" if "CUDA" in self.active_provider else "cpu"
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.expected_feature_dim: int | None = (
            input_shape[-1] if len(input_shape) >= 3 and isinstance(input_shape[-1], int) else None
        )

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """Predict class probability distribution for temporal sequence.

        Args:
            sequence: Array of shape (T, feature_dim) or (batch, T, feature_dim).

        Returns:
            Probability array of shape (num_classes,) or (batch, num_classes).
        """
        arr = np.asarray(sequence, dtype=np.float32)
        if self.expected_feature_dim is not None and arr.shape[-1] != self.expected_feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.expected_feature_dim}, got {arr.shape[-1]}"
            )

        is_batched = arr.ndim == 3
        if not is_batched:
            if arr.ndim != 2:
                raise ValueError(f"Expected sequence of 2 or 3 dimensions, got shape {arr.shape}")
            arr = np.expand_dims(arr, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: arr})
        logits = outputs[0]

        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs if is_batched else probs[0]


class RealtimeGesturePipeline:
    """End-to-end real-time sign language and gesture recognition pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: nn.Module | SequenceModelRunner | Path | str | None = None,
        landmark_extractor: HandLandmarkExtractor | None = None,
        capture_source: FrameCaptureSource | None = None,
        tts_worker: AsyncTTSWorker | None = None,
        smoother: PredictionSmoother | None = None,
        gesture_classes: list[str] | None = None,
        enable_tts: bool = True,
    ) -> None:
        self.config = config or PipelineConfig()
        self.vocabulary = (
            list(gesture_classes) if gesture_classes is not None else list(self.config.vocabulary)
        )
        self.capture_source = capture_source

        if isinstance(model, nn.Module):
            self.model_runner: SequenceModelRunner = PyTorchModelRunner(
                model, device=self.config.device
            )
        elif isinstance(model, (str, Path)):
            self.model_runner = ONNXModelRunner(model, execution_provider=self.config.device)
        elif model is not None:
            self.model_runner = model
        elif self.config.use_onnx:
            self.model_runner = ONNXModelRunner(
                model_path=self.config.onnx_model_path,
                num_threads=self.config.onnx_intra_op_threads,
                inter_op_threads=self.config.onnx_inter_op_threads,
                execution_provider=self.config.device,
                graph_optimization_level=self.config.onnx_graph_optimization_level,
            )
        else:
            from sign_translator.architecture import build_model

            nn_model = build_model(self.config)
            self.model_runner = PyTorchModelRunner(nn_model, device=self.config.device)

        self.extractor = landmark_extractor or HandLandmarkExtractor(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            handedness_aware=True,
            normalize=True,
            model_path=getattr(self.config, "mediapipe_model_path", None),
        )

        self.smoother = smoother or PredictionSmoother(
            window_size=self.config.smoothing_window,
            min_matches=self.config.smoothing_min_matches,
            confidence_threshold=self.config.confidence_threshold,
            cooldown_seconds=self.config.cooldown_seconds,
        )

        self.enable_tts = enable_tts
        self.tts_worker = tts_worker or AsyncTTSWorker()
        if enable_tts and not self.tts_worker.is_alive():
            self.tts_worker.start()

        self._frame_buffer: deque[list[float]] = deque(maxlen=self.config.sequence_length)

    def process_landmarks(
        self,
        landmarks: list[float] | np.ndarray,
        timestamp: float | None = None,
    ) -> GesturePrediction:
        """Process pre-extracted landmark vector through pipeline."""
        t_start = time.perf_counter()
        features = landmarks.tolist() if isinstance(landmarks, np.ndarray) else list(landmarks)
        self._frame_buffer.append(features)

        # Buffering phase before sequence length is reached
        if len(self._frame_buffer) < self.config.sequence_length:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return GesturePrediction(
                label="buffering",
                confidence=0.0,
                is_stable=False,
                latency_ms=latency_ms,
                triggered_tts=False,
            )

        seq_arr = np.array(self._frame_buffer, dtype=np.float32)
        probs = self.model_runner.predict_proba(seq_arr)
        pred_idx = int(np.argmax(probs))
        raw_conf = float(probs[pred_idx])
        raw_label = self.vocabulary[pred_idx] if pred_idx < len(self.vocabulary) else "unknown"

        smoothed = self.smoother.step(raw_label, raw_conf, current_time=timestamp)

        triggered_tts = False
        if smoothed.should_trigger_tts and self.enable_tts:
            self.tts_worker.say(smoothed.stable_label)
            triggered_tts = True

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return GesturePrediction(
            label=smoothed.stable_label if smoothed.is_stable else raw_label,
            confidence=smoothed.stable_confidence if smoothed.is_stable else raw_conf,
            is_stable=smoothed.is_stable,
            latency_ms=latency_ms,
            triggered_tts=triggered_tts,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> GesturePrediction:
        """Extract landmarks from raw video frame and run through inference pipeline."""
        landmark_frame = self.extractor.extract(frame)
        return self.process_landmarks(landmark_frame.features, timestamp=timestamp)

    def run(
        self,
        max_frames: int | None = None,
        capture_source: FrameCaptureSource | None = None,
    ) -> list[GesturePrediction]:
        """Run continuous inference loop over a video capture source.

        Args:
            max_frames: Optional upper limit on processed frames.
            capture_source: Video source; defaults to self.capture_source.

        Returns:
            List of GesturePrediction results.
        """
        source = capture_source or self.capture_source
        if source is None:
            raise ValueError("No capture_source provided to RealtimeGesturePipeline.")

        predictions: list[GesturePrediction] = []
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = source.read()
            if not ret or frame is None:
                break
            timestamp = time.perf_counter()
            pred = self.process_frame(frame, timestamp=timestamp)
            predictions.append(pred)
            frame_idx += 1

        return predictions

    def reset(self) -> None:
        """Reset temporal frame buffer and prediction smoother."""
        self._frame_buffer.clear()
        self.smoother.reset()

    def close(self) -> None:
        """Release pipeline extractor and background worker resources."""
        if self.capture_source is not None:
            self.capture_source.release()
        self.extractor.close()
        self.tts_worker.stop()

    def __enter__(self) -> RealtimeGesturePipeline:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
