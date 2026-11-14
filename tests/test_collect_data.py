"""Unit tests for data collection, frame ingestion, and dataset recording."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sign_translator.collect_data import (
    HandLandmarkExtractor,
    OpenCVCaptureSource,
    SequenceRecorder,
    SyntheticCaptureSource,
    ThreadedFrameReader,
)
from sign_translator.schemas import DataCollectionConfig, SequenceMetadata


@pytest.mark.unit
class TestSyntheticCaptureSource:
    """Tests for SyntheticCaptureSource mock video feed."""

    def test_synthetic_source_default_black_frames(self) -> None:
        """Verify synthetic capture produces 640x480 black frames."""
        source = SyntheticCaptureSource(width=640, height=480, max_frames=3)
        assert source.is_opened()

        for _ in range(3):
            ret, frame = source.read()
            assert ret is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8

        # Exhausted
        ret, frame = source.read()
        assert ret is False
        assert frame is None

        source.release()
        assert not source.is_opened()

    def test_synthetic_source_injected_frames(self) -> None:
        """Verify synthetic capture cycles through injected frame buffers."""
        f1 = np.ones((100, 100, 3), dtype=np.uint8) * 10
        f2 = np.ones((100, 100, 3), dtype=np.uint8) * 20
        source = SyntheticCaptureSource(frames=[f1, f2], max_frames=4)

        ret1, frame1 = source.read()
        ret2, frame2 = source.read()
        ret3, frame3 = source.read()

        assert ret1 and np.array_equal(frame1, f1)
        assert ret2 and np.array_equal(frame2, f2)
        assert ret3 and np.array_equal(frame3, f1)

        source.release()


@pytest.mark.unit
class TestThreadedFrameReader:
    """Tests for background threaded frame reader and queue management."""

    def test_threaded_reader_drop_and_retrieve(self) -> None:
        """Verify worker delivers frames and drops stale items to prevent lag."""
        source = SyntheticCaptureSource(width=320, height=240, max_frames=10)
        reader = ThreadedFrameReader(source, max_queue_size=2)
        reader.start()

        ret, frame = reader.read(timeout=1.0)
        assert ret is True
        assert frame is not None
        assert frame.shape == (240, 320, 3)

        reader.stop()
        assert not source.is_opened()


@pytest.mark.unit
class TestHandLandmarkExtractor:
    """Tests for MediaPipe landmark extraction, slot assignment, and zero-padding."""

    def test_extract_empty_or_black_frame(self) -> None:
        """Verify black frame produces 126 zeros and hands_detected=0."""
        extractor = HandLandmarkExtractor(normalize=True)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = extractor.extract(blank, frame_index=0)
        assert frame.frame_index == 0
        assert len(frame.features) == 126
        assert frame.hands_detected == 0
        assert frame.is_normalized is True
        assert all(f == 0.0 for f in frame.features)

        extractor.close()

    def test_extractor_context_manager(self) -> None:
        """Verify extractor works safely within context manager."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        with HandLandmarkExtractor() as extractor:
            frame = extractor.extract(blank, frame_index=1)
            assert frame.frame_index == 1

    def test_mock_mediapipe_handedness_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TC-16: Verify deterministic slotting: Left in slot 0, Right in slot 1."""
        extractor = HandLandmarkExtractor(handedness_aware=True, normalize=False)

        class MockLandmark:
            def __init__(self, val: float) -> None:
                self.x = val
                self.y = val
                self.z = val

        hand1 = [MockLandmark(1.0) for _ in range(21)]
        hand2 = [MockLandmark(2.0) for _ in range(21)]

        monkeypatch.setattr(
            extractor,
            "_detect_hands",
            lambda _: ([hand1, hand2], ["Right", "Left"]),
        )

        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8)
        result = extractor.extract(dummy_frame)

        features = np.array(result.features).reshape(2, 21, 3)

        # Left hand was labeled for hand2 (val 2.0), so slot 0 should have 2.0
        assert np.allclose(features[0], 2.0)
        # Right hand was labeled for hand1 (val 1.0), so slot 1 should have 1.0
        assert np.allclose(features[1], 1.0)
        assert result.hands_detected == 2

        extractor.close()


@pytest.mark.unit
class TestSequenceRecorder:
    """Tests for sequence recording and dual file persistence."""

    def test_record_and_save_sequence(self, tmp_path: Path) -> None:
        """TC-17: Record a sequence, save .npy and .json, and verify file roundtrip."""
        source = SyntheticCaptureSource(width=320, height=240, max_frames=50)
        config = DataCollectionConfig(sequence_length=15, output_dir=str(tmp_path))
        recorder = SequenceRecorder(config=config)

        sequence, metadata = recorder.record_sequence(
            source=source,
            gesture_label="hello",
            target_frames=15,
        )

        assert sequence.shape == (15, 126)
        assert sequence.dtype == np.float32
        assert metadata.gesture_label == "hello"
        assert metadata.num_frames == 15
        assert metadata.normalized is True

        # Save to temporary path
        npy_path, json_path = recorder.save_sequence(
            sequence=sequence,
            metadata=metadata,
            output_dir=tmp_path,
        )

        assert npy_path.exists()
        assert json_path.exists()
        assert npy_path.name.endswith(".npy")
        assert json_path.name.endswith(".json")

        # Verify .npy loadable
        loaded_seq = np.load(npy_path)
        assert loaded_seq.shape == (15, 126)
        assert np.allclose(loaded_seq, sequence)

        # Verify .json loadable into SequenceMetadata
        content = json_path.read_text(encoding="utf-8")
        loaded_meta = SequenceMetadata.model_validate_json(content)
        assert loaded_meta.gesture_label == "hello"
        assert loaded_meta.num_frames == 15
        assert loaded_meta.sequence_id == metadata.sequence_id

    def test_opencv_capture_source_interface(self) -> None:
        """Verify OpenCVCaptureSource adheres to FrameCaptureSource protocol."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap.read.return_value = (False, None)

        source = OpenCVCaptureSource(camera_index=0)
        source._cap = mock_cap

        assert not source.is_opened()
        ret, frame = source.read()
        assert ret is False
        assert frame is None

        source.release()
        mock_cap.release.assert_called_once()
