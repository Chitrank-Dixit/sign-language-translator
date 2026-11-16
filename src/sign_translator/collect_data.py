"""Data collection and landmark extraction pipeline for sign language gestures.

Integrates MediaPipe Hands with a decoupled capture architecture, deterministic
handedness slot assignment, scale/translation normalization, and paired
tensor/metadata persistence.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from pathlib import Path
from typing import Any, Protocol

import cv2
import mediapipe as mp
import numpy as np

from sign_translator.normalize import normalize_landmarks
from sign_translator.schemas import DataCollectionConfig, LandmarkFrame, SequenceMetadata


class FrameCaptureSource(Protocol):
    """Protocol defining interface for video capture sources."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read next video frame.

        Returns:
            Tuple of (success_flag, frame_bgr_or_none).
        """
        ...

    def release(self) -> None:
        """Release capture device resources."""
        ...

    def is_opened(self) -> bool:
        """Return True if capture stream is active and opened."""
        ...


class OpenCVCaptureSource:
    """Standard OpenCV VideoCapture source wrapper supporting webcams or video files."""

    def __init__(
        self,
        source: int | str | Path | None = None,
        camera_index: int | None = None,
    ) -> None:
        if camera_index is not None and source is None:
            source = camera_index
        elif source is None:
            source = 0
        self.source = source
        source_arg: Any = str(source) if isinstance(source, (str, Path)) else source
        self._cap = cv2.VideoCapture(source_arg)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        return ret, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def is_opened(self) -> bool:
        return bool(self._cap.isOpened()) if self._cap is not None else False


class SyntheticCaptureSource:
    """Mock synthetic frame generator for isolated headless testing."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        max_frames: int | None = None,
        frames: list[np.ndarray] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self._frames = frames
        self._frame_count = 0
        self._is_opened = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._is_opened:
            return False, None
        if self.max_frames is not None and self._frame_count >= self.max_frames:
            return False, None

        if self._frames:
            frame = self._frames[self._frame_count % len(self._frames)]
        else:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._frame_count += 1
        return True, frame

    def release(self) -> None:
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened


class ThreadedFrameReader:
    """Decoupled background video ingestion worker preventing frame latency build-up."""

    def __init__(self, source: FrameCaptureSource, max_queue_size: int = 2) -> None:
        self._source = source
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> ThreadedFrameReader:
        """Start background frame ingestion worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._source.read()
            if not ret or frame is None:
                continue
            if self._queue.full():
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(frame)

    def read(self, timeout: float = 0.5) -> tuple[bool, np.ndarray | None]:
        """Retrieve most recent frame without blocking video FPS."""
        try:
            frame = self._queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        """Stop worker thread and release underlying source."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._source.release()


class HandLandmarkExtractor:
    """MediaPipe Hands wrapper with deterministic handedness slots and zero-padding."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        handedness_aware: bool = True,
        normalize: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.handedness_aware = handedness_aware
        self.normalize = normalize
        if model_path:
            self.model_path = Path(model_path)
        elif Path("models/hand_landmarker.task").exists():
            self.model_path = Path("models/hand_landmarker.task")
        else:
            self.model_path = None
        self._landmarker: Any = None

        if self.model_path and self.model_path.exists():
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)

    def _detect_hands(self, rgb_frame: np.ndarray) -> tuple[list[Any], list[str]]:
        """Run landmark detection on RGB frame. Returns (landmarks_list, labels_list)."""
        if self._landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self._landmarker.detect(mp_image)
            hands_lms = detection_result.hand_landmarks or []
            labels: list[str] = []
            if detection_result.handedness:
                for h in detection_result.handedness:
                    labels.append(h[0].category_name if h else "Unknown")
            return hands_lms, labels
        return [], []

    def extract(self, frame: np.ndarray, frame_index: int = 0) -> LandmarkFrame:
        """Extract 126-D hand landmarks from a BGR image frame.

        Args:
            frame: BGR image array (H, W, 3).
            frame_index: Zero-based sequence frame index.

        Returns:
            Validated LandmarkFrame with 126 features and normalization flag.
        """
        raw_landmarks = np.zeros((2, 21, 3), dtype=np.float32)

        if frame is None or frame.size == 0:
            return LandmarkFrame(
                frame_index=frame_index,
                features=[0.0] * 126,
                hands_detected=0,
                is_normalized=self.normalize,
            )

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_lms, labels = self._detect_hands(rgb_frame)

        hands_detected = min(len(hands_lms), 2)
        for i in range(hands_detected):
            hand = hands_lms[i]
            slot = i
            if self.handedness_aware and i < len(labels):
                slot = 0 if labels[i].lower() == "left" else 1

            for j in range(min(len(hand), 21)):
                lm = hand[j]
                x = getattr(lm, "x", lm[0] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                y = getattr(lm, "y", lm[1] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                z = getattr(lm, "z", lm[2] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                raw_landmarks[slot, j, 0] = float(x)
                raw_landmarks[slot, j, 1] = float(y)
                raw_landmarks[slot, j, 2] = float(z)

        processed = normalize_landmarks(raw_landmarks) if self.normalize else raw_landmarks
        features = processed.flatten().tolist()

        return LandmarkFrame(
            frame_index=frame_index,
            features=features,
            hands_detected=hands_detected,
            is_normalized=self.normalize,
        )

    def close(self) -> None:
        """Close MediaPipe resources."""
        if self._landmarker is not None and hasattr(self._landmarker, "close"):
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandLandmarkExtractor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class SequenceRecorder:
    """Orchestrates temporal sequence collection and dual file persistence."""

    def __init__(
        self,
        extractor: HandLandmarkExtractor | None = None,
        config: DataCollectionConfig | None = None,
    ) -> None:
        self.config = config or DataCollectionConfig()
        self.extractor = extractor or HandLandmarkExtractor(
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            handedness_aware=self.config.handedness_aware,
            normalize=self.config.normalize,
        )

    def record_sequence(
        self,
        source: FrameCaptureSource,
        gesture_label: str,
        target_frames: int | None = None,
    ) -> tuple[np.ndarray, SequenceMetadata]:
        """Record a fixed-length sequence of landmark frames.

        Args:
            source: Active frame capture source.
            gesture_label: Target sign/gesture class name.
            target_frames: Optional override for sequence length.

        Returns:
            Tuple of (sequence_array of shape (T, 126), validated SequenceMetadata).
        """
        n_frames = target_frames or self.config.sequence_length
        frames_features: list[list[float]] = []

        for frame_idx in range(n_frames):
            ret, frame = source.read()
            if not ret or frame is None:
                # Missing frame fallback to zero landmarks
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            landmark_frame = self.extractor.extract(frame, frame_index=frame_idx)
            frames_features.append(landmark_frame.features)

        sequence = np.array(frames_features, dtype=np.float32)
        metadata = SequenceMetadata(
            gesture_label=gesture_label,
            num_frames=n_frames,
            feature_dim=self.config.feature_dim,
            normalized=self.config.normalize,
            fps=self.config.fps,
        )
        return sequence, metadata

    def save_sequence(
        self,
        sequence: np.ndarray,
        metadata: SequenceMetadata,
        output_dir: Path | str | None = None,
    ) -> tuple[Path, Path]:
        """Persist recorded sequence as paired .npy and .json metadata files.

        Args:
            sequence: Array of shape (T, 126).
            metadata: Associated SequenceMetadata model.
            output_dir: Destination base directory.

        Returns:
            Tuple of (saved_npy_path, saved_json_path).
        """
        base_dir = Path(output_dir or self.config.output_dir)
        target_dir = base_dir / metadata.gesture_label
        target_dir.mkdir(parents=True, exist_ok=True)

        npy_path = target_dir / f"seq_{metadata.sequence_id}.npy"
        json_path = target_dir / f"seq_{metadata.sequence_id}.json"

        np.save(npy_path, sequence)
        json_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

        return npy_path, json_path
