"""Unit tests verifying headless synthetic test fixtures and zero hardware dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sign_translator.schemas import PipelineConfig


@pytest.mark.unit
def test_sample_synthetic_frame(sample_synthetic_frame: np.ndarray) -> None:
    """Verify synthetic frame dimensions, data type, and zero-initialization."""
    assert sample_synthetic_frame.shape == (480, 640, 3)
    assert sample_synthetic_frame.dtype == np.uint8
    assert np.all(sample_synthetic_frame == 0)


@pytest.mark.unit
def test_sample_synthetic_frame_with_box(sample_synthetic_frame_with_box: np.ndarray) -> None:
    """Verify synthetic frame with contrast pattern."""
    assert sample_synthetic_frame_with_box.shape == (480, 640, 3)
    assert sample_synthetic_frame_with_box.dtype == np.uint8
    assert sample_synthetic_frame_with_box[150, 150, 0] == 255
    assert sample_synthetic_frame_with_box[0, 0, 0] == 128


@pytest.mark.unit
def test_sample_feature_vector(sample_feature_vector: list[float]) -> None:
    """Verify feature vector length and values."""
    assert len(sample_feature_vector) == 126
    assert all(isinstance(x, float) for x in sample_feature_vector)


@pytest.mark.unit
def test_sample_sequence_tensor(sample_sequence_tensor: np.ndarray) -> None:
    """Verify sequence tensor shape and dtype."""
    assert sample_sequence_tensor.shape == (30, 126)
    assert sample_sequence_tensor.dtype == np.float32


@pytest.mark.unit
def test_default_pipeline_config(default_pipeline_config: PipelineConfig) -> None:
    """Verify default pipeline config fixture."""
    assert default_pipeline_config.feature_dim == 126
    assert default_pipeline_config.sequence_length == 30


@pytest.mark.unit
def test_mock_cv2_capture(mock_cv2_capture: MagicMock) -> None:
    """Verify mock video capture yields synthetic frame without physical camera."""
    assert mock_cv2_capture.isOpened() is True
    ret, frame = mock_cv2_capture.read()
    assert ret is True
    assert frame.shape == (480, 640, 3)
    mock_cv2_capture.release()
    assert mock_cv2_capture.release.called


@pytest.mark.unit
def test_mock_tts_engine(mock_tts_engine: MagicMock) -> None:
    """Verify mock TTS engine responds without physical sound card."""
    mock_tts_engine.say("Hello")
    mock_tts_engine.runAndWait()
    assert mock_tts_engine.say.called
    assert mock_tts_engine.runAndWait.called
