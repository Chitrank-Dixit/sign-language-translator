"""Shared pytest fixtures and synthetic data generators for headless testing.

Ensures tests never access physical webcams, GPU hardware, displays, or audio drivers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sign_translator.schemas import PipelineConfig


@pytest.fixture
def sample_synthetic_frame() -> np.ndarray:
    """Generates a standard 640x480 3-channel black synthetic frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_synthetic_frame_with_box() -> np.ndarray:
    """Generates a synthetic frame with a high-contrast pattern for visual testing."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame[100:200, 100:200] = 255
    return frame


@pytest.fixture
def sample_feature_vector() -> list[float]:
    """Generates a 126-dimensional flat list of synthetic coordinates (2 hands x 21 x 3)."""
    return [0.0] * 126


@pytest.fixture
def sample_sequence_tensor() -> np.ndarray:
    """Generates a (30, 126) temporal sequence of synthetic normalized landmarks."""
    return np.zeros((30, 126), dtype=np.float32)


@pytest.fixture
def default_pipeline_config() -> PipelineConfig:
    """Provides a pristine default PipelineConfig instance."""
    return PipelineConfig()


@pytest.fixture
def mock_cv2_capture(sample_synthetic_frame: np.ndarray) -> MagicMock:
    """Provides a mocked cv2.VideoCapture returning synthetic frames."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, sample_synthetic_frame.copy())
    mock_cap.set.return_value = True
    mock_cap.release.return_value = None
    return mock_cap


@pytest.fixture
def mock_tts_engine() -> MagicMock:
    """Provides a mocked pyttsx3 engine avoiding physical audio hardware access."""
    engine = MagicMock()
    engine.say.return_value = None
    engine.runAndWait.return_value = None
    engine.setProperty.return_value = None
    return engine
