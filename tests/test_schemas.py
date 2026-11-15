"""Unit tests for domain schemas and configuration contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sign_translator.schemas import (
    DataCollectionConfig,
    GesturePrediction,
    LandmarkFrame,
    ModelArchitecture,
    PipelineConfig,
    SequenceMetadata,
    SmoothedPrediction,
    TrainingConfig,
    TrainResult,
    TTSMessage,
)


@pytest.mark.unit
class TestPipelineConfig:
    """Test suite for PipelineConfig settings contract."""

    def test_pipeline_config_defaults(self) -> None:
        """TC-01: Verify default settings instantiation and computed attributes."""
        config = PipelineConfig()
        assert config.sequence_length == 30
        assert config.num_hands == 2
        assert config.landmarks_per_hand == 21
        assert config.coords == 3
        assert config.feature_dim == 126
        assert config.vocabulary == ["hello", "thank_you", "yes", "no", "help"]
        assert config.architecture == ModelArchitecture.LSTM
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.dropout == 0.2
        assert config.smoothing_window == 10
        assert config.smoothing_min_matches == 6
        assert config.confidence_threshold == 0.85
        assert config.cooldown_seconds == 2.0

    def test_pipeline_config_custom_valid(self) -> None:
        """TC-02: Verify valid custom parameter overrides."""
        config = PipelineConfig(
            sequence_length=45,
            confidence_threshold=0.9,
            smoothing_window=15,
            smoothing_min_matches=10,
            architecture=ModelArchitecture.TRANSFORMER,
            vocabulary=["open", "close", "pinch"],
        )
        assert config.sequence_length == 45
        assert config.confidence_threshold == 0.9
        assert config.smoothing_window == 15
        assert config.smoothing_min_matches == 10
        assert config.architecture == ModelArchitecture.TRANSFORMER
        assert config.vocabulary == ["open", "close", "pinch"]

    def test_confidence_threshold_upper_bound(self) -> None:
        """TC-03: Verify confidence threshold rejection when > 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(confidence_threshold=1.05)
        assert "confidence_threshold" in str(exc_info.value)

    def test_confidence_threshold_lower_bound(self) -> None:
        """TC-04: Verify confidence threshold rejection when < 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(confidence_threshold=-0.01)
        assert "confidence_threshold" in str(exc_info.value)

    def test_smoothing_window_relationship_violation(self) -> None:
        """TC-05: Verify error when smoothing_min_matches > smoothing_window."""
        with pytest.raises(ValidationError, match="cannot exceed"):
            PipelineConfig(smoothing_window=10, smoothing_min_matches=12)

    def test_feature_dim_mismatch(self) -> None:
        """TC-06: Verify error when feature_dim does not match dimensional product."""
        with pytest.raises(ValidationError, match="feature_dim .* must equal"):
            PipelineConfig(
                num_hands=2,
                landmarks_per_hand=21,
                coords=3,
                feature_dim=100,
            )

    def test_vocabulary_duplicate_classes(self) -> None:
        """TC-07: Verify rejection of duplicate classes in vocabulary."""
        with pytest.raises(ValidationError, match="duplicate gesture classes"):
            PipelineConfig(vocabulary=["hello", "help", "hello"])

    def test_vocabulary_insufficient_size(self) -> None:
        """TC-08: Verify rejection of vocabulary with fewer than 2 classes."""
        with pytest.raises(ValidationError):
            PipelineConfig(vocabulary=["single_class"])

    def test_vocabulary_empty_or_whitespace_items(self) -> None:
        """TC-09: Verify rejection of empty or whitespace-only vocabulary entries."""
        with pytest.raises(ValidationError):
            PipelineConfig(vocabulary=["hello", "   "])

    def test_environment_variable_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TC-10: Verify environment variable overrides with SIGN_TR_ prefix."""
        monkeypatch.setenv("SIGN_TR_SEQUENCE_LENGTH", "40")
        monkeypatch.setenv("SIGN_TR_CONFIDENCE_THRESHOLD", "0.95")
        config = PipelineConfig()
        assert config.sequence_length == 40
        assert config.confidence_threshold == 0.95


@pytest.mark.unit
class TestSequenceMetadata:
    """Test suite for SequenceMetadata contract."""

    def test_sequence_metadata_nominal(self) -> None:
        """TC-11: Verify nominal creation with valid defaults."""
        meta = SequenceMetadata(gesture_label="thank_you")
        assert meta.gesture_label == "thank_you"
        assert meta.num_frames == 30
        assert meta.feature_dim == 126
        assert meta.normalized is True
        assert meta.fps == 30.0
        assert len(meta.sequence_id) > 0
        assert meta.timestamp.tzinfo is not None

    def test_sequence_metadata_immutability(self) -> None:
        """TC-12: Verify frozen immutability constraint."""
        meta = SequenceMetadata(gesture_label="thank_you")
        with pytest.raises(ValidationError):
            meta.gesture_label = "hello"  # type: ignore[misc]

    def test_sequence_metadata_empty_label(self) -> None:
        """TC-13: Verify rejection of whitespace or empty gesture label."""
        with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
            SequenceMetadata(gesture_label="   ")

    def test_sequence_metadata_invalid_num_frames(self) -> None:
        """TC-14: Verify rejection of num_frames < 1."""
        with pytest.raises(ValidationError):
            SequenceMetadata(gesture_label="yes", num_frames=0)


@pytest.mark.unit
class TestGesturePrediction:
    """Test suite for GesturePrediction contract."""

    def test_gesture_prediction_nominal_and_json_roundtrip(self) -> None:
        """TC-15: Verify nominal creation and lossless JSON serialization roundtrip."""
        pred = GesturePrediction(
            label="hello",
            confidence=0.92,
            is_stable=True,
            latency_ms=4.5,
        )
        assert pred.label == "hello"
        assert pred.confidence == 0.92
        assert pred.is_stable is True
        assert pred.latency_ms == 4.5

        # Serialization roundtrip
        json_data = pred.model_dump_json()
        parsed = json.loads(json_data)
        assert parsed["label"] == "hello"
        assert parsed["confidence"] == 0.92

        restored = GesturePrediction.model_validate_json(json_data)
        assert restored.label == pred.label
        assert restored.confidence == pred.confidence
        assert restored.latency_ms == pred.latency_ms

    def test_gesture_prediction_confidence_bounds(self) -> None:
        """TC-16: Verify rejection of out-of-range confidence scores."""
        with pytest.raises(ValidationError):
            GesturePrediction(label="hello", confidence=1.2, latency_ms=2.0)
        with pytest.raises(ValidationError):
            GesturePrediction(label="hello", confidence=-0.1, latency_ms=2.0)

    def test_gesture_prediction_negative_latency(self) -> None:
        """TC-17: Verify rejection of negative latency."""
        with pytest.raises(ValidationError):
            GesturePrediction(label="hello", confidence=0.9, latency_ms=-1.0)

    def test_gesture_prediction_immutability(self) -> None:
        """TC-18: Verify frozen immutability constraint."""
        pred = GesturePrediction(label="hello", confidence=0.9, latency_ms=2.0)
        with pytest.raises(ValidationError):
            pred.is_stable = True  # type: ignore[misc]

    def test_gesture_prediction_empty_label(self) -> None:
        """Verify rejection of empty or whitespace label."""
        with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
            GesturePrediction(label="   ", confidence=0.9, latency_ms=2.0)


@pytest.mark.unit
class TestLandmarkFrame:
    """Test suite for LandmarkFrame contract."""

    def test_landmark_frame_nominal(self) -> None:
        """TC-19: Verify nominal LandmarkFrame validation."""
        features = [0.0] * 126
        frame = LandmarkFrame(
            frame_index=0,
            features=features,
            hands_detected=1,
            is_normalized=True,
        )
        assert frame.frame_index == 0
        assert len(frame.features) == 126
        assert frame.hands_detected == 1
        assert frame.is_normalized is True

    def test_landmark_frame_length_mismatch(self) -> None:
        """TC-20: Verify rejection when features list is not exactly 126 elements."""
        with pytest.raises(ValidationError):
            LandmarkFrame(frame_index=0, features=[0.0] * 125)
        with pytest.raises(ValidationError):
            LandmarkFrame(frame_index=0, features=[0.0] * 127)

    def test_landmark_frame_invalid_hands_detected(self) -> None:
        """TC-21: Verify rejection of hands_detected outside [0, 2]."""
        with pytest.raises(ValidationError):
            LandmarkFrame(frame_index=0, features=[0.0] * 126, hands_detected=3)
        with pytest.raises(ValidationError):
            LandmarkFrame(frame_index=0, features=[0.0] * 126, hands_detected=-1)


@pytest.mark.unit
class TestDataCollectionConfig:
    """Test suite for DataCollectionConfig contract."""

    def test_data_collection_config_defaults(self) -> None:
        """TC-22: Verify nominal default configuration values."""
        config = DataCollectionConfig()
        assert config.sequence_length == 30
        assert config.feature_dim == 126
        assert config.num_hands == 2
        assert config.landmarks_per_hand == 21
        assert config.coords == 3
        assert config.min_detection_confidence == 0.7
        assert config.min_tracking_confidence == 0.5
        assert config.handedness_aware is True
        assert config.normalize is True
        assert config.fps == 30.0
        assert config.output_dir == "data/sequences"

    def test_data_collection_config_custom_valid(self) -> None:
        """TC-23: Verify custom valid configuration parameter overrides."""
        config = DataCollectionConfig(
            sequence_length=45,
            feature_dim=126,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
            handedness_aware=False,
            normalize=False,
            fps=60.0,
            output_dir="custom/dataset/path",
        )
        assert config.sequence_length == 45
        assert config.min_detection_confidence == 0.8
        assert config.min_tracking_confidence == 0.6
        assert config.handedness_aware is False
        assert config.normalize is False
        assert config.fps == 60.0
        assert config.output_dir == "custom/dataset/path"

    def test_data_collection_config_empty_output_dir(self) -> None:
        """TC-24: Verify rejection of empty or whitespace output directory."""
        with pytest.raises(ValidationError, match="output_dir cannot be empty or whitespace"):
            DataCollectionConfig(output_dir="   ")

    def test_data_collection_config_sequence_length_bounds(self) -> None:
        """TC-25: Verify rejection when sequence_length is outside [5, 120]."""
        with pytest.raises(ValidationError):
            DataCollectionConfig(sequence_length=4)
        with pytest.raises(ValidationError):
            DataCollectionConfig(sequence_length=121)

    def test_data_collection_config_confidence_bounds(self) -> None:
        """TC-26: Verify rejection when confidence parameters are out of [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            DataCollectionConfig(min_detection_confidence=-0.1)
        with pytest.raises(ValidationError):
            DataCollectionConfig(min_detection_confidence=1.1)
        with pytest.raises(ValidationError):
            DataCollectionConfig(min_tracking_confidence=-0.1)
        with pytest.raises(ValidationError):
            DataCollectionConfig(min_tracking_confidence=1.1)

    def test_data_collection_config_immutability(self) -> None:
        """TC-27: Verify frozen immutability constraint."""
        config = DataCollectionConfig()
        with pytest.raises(ValidationError):
            config.sequence_length = 60  # type: ignore[misc]


@pytest.mark.unit
class TestTrainingConfig:
    """Test suite for TrainingConfig model parameters and validation contracts."""

    def test_training_config_defaults(self) -> None:
        """TC-28: Verify default training parameters."""
        config = TrainingConfig()
        assert config.epochs == 20
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-4
        assert config.val_split == 0.2
        assert config.seed == 42
        assert config.device == "auto"
        assert config.mixed_precision is True
        assert config.pin_memory is True
        assert config.checkpoint_path == "models/weights/best_model.pt"
        assert config.early_stopping_patience == 5

    def test_training_config_custom_valid(self) -> None:
        """TC-29: Verify custom valid configuration parameters."""
        config = TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.005,
            weight_decay=1e-5,
            val_split=0.25,
            checkpoint_path="custom/path.pt",
            early_stopping_patience=10,
        )
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 0.005
        assert config.checkpoint_path == "custom/path.pt"

    def test_training_config_empty_checkpoint_path(self) -> None:
        """TC-30: Verify rejection of empty or whitespace checkpoint path."""
        with pytest.raises(ValidationError, match="checkpoint_path cannot be empty or whitespace"):
            TrainingConfig(checkpoint_path="   ")

    def test_training_config_bounds(self) -> None:
        """TC-31: Verify bounds validation on epochs, batch_size, val_split, and learning_rate."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)
        with pytest.raises(ValidationError):
            TrainingConfig(val_split=0.5)  # val_split must be lt=0.5
        with pytest.raises(ValidationError):
            TrainingConfig(val_split=-0.1)

    def test_training_config_immutability(self) -> None:
        """TC-32: Verify frozen immutability constraint."""
        config = TrainingConfig()
        with pytest.raises(ValidationError):
            config.epochs = 30  # type: ignore[misc]


@pytest.mark.unit
class TestTrainResult:
    """Test suite for TrainResult execution metrics contract."""

    def test_train_result_nominal(self) -> None:
        """TC-33: Verify valid TrainResult creation and serialization."""
        result = TrainResult(
            best_val_accuracy=0.96,
            best_val_loss=0.12,
            best_epoch=14,
            total_epochs=15,
            checkpoint_path="models/weights/best_model.pt",
            train_loss_history=[0.8, 0.4, 0.2],
            val_loss_history=[0.9, 0.5, 0.12],
            val_acc_history=[0.6, 0.85, 0.96],
            training_duration_seconds=12.4,
            model_architecture="lstm",
            num_classes=5,
        )
        assert result.best_val_accuracy == 0.96
        assert result.best_epoch == 14
        assert result.num_classes == 5
        dumped = result.model_dump()
        assert dumped["model_architecture"] == "lstm"

    def test_train_result_validation(self) -> None:
        """TC-34: Verify validation boundaries for accuracy, loss, and classes."""
        with pytest.raises(ValidationError):
            TrainResult(
                best_val_accuracy=1.5,  # > 1.0 invalid
                best_val_loss=0.1,
                best_epoch=1,
                total_epochs=2,
                checkpoint_path="test.pt",
                training_duration_seconds=1.0,
                model_architecture="lstm",
                num_classes=5,
            )
        with pytest.raises(ValidationError):
            TrainResult(
                best_val_accuracy=0.9,
                best_val_loss=-0.1,  # negative loss invalid
                best_epoch=1,
                total_epochs=2,
                checkpoint_path="test.pt",
                training_duration_seconds=1.0,
                model_architecture="lstm",
                num_classes=5,
            )
        with pytest.raises(ValidationError):
            TrainResult(
                best_val_accuracy=0.9,
                best_val_loss=0.1,
                best_epoch=1,
                total_epochs=2,
                checkpoint_path="test.pt",
                training_duration_seconds=1.0,
                model_architecture="lstm",
                num_classes=1,  # must have at least 2 classes
            )


@pytest.mark.unit
class TestSmoothedPrediction:
    """Test suite for SmoothedPrediction temporal output contract."""

    def test_smoothed_prediction_nominal(self) -> None:
        """TC-35: Verify nominal creation and immutability."""
        pred = SmoothedPrediction(
            raw_label="hello",
            raw_confidence=0.92,
            stable_label="hello",
            stable_confidence=0.91,
            is_stable=True,
            should_trigger_tts=True,
            match_count=7,
        )
        assert pred.raw_label == "hello"
        assert pred.stable_label == "hello"
        assert pred.is_stable is True
        assert pred.should_trigger_tts is True
        assert pred.match_count == 7

        with pytest.raises(ValidationError):
            pred.raw_label = "other"  # type: ignore[misc]

    def test_smoothed_prediction_confidence_bounds(self) -> None:
        """TC-36: Verify confidence bounds [0.0, 1.0] for raw and stable confidences."""
        with pytest.raises(ValidationError):
            SmoothedPrediction(
                raw_label="hello",
                raw_confidence=1.2,
                stable_label="hello",
                stable_confidence=0.9,
            )
        with pytest.raises(ValidationError):
            SmoothedPrediction(
                raw_label="hello",
                raw_confidence=0.9,
                stable_label="hello",
                stable_confidence=-0.1,
            )

    def test_smoothed_prediction_empty_labels(self) -> None:
        """TC-37: Verify rejection of empty or whitespace strings for labels."""
        with pytest.raises(ValidationError):
            SmoothedPrediction(
                raw_label="",
                raw_confidence=0.9,
                stable_label="hello",
                stable_confidence=0.9,
            )
        with pytest.raises(ValidationError):
            SmoothedPrediction(
                raw_label="hello",
                raw_confidence=0.9,
                stable_label="   ",
                stable_confidence=0.9,
            )

    def test_smoothed_prediction_match_count_negative(self) -> None:
        """TC-38: Verify match_count cannot be negative."""
        with pytest.raises(ValidationError):
            SmoothedPrediction(
                raw_label="hello",
                raw_confidence=0.9,
                stable_label="hello",
                stable_confidence=0.9,
                match_count=-1,
            )


@pytest.mark.unit
class TestTTSMessage:
    """Test suite for TTSMessage speech dispatch contract."""

    def test_tts_message_nominal(self) -> None:
        """TC-39: Verify nominal creation, automatic timestamp, and immutability."""
        msg = TTSMessage(text="hello")
        assert msg.text == "hello"
        assert msg.timestamp.tzinfo is not None

        with pytest.raises(ValidationError):
            msg.text = "bye"  # type: ignore[misc]

    def test_tts_message_empty_text(self) -> None:
        """TC-40: Verify rejection of empty or whitespace text."""
        with pytest.raises(ValidationError):
            TTSMessage(text="")
        with pytest.raises(ValidationError):
            TTSMessage(text="   ")
