"""Domain models and data contracts for Real-Time Sign Language Translator.

Enforces strict Pydantic v2 validation across pipeline settings, sequence
tensors, and real-time inference predictions.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(StrEnum):
    """Supported execution and acceleration compute device targets."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class ModelArchitecture(StrEnum):
    """Supported sequence model classification backbones."""

    LSTM = "lstm"
    TRANSFORMER = "transformer"


class PipelineConfig(BaseSettings):
    """Global configuration settings and hyperparameters for the pipeline.

    Supports configuration via environment variables prefixed with `SIGN_TR_`.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIGN_TR_",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # Spatial-temporal capture dimensions
    sequence_length: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Temporal sequence buffer length (T) in frames.",
    )
    num_hands: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Maximum number of hands tracked simultaneously.",
    )
    landmarks_per_hand: int = Field(
        default=21,
        ge=21,
        le=21,
        description="MediaPipe hand landmark points per tracked hand.",
    )
    coords: int = Field(
        default=3,
        ge=2,
        le=3,
        description="Spatial coordinates per landmark point (X, Y, Z).",
    )
    feature_dim: int = Field(
        default=126,
        ge=42,
        le=126,
        description="Total flattened feature vector dimension per frame.",
    )

    # Gesture vocabulary
    vocabulary: list[str] = Field(
        default_factory=lambda: ["hello", "thank_you", "yes", "no", "help"],
        min_length=2,
        description="Ordered list of recognizable sign language gesture labels.",
    )

    # Sequence model hyperparameters
    architecture: ModelArchitecture = Field(
        default=ModelArchitecture.LSTM,
        description="Model architecture for classification (lstm | transformer).",
    )
    hidden_dim: int = Field(
        default=128,
        ge=16,
        le=1024,
        description="Hidden state / embedding dimension.",
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of recurrent or self-attention layers.",
    )
    dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.9,
        description="Dropout rate for regularizing sequence models.",
    )
    bidirectional: bool = Field(
        default=False,
        description="Whether LSTM sequence model operates in bidirectional mode.",
    )
    transformer_heads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of attention heads for TransformerClassifier.",
    )

    # Real-time smoothing & debounce
    smoothing_window: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Sliding window size (K) in frames for prediction smoothing.",
    )
    smoothing_min_matches: int = Field(
        default=6,
        ge=1,
        le=60,
        description="Minimum identical predictions (M <= K) required for stability.",
    )
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum softmax probability threshold to accept a gesture.",
    )
    cooldown_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Minimum elapsed time between consecutive TTS audio triggers.",
    )

    # File paths and ONNX inference configuration
    model_weights_path: str = Field(
        default="models/weights/best_model.pt",
        description="Filesystem path to PyTorch checkpoint.",
    )
    onnx_model_path: str = Field(
        default="models/exported/gesture_model.onnx",
        description="Filesystem path to exported ONNX model.",
    )
    use_onnx: bool = Field(
        default=False,
        description="Whether to use ONNX Runtime engine for inference.",
    )
    onnx_intra_op_threads: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Intra-op thread pool size for ONNX session.",
    )
    onnx_inter_op_threads: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Inter-op thread pool size for ONNX session.",
    )
    onnx_graph_optimization_level: str = Field(
        default="all",
        description="Graph optimization level (all | basic | extended | disable).",
    )
    device: str = Field(
        default="auto",
        description="Compute device target ('auto', 'cuda', 'cpu', 'cuda:<id>').",
    )
    execution_providers: list[str] | None = Field(
        default=None,
        description="Explicit ONNX execution provider priority list.",
    )
    data_dir: str = Field(
        default="data/sequences",
        description="Base directory for captured NumPy sequence datasets.",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned in {"auto", "cpu", "cuda"}:
            return cleaned
        if re.match(r"^cuda:\d+$", cleaned):
            return cleaned
        raise ValueError(
            f"Invalid device target '{v}'. Expected 'auto', 'cuda', 'cpu', or 'cuda:<id>'."
        )

    @field_validator("vocabulary")
    @classmethod
    def validate_vocabulary(cls, v: list[str]) -> list[str]:
        cleaned = [item.strip() for item in v if item and item.strip()]
        if len(cleaned) < 2:
            raise ValueError("Vocabulary must contain at least 2 non-empty gesture classes.")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"Vocabulary contains duplicate gesture classes: {cleaned}")
        return cleaned

    @model_validator(mode="after")
    def validate_relationships(self) -> PipelineConfig:
        expected_feature_dim = self.num_hands * self.landmarks_per_hand * self.coords
        if self.feature_dim != expected_feature_dim:
            raise ValueError(
                f"feature_dim ({self.feature_dim}) must equal "
                f"num_hands * landmarks_per_hand * coords ({expected_feature_dim})"
            )
        if self.smoothing_min_matches > self.smoothing_window:
            raise ValueError(
                f"smoothing_min_matches ({self.smoothing_min_matches}) cannot exceed "
                f"smoothing_window ({self.smoothing_window})"
            )
        if (
            self.architecture == ModelArchitecture.TRANSFORMER
            and self.hidden_dim % self.transformer_heads != 0
        ):
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"transformer_heads ({self.transformer_heads})"
            )
        return self


class SequenceMetadata(BaseModel):
    """Metadata contract accompanying saved gesture landmark sequences."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    sequence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the recorded sequence.",
    )
    gesture_label: str = Field(
        ...,
        min_length=1,
        description="Ground truth gesture class label.",
    )
    num_frames: int = Field(
        default=30,
        ge=1,
        description="Total frames captured in the sequence tensor.",
    )
    feature_dim: int = Field(
        default=126,
        ge=1,
        description="Feature dimensionality per frame.",
    )
    normalized: bool = Field(
        default=True,
        description="True if coordinates are wrist-centered and scale-normalized.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the sequence was captured.",
    )
    fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Camera capture frame rate during sequence acquisition.",
    )

    @field_validator("gesture_label", mode="before")
    @classmethod
    def validate_label(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("gesture_label cannot be empty or whitespace.")
        return v


class GesturePrediction(BaseModel):
    """Prediction contract emitted by the real-time inference engine."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    label: str = Field(
        ...,
        min_length=1,
        description="Predicted gesture class name or background token.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability score of the predicted gesture.",
    )
    is_stable: bool = Field(
        default=False,
        description="True if prediction satisfies M-of-K temporal smoothing.",
    )
    triggered_tts: bool = Field(
        default=False,
        description="True if prediction triggered an asynchronous TTS speech event.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference and preprocessing elapsed time in milliseconds.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the prediction.",
    )

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Prediction label cannot be empty or whitespace.")
        return v


class LandmarkFrame(BaseModel):
    """Contract for a single frame landmark vector payload."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    frame_index: int = Field(
        ...,
        ge=0,
        description="Zero-based sequence frame index.",
    )
    features: list[float] = Field(
        ...,
        min_length=126,
        max_length=126,
        description="Flattened 126-D hand landmark coordinates.",
    )
    hands_detected: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Count of active hands detected in this frame (0, 1, or 2).",
    )
    is_normalized: bool = Field(
        default=False,
        description="Indicates whether spatial normalization has been applied.",
    )


class DataCollectionConfig(BaseModel):
    """Configuration contract for gesture landmark sequence acquisition."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    sequence_length: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Target frames per recorded gesture sequence.",
    )
    feature_dim: int = Field(
        default=126,
        ge=42,
        le=126,
        description="Feature dimension per frame (2 hands x 21 landmarks x 3 coords).",
    )
    num_hands: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Maximum simultaneous hands to track.",
    )
    landmarks_per_hand: int = Field(
        default=21,
        ge=21,
        le=21,
        description="MediaPipe landmarks per hand.",
    )
    coords: int = Field(
        default=3,
        ge=2,
        le=3,
        description="Spatial coordinates per landmark (X, Y, Z).",
    )
    min_detection_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MediaPipe hand detection confidence threshold.",
    )
    min_tracking_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MediaPipe hand landmark tracking confidence threshold.",
    )
    handedness_aware: bool = Field(
        default=True,
        description="If True, assigns Left hand to Slot 0 and Right hand to Slot 1.",
    )
    normalize: bool = Field(
        default=True,
        description="If True, applies wrist translation and max L2 scale normalization.",
    )
    fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Acquisition frame rate.",
    )
    output_dir: str = Field(
        default="data/sequences",
        description="Base filesystem directory for dataset persistence.",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("output_dir cannot be empty or whitespace.")
        return v


class TrainingConfig(BaseModel):
    """Pydantic v2 contract governing model training parameters and persistence."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    epochs: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum training epochs.",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=512,
        description="Mini-batch size for DataLoader.",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Learning rate for Adam optimizer.",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        le=0.1,
        description="L2 regularization factor.",
    )
    val_split: float = Field(
        default=0.2,
        ge=0.0,
        lt=0.5,
        description="Fraction of dataset allocated to validation.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility across splits and initialization.",
    )
    device: str = Field(
        default="auto",
        description="Compute device target ('auto', 'cuda', 'cpu', 'cuda:<id>').",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable torch.amp mixed precision acceleration on CUDA devices.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Enable pinned host memory for asynchronous CUDA DataLoader transfers.",
    )
    checkpoint_path: str = Field(
        default="models/weights/best_model.pt",
        description="Target filesystem destination for best model checkpoint.",
    )
    early_stopping_patience: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of non-improving epochs before early stopping triggers.",
    )

    @field_validator("device")
    @classmethod
    def validate_training_device(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned in {"auto", "cpu", "cuda"}:
            return cleaned
        if re.match(r"^cuda:\d+$", cleaned):
            return cleaned
        raise ValueError(
            f"Invalid training device '{v}'. Expected 'auto', 'cuda', 'cpu', or 'cuda:<id>'."
        )

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def validate_checkpoint_path(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("checkpoint_path cannot be empty or whitespace.")
        return v


class TrainResult(BaseModel):
    """Execution metrics and artifact metadata produced by the training loop."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    best_val_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Peak validation accuracy achieved during training.",
    )
    best_val_loss: float = Field(
        ...,
        ge=0.0,
        description="Validation loss corresponding to best validation epoch.",
    )
    best_epoch: int = Field(
        ...,
        ge=0,
        description="Epoch index (0-based) where best validation metrics occurred.",
    )
    total_epochs: int = Field(
        ...,
        ge=1,
        description="Total training epochs completed before termination.",
    )
    checkpoint_path: str = Field(
        ...,
        description="Filesystem path where best model checkpoint was saved.",
    )
    train_loss_history: list[float] = Field(
        default_factory=list,
        description="Epoch-by-epoch training loss trajectory.",
    )
    val_loss_history: list[float] = Field(
        default_factory=list,
        description="Epoch-by-epoch validation loss trajectory.",
    )
    val_acc_history: list[float] = Field(
        default_factory=list,
        description="Epoch-by-epoch validation accuracy trajectory.",
    )
    training_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed wall-clock training time in seconds.",
    )
    model_architecture: str = Field(
        ...,
        description="Classification backbone identifier ('lstm' or 'transformer').",
    )
    num_classes: int = Field(
        ...,
        ge=2,
        description="Number of target gesture classes trained.",
    )
    device_used: str = Field(
        default="cpu",
        description="Actual compute device utilized during training (e.g. 'cuda:0', 'cpu').",
    )
    gpu_name: str | None = Field(
        default=None,
        description="GPU hardware model if CUDA was utilized.",
    )


class SmoothedPrediction(BaseModel):
    """Output contract for stabilized temporal predictions emitted by PredictionSmoother."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    raw_label: str = Field(
        ...,
        min_length=1,
        description="Raw predicted gesture label from current frame forward pass.",
    )
    raw_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability score of raw frame prediction.",
    )
    stable_label: str = Field(
        ...,
        min_length=1,
        description="Stabilized gesture class label after M-of-K majority voting.",
    )
    stable_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Averaged confidence score across matching frames in sliding window.",
    )
    is_stable: bool = Field(
        default=False,
        description="True if candidate label satisfies M-of-K consensus requirement.",
    )
    should_trigger_tts: bool = Field(
        default=False,
        description="True if prediction is stable and cooldown debouncing interval has elapsed.",
    )
    match_count: int = Field(
        default=0,
        ge=0,
        description="Number of matching high-confidence frames in current sliding window.",
    )


class TTSMessage(BaseModel):
    """Speech synthesis dispatch request payload."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Speech synthesis utterance string.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the synthesis request.",
    )


class ONNXExportConfig(BaseModel):
    """Pydantic v2 contract governing ONNX graph serialization and validation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    checkpoint_path: str = Field(
        ...,
        min_length=1,
        description="Path to PyTorch checkpoint (.pt) containing model state dict and config.",
    )
    output_path: str = Field(
        default="models/exported/gesture_model.onnx",
        min_length=1,
        description="Target destination filesystem path for exported ONNX graph.",
    )
    opset_version: int = Field(
        default=17,
        ge=11,
        le=20,
        description="ONNX operator set version (recommended 17 for PyTorch 2.x LayerNorm/LSTM ops).",
    )
    dynamic_batch: bool = Field(
        default=True,
        description="Whether to export model with dynamic batch dimension ('batch_size').",
    )
    input_names: list[str] = Field(
        default_factory=lambda: ["sequence_input"],
        min_length=1,
        description="Input tensor node names in ONNX computational graph.",
    )
    output_names: list[str] = Field(
        default_factory=lambda: ["logits"],
        min_length=1,
        description="Output tensor node names in ONNX computational graph.",
    )
    verify_export: bool = Field(
        default=True,
        description="Whether to execute automated numerical parity verification post-export.",
    )
    atol: float = Field(
        default=1e-5,
        gt=0.0,
        le=1e-2,
        description="Maximum allowed absolute numerical difference between PyTorch and ONNX logits.",
    )
    optimize_model: bool = Field(
        default=True,
        description="Whether to apply ONNX Runtime offline graph optimizations.",
    )

    @field_validator("checkpoint_path", "output_path", mode="before")
    @classmethod
    def validate_non_empty_path(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Filesystem path cannot be empty or whitespace.")
        return v


class ExportResult(BaseModel):
    """Execution telemetry and parity audit metadata produced by export_torch_to_onnx."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    checkpoint_path: str = Field(..., description="Source PyTorch checkpoint path.")
    output_path: str = Field(..., description="Exported ONNX model destination path.")
    file_size_bytes: int = Field(
        ..., ge=1, description="Size of exported ONNX graph artifact in bytes."
    )
    architecture: str = Field(
        ..., description="Classification backbone identifier ('lstm' or 'transformer')."
    )
    opset_version: int = Field(
        ..., ge=11, description="ONNX opset version embedded in model protobuf."
    )
    dynamic_axes: dict[str, dict[int, str]] = Field(
        ...,
        description="Dynamic axes mapping configured during serialization.",
    )
    max_absolute_difference: float = Field(
        ...,
        ge=0.0,
        description="Peak absolute difference between PyTorch and ONNX outputs.",
    )
    parity_verified: bool = Field(..., description="True if max_absolute_difference <= atol.")
    export_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time required for conversion and verification in seconds.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of export operation.",
    )


class BenchmarkConfig(BaseModel):
    """Configuration contract for sequence model inference benchmarking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_warmup: int = Field(
        default=10, ge=1, le=1000, description="Warmup iterations to prime caches."
    )
    num_iterations: int = Field(
        default=100, ge=10, le=10000, description="Number of benchmark iterations."
    )
    batch_size: int = Field(
        default=1, ge=1, le=128, description="Mini-batch size evaluated per iteration."
    )
    batch_sizes: list[int] = Field(
        default_factory=lambda: [1, 8, 32, 64],
        min_length=1,
        description="List of batch sizes to evaluate in throughput/latency profiling sweep.",
    )
    target_latency_ms: float = Field(
        default=5.0, gt=0.0, le=100.0, description="Acceptance threshold in milliseconds."
    )
    device: str = Field(
        default="auto",
        description="Target compute device for benchmark ('auto', 'cuda', 'cpu').",
    )
    execution_provider: str = Field(
        default="auto",
        description="Target execution provider ('auto', 'CPUExecutionProvider', 'CUDAExecutionProvider').",
    )


class BenchmarkResult(BaseModel):
    """Detailed statistical metrics emitted by inference latency benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    backend: str = Field(
        ..., description="Inference runtime identifier ('ONNXModelRunner', 'PyTorchModelRunner')."
    )
    device: str = Field(
        default="cpu",
        description="Actual compute device evaluated ('cpu', 'cuda:0').",
    )
    execution_provider: str = Field(
        default="CPUExecutionProvider",
        description="Active ONNX execution provider.",
    )
    batch_size: int = Field(..., ge=1, description="Evaluated batch size.")
    num_iterations: int = Field(..., ge=10, description="Total timed iterations completed.")
    mean_latency_ms: float = Field(
        ..., ge=0.0, description="Arithmetic mean inference latency in ms."
    )
    median_latency_ms: float = Field(
        ..., ge=0.0, description="50th percentile (median) latency in ms."
    )
    p95_latency_ms: float = Field(..., ge=0.0, description="95th percentile latency in ms.")
    p99_latency_ms: float = Field(..., ge=0.0, description="99th percentile latency in ms.")
    min_latency_ms: float = Field(..., ge=0.0, description="Minimum recorded latency in ms.")
    max_latency_ms: float = Field(..., ge=0.0, description="Maximum recorded latency in ms.")
    throughput_fps: float = Field(..., ge=0.0, description="Effective inferences per second.")
    meets_target: bool = Field(..., description="True if mean_latency_ms <= target_latency_ms.")
    gpu_name: str | None = Field(
        default=None,
        description="GPU hardware model if CUDA was evaluated.",
    )
    vram_allocated_mb: float | None = Field(
        default=None,
        description="Peak GPU memory allocated in MB if CUDA was evaluated.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when benchmark completed.",
    )
