"""Exhaustive unit and integration tests for ONNX export, parity verification, and edge runtime.

Implements all 20 test cases from Milestone 5 Test Matrix:
- TC-01: Export roundtrip for unidirectional LSTM model; check with onnx.checker.
- TC-02: Export roundtrip for bidirectional LSTM model.
- TC-03: Export roundtrip for TransformerClassifier (opset 17).
- TC-04: Export from saved .pt checkpoint using save_checkpoint.
- TC-05: Numerical parity between PyTorch and ONNX logits (LSTM, atol=1e-5).
- TC-06: Numerical parity between PyTorch and ONNX logits (Transformer, atol=1e-5).
- TC-07: Softmax probability distribution equivalence.
- TC-08: Dynamic batching with batch size 1.
- TC-09: Dynamic batching with batch size 4.
- TC-10: Dynamic batching with batch size 16.
- TC-11: CPU Latency benchmark verifies single-sequence latency < 5.0ms (meets_target == True).
- TC-12: Protocol conformance (ONNXModelRunner conforms to SequenceModelRunner).
- TC-13: Pipeline integration with RealtimeGesturePipeline (use_onnx=True and ONNXModelRunner).
- TC-14: Missing checkpoint file raises FileNotFoundError.
- TC-15: Missing ONNX model file raises FileNotFoundError in ONNXModelRunner.
- TC-16: Invalid opset version (e.g. 7) raises Pydantic ValidationError.
- TC-17: All-zeros sequence tensor produces finite probabilities without NaN/Inf.
- TC-18: Constant-value sequence produces finite probabilities summing to 1.0.
- TC-19: Feature dimension mismatch (e.g. shape (30, 100) instead of (30, 126)) raises ValueError.
- TC-20: Corrupted ONNX file raises Exception on session creation.
Plus schema serialization and validation contracts for ONNXExportConfig, ExportResult,
BenchmarkConfig, and BenchmarkResult.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from sign_translator.architecture import LSTMClassifier, TransformerClassifier, build_model
from sign_translator.export_onnx import (
    benchmark_inference,
    export_torch_to_onnx,
    verify_onnx_parity,
)
from sign_translator.live_inference import (
    ONNXModelRunner,
    PyTorchModelRunner,
    RealtimeGesturePipeline,
    SequenceModelRunner,
)
from sign_translator.schemas import (
    BenchmarkConfig,
    BenchmarkResult,
    ExportResult,
    ModelArchitecture,
    ONNXExportConfig,
    PipelineConfig,
    TrainingConfig,
)
from sign_translator.train import save_checkpoint


def _create_lstm_config(bidirectional: bool = False) -> PipelineConfig:
    """Helper to create minimal LSTM pipeline configuration for fast headless testing."""
    return PipelineConfig(
        architecture=ModelArchitecture.LSTM,
        bidirectional=bidirectional,
        hidden_dim=32,
        num_layers=1,
        sequence_length=30,
        feature_dim=126,
        vocabulary=["hello", "thank_you", "yes", "no", "help"],
    )


def _create_transformer_config() -> PipelineConfig:
    """Helper to create minimal Transformer pipeline configuration for fast headless testing."""
    return PipelineConfig(
        architecture=ModelArchitecture.TRANSFORMER,
        hidden_dim=32,
        transformer_heads=4,
        transformer_dim_feedforward=64,
        num_layers=1,
        sequence_length=30,
        feature_dim=126,
        vocabulary=["hello", "thank_you", "yes", "no", "help"],
    )


@pytest.fixture(scope="module")
def shared_exported_lstm(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, nn.Module, PipelineConfig]:
    """Module-scoped fixture providing an exported ONNX LSTM model for dynamic batching tests."""
    tmp_dir = tmp_path_factory.mktemp("shared_lstm")
    pipe_cfg = _create_lstm_config(bidirectional=False)
    torch.manual_seed(42)
    model = build_model(pipe_cfg)
    model.eval()

    onnx_path = tmp_dir / "shared_lstm.onnx"
    export_cfg = ONNXExportConfig(
        checkpoint_path="dummy.pt",
        output_path=str(onnx_path),
        opset_version=17,
        dynamic_batch=True,
        verify_export=True,
    )
    export_torch_to_onnx(model, export_cfg, pipeline_config=pipe_cfg)
    return onnx_path, model, pipe_cfg


@pytest.fixture(scope="module")
def shared_exported_transformer(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, nn.Module, PipelineConfig]:
    """Module-scoped fixture providing an exported ONNX Transformer model for verification."""
    tmp_dir = tmp_path_factory.mktemp("shared_trans")
    pipe_cfg = _create_transformer_config()
    torch.manual_seed(42)
    model = build_model(pipe_cfg)
    model.eval()

    onnx_path = tmp_dir / "shared_transformer.onnx"
    export_cfg = ONNXExportConfig(
        checkpoint_path="dummy.pt",
        output_path=str(onnx_path),
        opset_version=17,
        dynamic_batch=True,
        verify_export=True,
    )
    export_torch_to_onnx(model, export_cfg, pipeline_config=pipe_cfg)
    return onnx_path, model, pipe_cfg


# ==============================================================================
# Milestone 5 Core Test Matrix: TC-01 to TC-20
# ==============================================================================


@pytest.mark.unit
class TestONNXExportCore:
    """Core export roundtrips, graph checking, and parity verification."""

    def test_tc01_export_roundtrip_unidirectional_lstm(self, tmp_path: Path) -> None:
        """TC-01: Export roundtrip for unidirectional LSTM model; check with onnx.checker."""
        pipe_cfg = _create_lstm_config(bidirectional=False)
        model = build_model(pipe_cfg)
        out_path = tmp_path / "lstm_uni.onnx"
        export_cfg = ONNXExportConfig(
            checkpoint_path="placeholder.pt",
            output_path=str(out_path),
            opset_version=17,
            dynamic_batch=True,
            verify_export=True,
        )

        result = export_torch_to_onnx(model, export_cfg, pipeline_config=pipe_cfg)

        assert isinstance(result, ExportResult)
        assert result.parity_verified is True
        assert result.architecture == "lstm"
        assert result.opset_version == 17
        assert result.max_absolute_difference <= export_cfg.atol
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Validate graph integrity via onnx.checker
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

    def test_tc02_export_roundtrip_bidirectional_lstm(self, tmp_path: Path) -> None:
        """TC-02: Export roundtrip for bidirectional LSTM model."""
        pipe_cfg = _create_lstm_config(bidirectional=True)
        model = build_model(pipe_cfg)
        assert isinstance(model, LSTMClassifier)
        assert model.bidirectional is True

        out_path = tmp_path / "lstm_bi.onnx"
        export_cfg = ONNXExportConfig(
            checkpoint_path="placeholder.pt",
            output_path=str(out_path),
            opset_version=17,
            dynamic_batch=True,
            verify_export=True,
        )

        result = export_torch_to_onnx(model, export_cfg, pipeline_config=pipe_cfg)

        assert result.parity_verified is True
        assert out_path.exists()
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

    def test_tc03_export_roundtrip_transformer(self, tmp_path: Path) -> None:
        """TC-03: Export roundtrip for TransformerClassifier (opset 17)."""
        pipe_cfg = _create_transformer_config()
        model = build_model(pipe_cfg)
        assert isinstance(model, TransformerClassifier)

        out_path = tmp_path / "transformer.onnx"
        export_cfg = ONNXExportConfig(
            checkpoint_path="placeholder.pt",
            output_path=str(out_path),
            opset_version=17,
            dynamic_batch=True,
            verify_export=True,
        )

        result = export_torch_to_onnx(model, export_cfg, pipeline_config=pipe_cfg)

        assert result.parity_verified is True
        assert result.architecture == "transformer"
        assert out_path.exists()
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

    def test_tc04_export_from_saved_checkpoint(self, tmp_path: Path) -> None:
        """TC-04: Export from saved .pt checkpoint using save_checkpoint."""
        pipe_cfg = _create_lstm_config(bidirectional=False)
        model = build_model(pipe_cfg)

        ckpt_path = tmp_path / "model_ckpt.pt"
        train_cfg = TrainingConfig(checkpoint_path=str(ckpt_path))
        saved_path = save_checkpoint(
            model=model,
            pipeline_config=pipe_cfg,
            training_config=train_cfg,
            epoch=5,
            val_acc=0.98,
            val_loss=0.04,
            path=ckpt_path,
        )
        assert saved_path.exists()

        out_path = tmp_path / "exported_from_ckpt.onnx"
        export_cfg = ONNXExportConfig(
            checkpoint_path=str(ckpt_path),
            output_path=str(out_path),
            opset_version=17,
            dynamic_batch=True,
            verify_export=True,
        )

        result = export_torch_to_onnx(ckpt_path, export_cfg)

        assert result.parity_verified is True
        assert result.architecture == "lstm"
        assert out_path.exists()
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

    def test_tc05_numerical_parity_lstm(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-05: Numerical parity between PyTorch and ONNX logits (LSTM, atol=1e-5)."""
        onnx_path, model, pipe_cfg = shared_exported_lstm
        max_diff = verify_onnx_parity(
            torch_model=model,
            onnx_path=onnx_path,
            sequence_length=pipe_cfg.sequence_length,
            feature_dim=pipe_cfg.feature_dim,
            batch_sizes=(1, 2, 4),
            atol=1e-5,
        )
        assert max_diff <= 1e-5

    def test_tc06_numerical_parity_transformer(
        self,
        shared_exported_transformer: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-06: Numerical parity between PyTorch and ONNX logits (Transformer, atol=1e-5)."""
        onnx_path, model, pipe_cfg = shared_exported_transformer
        max_diff = verify_onnx_parity(
            torch_model=model,
            onnx_path=onnx_path,
            sequence_length=pipe_cfg.sequence_length,
            feature_dim=pipe_cfg.feature_dim,
            batch_sizes=(1, 2, 4),
            atol=1e-5,
        )
        assert max_diff <= 1e-5

    def test_tc07_softmax_probability_distribution_equivalence(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-07: Softmax probability distribution equivalence between PyTorch and ONNX."""
        onnx_path, model, pipe_cfg = shared_exported_lstm
        pt_runner = PyTorchModelRunner(model)
        onnx_runner = ONNXModelRunner(onnx_path)

        np.random.seed(999)
        sample = np.random.randn(pipe_cfg.sequence_length, pipe_cfg.feature_dim).astype(np.float32)

        pt_probs = pt_runner.predict_proba(sample)
        onnx_probs = onnx_runner.predict_proba(sample)

        np.testing.assert_allclose(pt_probs, onnx_probs, atol=1e-4)
        assert np.isclose(float(np.sum(onnx_probs)), 1.0, atol=1e-5)
        assert np.all(onnx_probs >= 0.0)
        assert len(onnx_probs) == len(pipe_cfg.vocabulary)


@pytest.mark.unit
class TestDynamicBatchingAndInference:
    """Dynamic batching axes and runtime inference validation."""

    def test_tc08_dynamic_batching_batch_1(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-08: Dynamic batching with batch size 1."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        seq_b1 = np.random.randn(1, pipe_cfg.sequence_length, pipe_cfg.feature_dim).astype(
            np.float32
        )
        probs = runner.predict_proba(seq_b1)

        assert probs.shape == (1, len(pipe_cfg.vocabulary))
        assert np.isclose(float(np.sum(probs[0])), 1.0, atol=1e-5)

        # Also test unbatched (2D) input format returning 1D array
        seq_2d = np.random.randn(pipe_cfg.sequence_length, pipe_cfg.feature_dim).astype(np.float32)
        probs_2d = runner.predict_proba(seq_2d)
        assert probs_2d.shape == (len(pipe_cfg.vocabulary),)
        assert np.isclose(float(np.sum(probs_2d)), 1.0, atol=1e-5)

    def test_tc09_dynamic_batching_batch_4(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-09: Dynamic batching with batch size 4."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        seq_b4 = np.random.randn(4, pipe_cfg.sequence_length, pipe_cfg.feature_dim).astype(
            np.float32
        )
        probs = runner.predict_proba(seq_b4)

        assert probs.shape == (4, len(pipe_cfg.vocabulary))
        for b in range(4):
            assert np.isclose(float(np.sum(probs[b])), 1.0, atol=1e-5)
            assert np.all(probs[b] >= 0.0)

    def test_tc10_dynamic_batching_batch_16(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-10: Dynamic batching with batch size 16."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        seq_b16 = np.random.randn(16, pipe_cfg.sequence_length, pipe_cfg.feature_dim).astype(
            np.float32
        )
        probs = runner.predict_proba(seq_b16)

        assert probs.shape == (16, len(pipe_cfg.vocabulary))
        for b in range(16):
            assert np.isclose(float(np.sum(probs[b])), 1.0, atol=1e-5)
            assert np.all(probs[b] >= 0.0)

    def test_tc11_cpu_latency_benchmark(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-11: CPU Latency benchmark verifies single-sequence latency < 5.0ms (meets_target == True)."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path, num_threads=2)

        cfg = BenchmarkConfig(
            num_warmup=10,
            num_iterations=50,
            batch_size=1,
            target_latency_ms=5.0,
        )

        result = benchmark_inference(
            runner=runner,
            sequence_length=pipe_cfg.sequence_length,
            feature_dim=pipe_cfg.feature_dim,
            config=cfg,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.backend == "ONNXModelRunner"
        assert result.batch_size == 1
        assert result.num_iterations == 50
        assert result.mean_latency_ms > 0.0
        assert result.throughput_fps > 0.0
        assert result.min_latency_ms <= result.mean_latency_ms <= result.max_latency_ms
        assert result.median_latency_ms <= result.p95_latency_ms <= result.p99_latency_ms
        # CPU lightweight model must satisfy < 5.0ms on modern CPU hardware
        assert result.mean_latency_ms < 5.0, (
            f"Mean latency {result.mean_latency_ms:.3f}ms exceeded 5.0ms"
        )
        assert result.meets_target is True

    def test_tc12_protocol_conformance(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-12: Protocol conformance (ONNXModelRunner conforms to SequenceModelRunner)."""
        onnx_path, _, _ = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        assert isinstance(runner, SequenceModelRunner)
        assert hasattr(runner, "predict_proba")
        assert callable(runner.predict_proba)

    def test_tc13_pipeline_integration_with_onnx(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-13: Pipeline integration with RealtimeGesturePipeline via use_onnx=True and ONNXModelRunner."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        # Mode A: Passing ONNXModelRunner instance directly
        pipeline_a = RealtimeGesturePipeline(
            config=pipe_cfg,
            model=runner,
            enable_tts=False,
        )
        assert pipeline_a.model_runner is runner

        # Feed 30 frames to fill window and evaluate
        for _ in range(pipe_cfg.sequence_length):
            pred_a = pipeline_a.process_landmarks([0.0] * pipe_cfg.feature_dim)
        assert pred_a.label in pipe_cfg.vocabulary
        assert 0.0 <= pred_a.confidence <= 1.0

        # Mode B: PipelineConfig with use_onnx=True and onnx_model_path
        onnx_pipeline_cfg = pipe_cfg.model_copy(
            update={"use_onnx": True, "onnx_model_path": str(onnx_path)}
        )
        pipeline_b = RealtimeGesturePipeline(
            config=onnx_pipeline_cfg,
            model=None,
            enable_tts=False,
        )
        assert isinstance(pipeline_b.model_runner, ONNXModelRunner)

        for _ in range(pipe_cfg.sequence_length):
            pred_b = pipeline_b.process_landmarks([0.1] * pipe_cfg.feature_dim)
        assert pred_b.label in pipe_cfg.vocabulary
        assert 0.0 <= pred_b.confidence <= 1.0


@pytest.mark.unit
class TestONNXRobustnessAndEdgeCases:
    """Edge cases, error handling, input shape bounds, and degenerate tensors."""

    def test_tc14_missing_checkpoint_raises_file_not_found(self, tmp_path: Path) -> None:
        """TC-14: Missing checkpoint file raises FileNotFoundError."""
        non_existent_ckpt = tmp_path / "non_existent_model.pt"
        export_cfg = ONNXExportConfig(checkpoint_path=str(non_existent_ckpt))

        with pytest.raises(FileNotFoundError, match="checkpoint"):
            export_torch_to_onnx(non_existent_ckpt, export_cfg)

    def test_tc15_missing_onnx_model_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """TC-15: Missing ONNX model file raises FileNotFoundError in ONNXModelRunner."""
        non_existent_onnx = tmp_path / "missing_graph.onnx"

        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            ONNXModelRunner(non_existent_onnx)

    def test_tc16_invalid_opset_raises_validation_error(self) -> None:
        """TC-16: Invalid opset version (e.g. 7 or 25) raises Pydantic ValidationError."""
        # Lower bound violation (opset 7 < ge=11)
        with pytest.raises(ValidationError) as exc_info_low:
            ONNXExportConfig(checkpoint_path="model.pt", opset_version=7)
        assert "opset_version" in str(exc_info_low.value)

        # Upper bound violation (opset 25 > le=20)
        with pytest.raises(ValidationError) as exc_info_high:
            ONNXExportConfig(checkpoint_path="model.pt", opset_version=25)
        assert "opset_version" in str(exc_info_high.value)

    def test_tc17_all_zeros_sequence_produces_finite_probabilities(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-17: All-zeros sequence tensor produces finite probabilities without NaN/Inf."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        zero_seq = np.zeros((pipe_cfg.sequence_length, pipe_cfg.feature_dim), dtype=np.float32)
        probs = runner.predict_proba(zero_seq)

        assert probs.shape == (len(pipe_cfg.vocabulary),)
        assert np.all(np.isfinite(probs))
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

    def test_tc18_constant_value_sequence_produces_finite_probabilities(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-18: Constant-value sequence produces finite probabilities summing to 1.0."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        const_seq = np.full(
            (pipe_cfg.sequence_length, pipe_cfg.feature_dim),
            fill_value=0.75,
            dtype=np.float32,
        )
        probs = runner.predict_proba(const_seq)

        assert probs.shape == (len(pipe_cfg.vocabulary),)
        assert np.all(np.isfinite(probs))
        assert not np.any(np.isnan(probs))
        assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

    def test_tc19_feature_dim_mismatch_raises_value_error(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """TC-19: Feature dimension mismatch (e.g. shape (30, 100) instead of (30, 126)) raises ValueError."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        bad_seq = np.zeros((pipe_cfg.sequence_length, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            runner.predict_proba(bad_seq)

        # Also test invalid dimensions (1D or 4D)
        with pytest.raises(ValueError, match="Expected sequence of 2 or 3 dimensions"):
            runner.predict_proba(np.zeros((126,), dtype=np.float32))

        with pytest.raises(ValueError, match="Expected sequence of 2 or 3 dimensions"):
            runner.predict_proba(np.zeros((1, 2, 30, 126), dtype=np.float32))

    def test_tc20_corrupted_onnx_file_raises_exception(self, tmp_path: Path) -> None:
        """TC-20: Corrupted ONNX file raises Exception on session creation."""
        corrupted_file = tmp_path / "corrupted_graph.onnx"
        corrupted_file.write_bytes(b"INVALID_MAGIC_HEADER_NOT_A_VALID_PROTOBUF")

        with pytest.raises(Exception, match="(?i)protobuf|load|failed|file"):
            ONNXModelRunner(corrupted_file)


# ==============================================================================
# Pydantic v2 Schema Contract & Serialization Tests
# ==============================================================================


@pytest.mark.unit
class TestONNXSchemas:
    """Strict Pydantic v2 contract validation for Milestone 5 schemas."""

    def test_onnx_export_config_nominal_and_immutability(self) -> None:
        """Verify nominal construction, defaults, JSON roundtrip, and immutability."""
        cfg = ONNXExportConfig(checkpoint_path="checkpoints/best.pt")
        assert cfg.output_path == "models/exported/gesture_model.onnx"
        assert cfg.opset_version == 17
        assert cfg.dynamic_batch is True
        assert cfg.atol == 1e-5
        assert cfg.verify_export is True
        assert cfg.optimize_model is True

        # Immutability
        with pytest.raises(ValidationError):
            cfg.opset_version = 18  # type: ignore[misc]

        # Extra fields forbidden
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="test.pt", extra_field=123)  # type: ignore[call-arg]

        # JSON roundtrip
        json_str = cfg.model_dump_json()
        restored = ONNXExportConfig.model_validate_json(json_str)
        assert restored == cfg

    def test_onnx_export_config_path_validation(self) -> None:
        """Verify empty and whitespace paths are rejected."""
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="")
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="   ")
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="valid.pt", output_path="")
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="valid.pt", output_path="  \t ")

    def test_onnx_export_config_atol_bounds(self) -> None:
        """Verify atol constraints (gt=0.0, le=1e-2)."""
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="m.pt", atol=0.0)
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="m.pt", atol=-1e-5)
        with pytest.raises(ValidationError):
            ONNXExportConfig(checkpoint_path="m.pt", atol=0.1)

    def test_export_result_nominal_and_immutability(self) -> None:
        """Verify ExportResult schema contract, immutability, and JSON roundtrip."""
        res = ExportResult(
            checkpoint_path="ckpt.pt",
            output_path="model.onnx",
            file_size_bytes=1024,
            architecture="lstm",
            opset_version=17,
            dynamic_axes={"sequence_input": {0: "batch_size"}},
            max_absolute_difference=1.2e-6,
            parity_verified=True,
            export_duration_seconds=0.15,
        )
        assert res.file_size_bytes == 1024
        assert res.parity_verified is True

        # Immutability
        with pytest.raises(ValidationError):
            res.file_size_bytes = 2048  # type: ignore[misc]

        # Extra forbidden
        with pytest.raises(ValidationError):
            ExportResult(
                checkpoint_path="ckpt.pt",
                output_path="model.onnx",
                file_size_bytes=1024,
                architecture="lstm",
                opset_version=17,
                dynamic_axes={},
                max_absolute_difference=1e-6,
                parity_verified=True,
                export_duration_seconds=0.1,
                rogue="forbidden",  # type: ignore[call-arg]
            )

        # JSON roundtrip
        json_data = res.model_dump_json()
        assert ExportResult.model_validate_json(json_data) == res

    def test_benchmark_config_nominal_and_bounds(self) -> None:
        """Verify BenchmarkConfig defaults, bounds, immutability, and rejection of invalid values."""
        cfg = BenchmarkConfig()
        assert cfg.num_warmup == 10
        assert cfg.num_iterations == 100
        assert cfg.batch_size == 1
        assert cfg.target_latency_ms == 5.0

        # Immutability
        with pytest.raises(ValidationError):
            cfg.num_iterations = 200  # type: ignore[misc]

        # Bounds checks
        with pytest.raises(ValidationError):
            BenchmarkConfig(num_warmup=0)
        with pytest.raises(ValidationError):
            BenchmarkConfig(num_iterations=5)  # min 10
        with pytest.raises(ValidationError):
            BenchmarkConfig(batch_size=0)
        with pytest.raises(ValidationError):
            BenchmarkConfig(batch_size=256)  # max 128
        with pytest.raises(ValidationError):
            BenchmarkConfig(target_latency_ms=0.0)
        with pytest.raises(ValidationError):
            BenchmarkConfig(target_latency_ms=-1.0)

    def test_benchmark_result_nominal_and_immutability(self) -> None:
        """Verify BenchmarkResult schema, non-negativity constraints, and immutability."""
        res = BenchmarkResult(
            backend="ONNXModelRunner",
            batch_size=1,
            num_iterations=100,
            mean_latency_ms=1.23,
            median_latency_ms=1.20,
            p95_latency_ms=1.45,
            p99_latency_ms=1.60,
            min_latency_ms=1.10,
            max_latency_ms=1.75,
            throughput_fps=813.0,
            meets_target=True,
        )
        assert res.meets_target is True
        assert res.mean_latency_ms == 1.23

        # Immutability
        with pytest.raises(ValidationError):
            res.meets_target = False  # type: ignore[misc]

        # Negative latencies rejected
        with pytest.raises(ValidationError):
            BenchmarkResult(
                backend="ONNXModelRunner",
                batch_size=1,
                num_iterations=100,
                mean_latency_ms=-1.0,
                median_latency_ms=1.2,
                p95_latency_ms=1.4,
                p99_latency_ms=1.6,
                min_latency_ms=1.0,
                max_latency_ms=2.0,
                throughput_fps=100.0,
                meets_target=True,
            )

        # JSON roundtrip
        json_data = res.model_dump_json()
        assert BenchmarkResult.model_validate_json(json_data) == res


# ==============================================================================
# Additional Edge Case and Parity Failure Coverage
# ==============================================================================


@pytest.mark.unit
class TestExportParityFailureAndEdgePaths:
    """Tests failure paths during verification and unsupported arguments."""

    def test_verify_onnx_parity_missing_file(self, tmp_path: Path) -> None:
        """verify_onnx_parity raises FileNotFoundError when graph file is absent."""
        pipe_cfg = _create_lstm_config()
        model = build_model(pipe_cfg)
        with pytest.raises(FileNotFoundError):
            verify_onnx_parity(
                torch_model=model,
                onnx_path=tmp_path / "does_not_exist.onnx",
            )

    def test_verify_onnx_parity_exceeds_atol_raises_value_error(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """verify_onnx_parity raises ValueError when difference exceeds artificially tiny atol."""
        onnx_path, model, pipe_cfg = shared_exported_lstm
        with pytest.raises(ValueError, match="Parity verification failed"):
            verify_onnx_parity(
                torch_model=model,
                onnx_path=onnx_path,
                sequence_length=pipe_cfg.sequence_length,
                feature_dim=pipe_cfg.feature_dim,
                batch_sizes=(1,),
                atol=1e-12,  # Floating point epsilon guarantee to fail
            )

    def test_export_unsupported_model_type_raises_type_error(self) -> None:
        """export_torch_to_onnx raises TypeError when invalid object type is supplied."""
        export_cfg = ONNXExportConfig(checkpoint_path="dummy.pt")
        with pytest.raises(TypeError, match="Unsupported model_or_checkpoint type"):
            export_torch_to_onnx(12345, export_cfg)  # type: ignore[arg-type]

    def test_benchmark_inference_multi_batch(
        self,
        shared_exported_lstm: tuple[Path, nn.Module, PipelineConfig],
    ) -> None:
        """benchmark_inference functions correctly when batch_size > 1."""
        onnx_path, _, pipe_cfg = shared_exported_lstm
        runner = ONNXModelRunner(onnx_path)

        cfg = BenchmarkConfig(
            num_warmup=5,
            num_iterations=20,
            batch_size=4,
            target_latency_ms=20.0,
        )

        res = benchmark_inference(
            runner=runner,
            sequence_length=pipe_cfg.sequence_length,
            feature_dim=pipe_cfg.feature_dim,
            config=cfg,
        )

        assert res.batch_size == 4
        assert res.num_iterations == 20
        assert res.mean_latency_ms > 0.0
        assert res.throughput_fps > 0.0
