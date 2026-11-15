"""Unit and integration tests for CUDA acceleration, GPU enablement, and device fallback.

Implements all 12 test cases from the CUDA Acceleration Specification and Matrix:
- TC-CUDA-01: resolve_torch_device('cpu') returns torch.device('cpu') in all environments.
- TC-CUDA-02: resolve_torch_device('cuda') gracefully falls back to CPU when CUDA is unavailable.
- TC-CUDA-03: resolve_torch_device('auto') probes availability and selects CUDA or CPU.
- TC-CUDA-04: resolve_torch_device('invalid_xyz') raises ValueError.
- TC-CUDA-05: resolve_ort_providers('auto') returns prioritized providers based on availability.
- TC-CUDA-06: resolve_ort_providers('CUDAExecutionProvider') behavior when available vs unavailable.
- TC-CUDA-07: resolve_ort_providers('CPUExecutionProvider') returns ['CPUExecutionProvider'].
- TC-CUDA-08: get_device_telemetry() schema, metrics verification, and exception safety.
- TC-CUDA-09: Schema validation for device fields across PipelineConfig, TrainingConfig,
              BenchmarkConfig, and BenchmarkResult.
- TC-CUDA-10: Mixed precision training logic and GradScaler workflow (train_epoch & evaluate).
- TC-CUDA-11: PyTorchModelRunner and ONNXModelRunner device propagation, inference, and fallback.
- TC-CUDA-12: benchmark_inference with CUDA timing synchronization and telemetry population.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sign_translator.architecture import LSTMClassifier
from sign_translator.device import (
    get_device_telemetry,
    is_cuda_available,
    resolve_ort_providers,
    resolve_torch_device,
)
from sign_translator.export_onnx import benchmark_inference, verify_onnx_parity
from sign_translator.live_inference import ONNXModelRunner, PyTorchModelRunner
from sign_translator.schemas import (
    BenchmarkConfig,
    BenchmarkResult,
    PipelineConfig,
    TrainingConfig,
    TrainResult,
)
from sign_translator.train import evaluate, train_epoch, train_model


# ---------------------------------------------------------------------------
# TC-CUDA-01 to TC-CUDA-04: PyTorch Device Resolution
# ---------------------------------------------------------------------------
class TestTorchDeviceResolution:
    """Tests for resolve_torch_device and is_cuda_available."""

    @pytest.mark.parametrize("dev_str", ["cpu", "CPU", "Cpu", "  cpu  "])
    def test_tc_cuda_01_resolve_cpu_nominal(self, dev_str: str) -> None:
        """TC-CUDA-01: resolve_torch_device('cpu') returns torch.device('cpu') regardless of CUDA availability."""
        with patch("sign_translator.device.torch.cuda.is_available", return_value=True):
            resolved = resolve_torch_device(dev_str)
            assert resolved == torch.device("cpu")

        with patch("sign_translator.device.torch.cuda.is_available", return_value=False):
            resolved = resolve_torch_device(dev_str)
            assert resolved == torch.device("cpu")

    def test_tc_cuda_02_resolve_cuda_fallback_when_unavailable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TC-CUDA-02: resolve_torch_device('cuda') falls back to CPU when CUDA is unavailable."""
        with patch("sign_translator.device.torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.WARNING):
                resolved = resolve_torch_device("cuda")
                assert resolved == torch.device("cpu")
                assert "Falling back to CPU" in caplog.text

            caplog.clear()
            with caplog.at_level(logging.WARNING):
                resolved_id = resolve_torch_device("cuda:0")
                assert resolved_id == torch.device("cpu")
                assert "Falling back to CPU" in caplog.text

    def test_tc_cuda_03_resolve_auto_and_cuda_when_available(self) -> None:
        """TC-CUDA-03: resolve_torch_device('auto') and 'cuda' probe and return CUDA or CPU accordingly."""
        # When CUDA is available
        with patch("sign_translator.device.torch.cuda.is_available", return_value=True):
            assert resolve_torch_device("auto") == torch.device("cuda:0")
            assert resolve_torch_device("cuda") == torch.device("cuda:0")
            assert resolve_torch_device("cuda:0") == torch.device("cuda:0")
            assert resolve_torch_device("cuda:1") == torch.device("cuda:1")

        # When CUDA is absent
        with patch("sign_translator.device.torch.cuda.is_available", return_value=False):
            assert resolve_torch_device("auto") == torch.device("cpu")
            assert resolve_torch_device("") == torch.device("cpu")
            assert resolve_torch_device(None) == torch.device("cpu")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "invalid_target",
        ["invalid_xyz", "tpu", "opencl", "cuda:abc", "rocm", "mps", "cuda:-1", "metal"],
    )
    def test_tc_cuda_04_resolve_invalid_target_raises_value_error(
        self, invalid_target: str
    ) -> None:
        """TC-CUDA-04: resolve_torch_device('invalid_xyz') raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device target"):
            resolve_torch_device(invalid_target)

    def test_is_cuda_available_exception_safety(self) -> None:
        """Safe probe returns False if torch.cuda.is_available raises an exception."""
        with patch(
            "sign_translator.device.torch.cuda.is_available",
            side_effect=RuntimeError("CUDA driver initialization failed"),
        ):
            assert is_cuda_available() is False


# ---------------------------------------------------------------------------
# TC-CUDA-05 to TC-CUDA-07: ONNX Runtime Provider Resolution
# ---------------------------------------------------------------------------
class TestOrtProviderResolution:
    """Tests for resolve_ort_providers across available execution providers."""

    def test_tc_cuda_05_resolve_ort_providers_auto(self) -> None:
        """TC-CUDA-05: resolve_ort_providers('auto') returns prioritized list based on availability."""
        # Case A: CUDAExecutionProvider available in ORT and CUDA functional
        with (
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=True),
        ):
            providers = resolve_ort_providers("auto")
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Case B: CUDAExecutionProvider in ORT but CUDA not functional
        with (
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=False),
        ):
            providers = resolve_ort_providers("auto")
            assert providers == ["CPUExecutionProvider"]

        # Case C: CUDAExecutionProvider not available in ORT
        with (
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=False),
        ):
            providers = resolve_ort_providers("auto")
            assert providers == ["CPUExecutionProvider"]

    def test_tc_cuda_06_resolve_ort_providers_explicit_cuda(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TC-CUDA-06: resolve_ort_providers behavior with explicit CUDA target or preferred provider."""
        # Available in ORT and CUDA functional
        with (
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=True),
        ):
            assert resolve_ort_providers("cuda") == [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            assert resolve_ort_providers(preferred_provider="CUDAExecutionProvider") == [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        # CUDA requested but unavailable in ORT
        with (
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=False),
        ):
            with caplog.at_level(logging.WARNING):
                providers = resolve_ort_providers("cuda")
                assert providers == ["CPUExecutionProvider"]
                assert "Falling back to CPUExecutionProvider" in caplog.text

            caplog.clear()
            with caplog.at_level(logging.WARNING):
                providers = resolve_ort_providers(preferred_provider="CUDAExecutionProvider")
                assert providers == ["CPUExecutionProvider"]
                assert "not in available providers" in caplog.text

    def test_tc_cuda_07_resolve_ort_providers_cpu(self) -> None:
        """TC-CUDA-07: resolve_ort_providers('cpu') returns ['CPUExecutionProvider'] unconditionally."""
        with (
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=True),
        ):
            assert resolve_ort_providers("cpu") == ["CPUExecutionProvider"]
            assert resolve_ort_providers(preferred_provider="CPUExecutionProvider") == [
                "CPUExecutionProvider"
            ]


# ---------------------------------------------------------------------------
# TC-CUDA-08: GPU Hardware Telemetry
# ---------------------------------------------------------------------------
class TestDeviceTelemetry:
    """Tests for get_device_telemetry schema and metrics."""

    def test_tc_cuda_08_get_device_telemetry_cuda_absent(self) -> None:
        """TC-CUDA-08: Telemetry returns nominal None fields when CUDA is absent."""
        with patch("sign_translator.device.is_cuda_available", return_value=False):
            telemetry = get_device_telemetry()
            assert telemetry["cuda_available"] is False
            assert telemetry["device_count"] == 0
            assert telemetry["gpu_name"] is None
            assert telemetry["vram_allocated_mb"] is None
            assert telemetry["vram_reserved_mb"] is None
            assert telemetry["vram_total_mb"] is None
            assert telemetry["compute_capability"] is None

    def test_tc_cuda_08_get_device_telemetry_cuda_present(self) -> None:
        """TC-CUDA-08: Telemetry queries and returns GPU hardware metrics when CUDA is present."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB = 16384 MB

        with (
            patch("sign_translator.device.is_cuda_available", return_value=True),
            patch("sign_translator.device.torch.cuda.device_count", return_value=1),
            patch(
                "sign_translator.device.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 5060 Ti",
            ),
            patch(
                "sign_translator.device.torch.cuda.get_device_capability",
                return_value=(12, 0),
            ),
            patch(
                "sign_translator.device.torch.cuda.memory_allocated",
                return_value=104857600,  # 100 MB
            ),
            patch(
                "sign_translator.device.torch.cuda.memory_reserved",
                return_value=209715200,  # 200 MB
            ),
            patch(
                "sign_translator.device.torch.cuda.get_device_properties",
                return_value=mock_props,
            ),
        ):
            telemetry = get_device_telemetry()
            assert telemetry["cuda_available"] is True
            assert telemetry["device_count"] == 1
            assert telemetry["gpu_name"] == "NVIDIA GeForce RTX 5060 Ti"
            assert telemetry["compute_capability"] == (12, 0)
            assert telemetry["vram_allocated_mb"] == 100.0
            assert telemetry["vram_reserved_mb"] == 200.0
            assert telemetry["vram_total_mb"] == 16384.0

    def test_tc_cuda_08_get_device_telemetry_exception_resilience(self) -> None:
        """Telemetry queries catch exceptions during hardware attribute inspection."""
        with (
            patch("sign_translator.device.is_cuda_available", return_value=True),
            patch("sign_translator.device.torch.cuda.device_count", return_value=1),
            patch(
                "sign_translator.device.torch.cuda.get_device_name",
                side_effect=RuntimeError("Driver query failure"),
            ),
        ):
            telemetry = get_device_telemetry()
            assert telemetry["cuda_available"] is True
            assert telemetry["gpu_name"] is None


# ---------------------------------------------------------------------------
# TC-CUDA-09: Schema Validation Across Pydantic v2 Models
# ---------------------------------------------------------------------------
class TestSchemaDeviceValidation:
    """Tests for device field validation in PipelineConfig, TrainingConfig, BenchmarkConfig, BenchmarkResult."""

    @pytest.mark.parametrize("valid_dev", ["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    def test_tc_cuda_09_pipeline_config_valid_device(self, valid_dev: str) -> None:
        """TC-CUDA-09: PipelineConfig accepts valid device specifications."""
        cfg = PipelineConfig(device=valid_dev)
        assert cfg.device == valid_dev
        assert cfg.execution_providers is None

    @pytest.mark.parametrize("invalid_dev", ["tpu", "opencl", "cuda:xyz", "cuda:-1", "rocm"])
    def test_tc_cuda_09_pipeline_config_invalid_device(self, invalid_dev: str) -> None:
        """TC-CUDA-09: PipelineConfig rejects invalid device specifications."""
        with pytest.raises(ValidationError, match="Invalid device target"):
            PipelineConfig(device=invalid_dev)

    @pytest.mark.parametrize("valid_dev", ["auto", "cpu", "cuda", "cuda:0"])
    def test_tc_cuda_09_training_config_valid_device(self, valid_dev: str) -> None:
        """TC-CUDA-09: TrainingConfig accepts valid devices and exposes AMP / pin_memory."""
        cfg = TrainingConfig(device=valid_dev, mixed_precision=True, pin_memory=True)
        assert cfg.device == valid_dev
        assert cfg.mixed_precision is True
        assert cfg.pin_memory is True

    @pytest.mark.parametrize("invalid_dev", ["tpu", "invalid_dev", "cuda:xyz"])
    def test_tc_cuda_09_training_config_invalid_device(self, invalid_dev: str) -> None:
        """TC-CUDA-09: TrainingConfig rejects invalid devices."""
        with pytest.raises(ValidationError, match="Invalid training device"):
            TrainingConfig(device=invalid_dev)

    @pytest.mark.parametrize("valid_dev", ["auto", "cpu", "cuda", "cuda:0"])
    def test_tc_cuda_09_benchmark_config_valid_device(self, valid_dev: str) -> None:
        """TC-CUDA-09: BenchmarkConfig accepts valid devices, sweeps batch sizes."""
        cfg = BenchmarkConfig(device=valid_dev, batch_sizes=[1, 8, 32, 64])
        assert cfg.device == valid_dev
        assert cfg.batch_sizes == [1, 8, 32, 64]
        assert cfg.execution_provider == "auto"

    @pytest.mark.parametrize("invalid_dev", ["tpu", "unknown_dev", "cuda:abc"])
    def test_tc_cuda_09_benchmark_config_invalid_device(self, invalid_dev: str) -> None:
        """TC-CUDA-09: BenchmarkConfig rejects invalid device targets."""
        with pytest.raises(ValidationError, match="Invalid device target"):
            BenchmarkConfig(device=invalid_dev)

    @pytest.mark.parametrize("valid_dev", ["cpu", "cuda", "cuda:0"])
    def test_tc_cuda_09_benchmark_result_valid_device_and_telemetry(self, valid_dev: str) -> None:
        """TC-CUDA-09: BenchmarkResult validates device and telemetry attributes."""
        res = BenchmarkResult(
            backend="PyTorchModelRunner",
            device=valid_dev,
            execution_provider="PyTorch-CUDA",
            batch_size=1,
            num_iterations=100,
            mean_latency_ms=1.2,
            median_latency_ms=1.1,
            p95_latency_ms=1.5,
            p99_latency_ms=1.8,
            min_latency_ms=0.9,
            max_latency_ms=2.0,
            throughput_fps=833.3,
            meets_target=True,
            gpu_name="NVIDIA GeForce RTX 5060 Ti",
            vram_allocated_mb=128.5,
        )
        assert res.device == valid_dev
        assert res.gpu_name == "NVIDIA GeForce RTX 5060 Ti"
        assert res.vram_allocated_mb == 128.5

    @pytest.mark.parametrize("invalid_dev", ["tpu", "auto", "invalid"])
    def test_tc_cuda_09_benchmark_result_invalid_device(self, invalid_dev: str) -> None:
        """TC-CUDA-09: BenchmarkResult requires resolved concrete device (cpu, cuda, cuda:<id>)."""
        with pytest.raises(ValidationError, match="Invalid benchmark device"):
            BenchmarkResult(
                backend="PyTorchModelRunner",
                device=invalid_dev,
                execution_provider="CPUExecutionProvider",
                batch_size=1,
                num_iterations=100,
                mean_latency_ms=1.0,
                median_latency_ms=1.0,
                p95_latency_ms=1.0,
                p99_latency_ms=1.0,
                min_latency_ms=1.0,
                max_latency_ms=1.0,
                throughput_fps=1000.0,
                meets_target=True,
            )

    def test_tc_cuda_09_train_result_device_used(self) -> None:
        """TC-CUDA-09: TrainResult captures device_used and gpu_name."""
        res = TrainResult(
            best_val_accuracy=0.98,
            best_val_loss=0.12,
            best_epoch=15,
            total_epochs=20,
            checkpoint_path="models/weights/best_model.pt",
            training_duration_seconds=12.5,
            model_architecture="lstm",
            num_classes=5,
            train_loss_history=[0.5, 0.12],
            val_loss_history=[0.6, 0.15],
            val_acc_history=[0.8, 0.98],
            device_used="cuda:0",
            gpu_name="NVIDIA GeForce RTX 5060 Ti",
        )
        assert res.device_used == "cuda:0"
        assert res.gpu_name == "NVIDIA GeForce RTX 5060 Ti"


# ---------------------------------------------------------------------------
# TC-CUDA-10: Mixed Precision Training & GradScaler Logic
# ---------------------------------------------------------------------------
class TestMixedPrecisionAndTraining:
    """Tests for train_epoch, evaluate, and train_model with AMP and GPU configuration."""

    def test_tc_cuda_10_train_epoch_with_amp_scaler(self) -> None:
        """TC-CUDA-10: train_epoch exercises GradScaler scale, unscale, step, and update."""
        model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=3, num_layers=1)
        dataset = TensorDataset(torch.randn(8, 30, 126), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        mock_scaler = MagicMock()

        loss, acc = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=mock_scaler,
            use_amp=True,
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        # Scaler operations should be invoked once per batch (2 batches)
        assert mock_scaler.scale.call_count == 2
        assert mock_scaler.unscale_.call_count == 2
        assert mock_scaler.step.call_count == 2
        assert mock_scaler.update.call_count == 2

    def test_tc_cuda_10_train_epoch_without_amp(self) -> None:
        """TC-CUDA-10: train_epoch behaves nominally when use_amp is False."""
        model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=3, num_layers=1)
        dataset = TensorDataset(torch.randn(8, 30, 126), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        mock_scaler = MagicMock()

        loss, acc = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=mock_scaler,
            use_amp=False,
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        # Scaler must not be used when use_amp is False
        assert mock_scaler.scale.call_count == 0

    def test_tc_cuda_10_evaluate_with_amp(self) -> None:
        """TC-CUDA-10: evaluate runs properly with use_amp=True."""
        model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=3, num_layers=1)
        dataset = TensorDataset(torch.randn(4, 30, 126), torch.tensor([0, 1, 2, 0]))
        loader = DataLoader(dataset, batch_size=2)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, acc = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            use_amp=True,
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_tc_cuda_10_train_model_cpu_and_gpu_config(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TC-CUDA-10: train_model executes on CPU and records device telemetry."""
        p_cfg = PipelineConfig(
            sequence_length=10,
            vocabulary=["hello", "world"],
            hidden_dim=16,
            num_layers=1,
        )
        t_cfg = TrainingConfig(
            epochs=2,
            batch_size=4,
            device="cpu",
            checkpoint_path=str(tmp_path / "model.pt"),  # type: ignore[operator]
        )

        model, result = train_model(
            pipeline_config=p_cfg,
            training_config=t_cfg,
            dataset=None,
        )

        assert isinstance(result, TrainResult)
        assert result.device_used == "cpu"
        assert result.gpu_name is None
        assert result.total_epochs >= 1


# ---------------------------------------------------------------------------
# TC-CUDA-11: Model Runner Device Propagation and Fallback
# ---------------------------------------------------------------------------
class TestModelRunnerDeviceAndFallback:
    """Tests for PyTorchModelRunner and ONNXModelRunner device binding and graceful fallback."""

    def test_tc_cuda_11_pytorch_runner_device_and_prediction(self) -> None:
        """TC-CUDA-11: PyTorchModelRunner resolves device and computes probabilities."""
        model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=3, num_layers=1)
        runner = PyTorchModelRunner(model, device="cpu")
        assert runner.device == torch.device("cpu")

        seq = np.random.randn(30, 126).astype(np.float32)
        probs = runner.predict_proba(seq)
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

    def test_tc_cuda_11_pytorch_runner_cuda_fallback_when_unavailable(self) -> None:
        """TC-CUDA-11: PyTorchModelRunner falls back to CPU when CUDA is requested but absent."""
        with patch("sign_translator.device.torch.cuda.is_available", return_value=False):
            model = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=3, num_layers=1)
            runner = PyTorchModelRunner(model, device="cuda")
            assert runner.device == torch.device("cpu")

    def test_tc_cuda_11_onnx_runner_provider_fallback(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TC-CUDA-11: ONNXModelRunner catches CUDA provider failure and falls back to CPU."""
        dummy_onnx_file = tmp_path / "dummy.onnx"  # type: ignore[operator]
        dummy_onnx_file.write_bytes(b"dummy_content")

        mock_session_cpu = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_sequence"
        mock_input.shape = [1, 30, 126]
        mock_output = MagicMock()
        mock_output.name = "logits"

        mock_session_cpu.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_cpu.get_inputs.return_value = [mock_input]
        mock_session_cpu.get_outputs.return_value = [mock_output]

        def mock_session_init(path: str, sess_options: object, providers: list[str]) -> MagicMock:
            if "CUDAExecutionProvider" in providers:
                raise RuntimeError("CUDA provider failed to load dependencies")
            return mock_session_cpu

        with (
            patch("onnxruntime.InferenceSession", side_effect=mock_session_init),
            patch(
                "sign_translator.device.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("sign_translator.device.is_cuda_available", return_value=True),
        ):
            runner = ONNXModelRunner(dummy_onnx_file, execution_provider="CUDAExecutionProvider")
            assert runner.active_provider == "CPUExecutionProvider"
            assert runner.device == "cpu"


# ---------------------------------------------------------------------------
# TC-CUDA-12: Inference Benchmarking & CUDA Synchronization
# ---------------------------------------------------------------------------
class TestBenchmarkInferenceCuda:
    """Tests for benchmark_inference synchronization and telemetry population."""

    def test_tc_cuda_12_benchmark_inference_cuda_synchronization(self) -> None:
        """TC-CUDA-12: benchmark_inference invokes torch.cuda.synchronize and records GPU telemetry."""
        mock_runner = MagicMock()
        mock_runner._session = None
        mock_runner.session = None
        mock_runner.device = "cuda:0"
        mock_runner.predict_proba.return_value = np.array([0.7, 0.2, 0.1], dtype=np.float32)

        config = BenchmarkConfig(
            num_warmup=2,
            num_iterations=10,
            device="cuda",
        )

        mock_telemetry = {
            "cuda_available": True,
            "device_count": 1,
            "gpu_name": "NVIDIA GeForce RTX 5060 Ti",
            "compute_capability": (12, 0),
            "vram_allocated_mb": 150.5,
            "vram_reserved_mb": 256.0,
            "vram_total_mb": 16384.0,
        }

        with (
            patch("sign_translator.export_onnx.torch.cuda.is_available", return_value=True),
            patch("sign_translator.export_onnx.torch.cuda.synchronize") as mock_sync,
            patch("sign_translator.export_onnx.get_device_telemetry", return_value=mock_telemetry),
        ):
            result = benchmark_inference(
                runner=mock_runner,
                sequence_length=30,
                feature_dim=126,
                config=config,
            )

            assert isinstance(result, BenchmarkResult)
            assert result.device == "cuda:0"
            assert result.gpu_name == "NVIDIA GeForce RTX 5060 Ti"
            assert result.vram_allocated_mb == 150.5
            # torch.cuda.synchronize called during warmup (1) and iterations (2 per iteration: before and after)
            assert mock_sync.call_count >= 5

    def test_tc_cuda_12_benchmark_inference_cpu_no_cuda_sync(self) -> None:
        """TC-CUDA-12: benchmark_inference on CPU does not invoke torch.cuda.synchronize."""
        mock_runner = MagicMock()
        mock_runner._session = None
        mock_runner.session = None
        mock_runner.device = "cpu"
        mock_runner.predict_proba.return_value = np.array([0.8, 0.2], dtype=np.float32)

        config = BenchmarkConfig(
            num_warmup=2,
            num_iterations=10,
            device="cpu",
        )

        with (
            patch("sign_translator.export_onnx.torch.cuda.is_available", return_value=False),
            patch("sign_translator.export_onnx.torch.cuda.synchronize") as mock_sync,
            patch(
                "sign_translator.export_onnx.get_device_telemetry",
                return_value={"cuda_available": False, "gpu_name": None, "vram_allocated_mb": None},
            ),
        ):
            result = benchmark_inference(
                runner=mock_runner,
                sequence_length=30,
                feature_dim=126,
                config=config,
            )

            assert isinstance(result, BenchmarkResult)
            assert result.device == "cpu"
            assert result.gpu_name is None
            assert result.vram_allocated_mb is None
            assert mock_sync.call_count == 0

    def test_tc_cuda_12_multi_batch_parity_verification_helper(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TC-CUDA-12: verify_onnx_parity verifies numerical equivalence across multiple batch sizes."""
        dummy_onnx = tmp_path / "dummy.onnx"  # type: ignore[operator]
        dummy_onnx.write_bytes(b"dummy_onnx_bytes")

        mock_torch_model = MagicMock()
        mock_torch_model.eval.return_value = None

        def mock_torch_forward(tensor: torch.Tensor) -> torch.Tensor:
            b = tensor.shape[0]
            # Deterministic linear transform
            return torch.ones((b, 3), dtype=torch.float32)

        mock_torch_model.side_effect = mock_torch_forward

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input_seq")]
        mock_session.get_inputs()[0].name = "input_sequence"
        mock_session.get_outputs.return_value = [MagicMock(name="logits")]
        mock_session.get_outputs()[0].name = "logits"

        def mock_ort_run(
            output_names: list[str], feed_dict: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            arr = list(feed_dict.values())[0]
            b = arr.shape[0]
            return [np.ones((b, 3), dtype=np.float32)]

        mock_session.run.side_effect = mock_ort_run

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            diff = verify_onnx_parity(
                torch_model=mock_torch_model,
                onnx_path=dummy_onnx,
                sequence_length=30,
                feature_dim=126,
                batch_sizes=(1, 8, 32, 64),
                atol=1e-5,
            )
            assert diff <= 1e-5
