"""ONNX graph export, dynamic batching, opset verification, and inference benchmarking.

Serializes PyTorch sequence classification backbones (LSTM, Transformer) into
standard ONNX format with dynamic batch axes, validates numerical parity
against PyTorch logits, and benchmarks CPU inference latency.
"""

from __future__ import annotations

import inspect
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from sign_translator.schemas import (
    BenchmarkConfig,
    BenchmarkResult,
    ExportResult,
    ONNXExportConfig,
    PipelineConfig,
)
from sign_translator.train import load_checkpoint

if TYPE_CHECKING:
    from sign_translator.live_inference import SequenceModelRunner


def export_torch_to_onnx(
    model_or_checkpoint: nn.Module | Path | str,
    config: ONNXExportConfig,
    pipeline_config: PipelineConfig | None = None,
) -> ExportResult:
    """Export a PyTorch sequence classification model or checkpoint to an ONNX graph.

    Configures dynamic batch axes ('batch_size'), applies constant folding,
    validates the exported graph with onnx.checker, and performs logit-level
    parity checks against the original PyTorch model.

    Args:
        model_or_checkpoint: Instantiated nn.Module or filesystem path to .pt checkpoint.
        config: ONNX export configuration and parity tolerances.
        pipeline_config: Optional pipeline configuration (inferred from checkpoint if omitted).

    Returns:
        ExportResult metadata contract detailing export artifacts and parity metrics.

    Raises:
        FileNotFoundError: If a checkpoint path is specified but does not exist.
        ValueError: If graph export fails verification or exceeds atol parity threshold.
    """
    t_start = time.perf_counter()

    # Resolve PyTorch model and metadata
    architecture_name = "unknown"
    if isinstance(model_or_checkpoint, (str, Path)):
        ckpt_path = Path(model_or_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt_path}")
        model, ckpt_meta = load_checkpoint(ckpt_path)
        if pipeline_config is None and "pipeline_config" in ckpt_meta:
            pipeline_config = PipelineConfig.model_validate(ckpt_meta["pipeline_config"])
        architecture_name = str(ckpt_meta.get("model_architecture", "unknown"))
    elif isinstance(model_or_checkpoint, nn.Module):
        model = model_or_checkpoint
        if pipeline_config is not None:
            architecture_name = pipeline_config.architecture.value
        else:
            architecture_name = "lstm" if "LSTM" in model.__class__.__name__ else "transformer"
    else:
        raise TypeError(f"Unsupported model_or_checkpoint type: {type(model_or_checkpoint)}")

    model.eval()

    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare dummy input tensor (B=1, T=sequence_length, D=feature_dim)
    seq_len = pipeline_config.sequence_length
    feat_dim = pipeline_config.feature_dim
    dummy_input = torch.zeros((1, seq_len, feat_dim), dtype=torch.float32)

    # Dynamic axes mapping
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if config.dynamic_batch:
        dynamic_axes = {
            config.input_names[0]: {0: "batch_size"},
            config.output_names[0]: {0: "batch_size"},
        }

    # Execute PyTorch ONNX export
    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": config.opset_version,
        "do_constant_folding": config.optimize_model,
        "input_names": config.input_names,
        "output_names": config.output_names,
        "dynamic_axes": dynamic_axes,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        **export_kwargs,
    )

    # Validate exported ONNX graph structure
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Parity verification
    max_abs_diff = 0.0
    parity_verified = True
    if config.verify_export:
        max_abs_diff = verify_onnx_parity(
            torch_model=model,
            onnx_path=output_path,
            sequence_length=seq_len,
            feature_dim=feat_dim,
            batch_sizes=(1, 4),
            atol=config.atol,
        )
        parity_verified = max_abs_diff <= config.atol

    file_size = output_path.stat().st_size
    duration = time.perf_counter() - t_start

    return ExportResult(
        checkpoint_path=str(config.checkpoint_path),
        output_path=str(output_path),
        file_size_bytes=file_size,
        architecture=architecture_name,
        opset_version=config.opset_version,
        dynamic_axes=dynamic_axes or {},
        max_absolute_difference=float(max_abs_diff),
        parity_verified=parity_verified,
        export_duration_seconds=float(duration),
        timestamp=datetime.now(UTC),
    )


def verify_onnx_parity(
    torch_model: nn.Module,
    onnx_path: Path | str,
    sequence_length: int = 30,
    feature_dim: int = 126,
    batch_sizes: tuple[int, ...] = (1, 4),
    atol: float = 1e-5,
) -> float:
    """Verify numerical parity between PyTorch and ONNX Runtime logits across batch sizes.

    Args:
        torch_model: PyTorch model instance in eval mode.
        onnx_path: Filesystem path to exported ONNX graph.
        sequence_length: Sequence length (T) of input frames.
        feature_dim: Feature dimension (D) per frame.
        batch_sizes: Tuple of mini-batch sizes to validate dynamic batching.
        atol: Maximum allowed absolute numerical difference.

    Returns:
        Peak absolute difference observed across all evaluated batch sizes.

    Raises:
        FileNotFoundError: If onnx_path does not exist.
        ValueError: If absolute difference exceeds atol for any tested batch size.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    torch_model.eval()

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 2
    session_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        session_opts,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    max_overall_diff = 0.0

    for b in batch_sizes:
        # Deterministic synthetic test sequence
        np.random.seed(42 + b)
        sample_np = np.random.randn(b, sequence_length, feature_dim).astype(np.float32)

        # PyTorch forward pass
        with torch.no_grad():
            pt_out = torch_model(torch.from_numpy(sample_np)).cpu().numpy()

        # ONNX forward pass
        ort_out = session.run([output_name], {input_name: sample_np})[0]

        diff = float(np.max(np.abs(pt_out - ort_out)))
        max_overall_diff = max(max_overall_diff, diff)

        if diff > atol:
            raise ValueError(
                f"Parity verification failed for batch size {b}: max diff {diff} > atol {atol}"
            )

    return max_overall_diff


def benchmark_inference(
    runner: SequenceModelRunner,
    sequence_length: int = 30,
    feature_dim: int = 126,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Profile inference latency and throughput for a sequence classification runner.

    Executes cache warmup iterations followed by high-resolution time profiling
    using time.perf_counter().

    Args:
        runner: Object implementing SequenceModelRunner protocol.
        sequence_length: Temporal frame count (T).
        feature_dim: Coordinates feature dimension (D).
        config: Benchmark configuration contract; defaults to BenchmarkConfig().

    Returns:
        BenchmarkResult statistical summary with mean, percentiles, and pass/fail flag.
    """
    cfg = config or BenchmarkConfig()

    np.random.seed(42)
    if cfg.batch_size == 1:
        sample_input = np.random.randn(sequence_length, feature_dim).astype(np.float32)
    else:
        sample_input = np.random.randn(cfg.batch_size, sequence_length, feature_dim).astype(
            np.float32
        )

    # Warmup phase
    for _ in range(cfg.num_warmup):
        _ = runner.predict_proba(sample_input)

    # Measurement phase
    latencies: list[float] = []
    for _ in range(cfg.num_iterations):
        t0 = time.perf_counter()
        _ = runner.predict_proba(sample_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(statistics.mean(latencies))
    median_lat = float(statistics.median(latencies))
    sorted_lats = sorted(latencies)

    # 95th and 99th percentiles
    p95_idx = min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)
    p99_idx = min(int(len(sorted_lats) * 0.99), len(sorted_lats) - 1)
    p95_lat = float(sorted_lats[p95_idx])
    p99_lat = float(sorted_lats[p99_idx])
    min_lat = float(sorted_lats[0])
    max_lat = float(sorted_lats[-1])

    fps = float(1000.0 / mean_lat * cfg.batch_size) if mean_lat > 0 else 0.0

    backend_name = runner.__class__.__name__

    return BenchmarkResult(
        backend=backend_name,
        batch_size=cfg.batch_size,
        num_iterations=cfg.num_iterations,
        mean_latency_ms=mean_lat,
        median_latency_ms=median_lat,
        p95_latency_ms=p95_lat,
        p99_latency_ms=p99_lat,
        min_latency_ms=min_lat,
        max_latency_ms=max_lat,
        throughput_fps=fps,
        meets_target=mean_lat <= cfg.target_latency_ms,
        timestamp=datetime.now(UTC),
    )


if __name__ == "__main__":
    from sign_translator.architecture import build_model
    from sign_translator.live_inference import ONNXModelRunner

    pipeline_cfg = PipelineConfig()
    model = build_model(pipeline_cfg)
    export_cfg = ONNXExportConfig(
        checkpoint_path=pipeline_cfg.model_weights_path,
        output_path=pipeline_cfg.onnx_model_path,
    )
    print(f"Exporting model to ONNX: {export_cfg.output_path}...")
    res = export_torch_to_onnx(model, export_cfg, pipeline_config=pipeline_cfg)
    print(
        f"Export succeeded! File size: {res.file_size_bytes} bytes, "
        f"parity verified: {res.parity_verified}"
    )
    runner = ONNXModelRunner(export_cfg.output_path)
    bench = benchmark_inference(runner)
    print(
        f"Benchmark: mean latency {bench.mean_latency_ms:.4f} ms "
        f"({bench.throughput_fps:.0f} FPS), meets target: {bench.meets_target}"
    )
