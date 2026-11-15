# ==============================================================================
# Makefile: Real-Time Sign Language & Gesture Translator
# ==============================================================================

.DEFAULT_GOAL := help
.PHONY: help build test test-cov lint lint-fix format format-check check train train-gpu export-onnx benchmark benchmark-gpu smoke-test run run-gpu run-video run-camera shell clean

# ------------------------------------------------------------------------------
# Container & Environment Configuration
# ------------------------------------------------------------------------------
IMAGE_NAME ?= sign_language_translator-app

# Autodetect WSL vs native Linux / CI
HAS_PODMAN_EXE := $(shell which podman.exe 2>/dev/null)
HAS_WSLPATH    := $(shell which wslpath 2>/dev/null)

ifneq ($(HAS_PODMAN_EXE),)
  PODMAN ?= podman.exe
  ifneq ($(HAS_WSLPATH),)
    MOUNT_PATH ?= $(shell wslpath -w "$$(pwd)")
  else
    MOUNT_PATH ?= $(CURDIR)
  endif
else
  PODMAN ?= podman
  MOUNT_PATH ?= $(CURDIR)
endif

CONTAINER_RUN ?= $(PODMAN) run --rm -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)

# ------------------------------------------------------------------------------
# Help Target
# ------------------------------------------------------------------------------
help:
	@echo "================================================================================"
	@echo " Real-Time Sign Language & Gesture Translator - Make Commands"
	@echo "================================================================================"
	@echo " Container & Environment:"
	@echo "   make build         - Build the Podman container image via Containerfile"
	@echo "   make shell         - Open an interactive bash shell inside the container"
	@echo ""
	@echo " Testing & Quality Control:"
	@echo "   make test          - Run full pytest test suite (141 tests) inside Podman"
	@echo "   make test-cov      - Run pytest with code coverage report"
	@echo "   make lint          - Run Ruff linter check"
	@echo "   make lint-fix      - Run Ruff linter with automatic fix"
	@echo "   make format        - Format codebase with Ruff"
	@echo "   make format-check  - Verify code formatting with Ruff"
	@echo "   make check         - Run both format-check and lint"
	@echo ""
	@echo " Model & Pipeline Operations:"
	@echo "   make train         - Train sequence model (LSTM / Transformer)"
	@echo "   make train-gpu     - Train sequence model with CUDA acceleration & AMP"
	@echo "   make export-onnx   - Export PyTorch checkpoint to ONNX with dynamic batching"
	@echo "   make benchmark     - Run ONNX CPU latency benchmark (target < 5.0 ms)"
	@echo "   make benchmark-gpu - Run ONNX GPU latency benchmark (CUDAExecutionProvider)"
	@echo "   make smoke-test    - Run real-time inference pipeline synthetic smoke test"
	@echo "   make run           - Run CLI translator on synthetic video stream"
	@echo "   make run-gpu       - Run CLI translator on synthetic stream with CUDA device"
	@echo "   make run-video     - Run CLI translator on a video file (VIDEO=path.mp4)"
	@echo "   make run-camera    - Run CLI translator on live webcam stream"
	@echo ""
	@echo " Maintenance:"
	@echo "   make clean         - Remove bytecode, cache directories, and test artifacts"
	@echo "================================================================================"

# ------------------------------------------------------------------------------
# Container Targets
# ------------------------------------------------------------------------------
build:
	@echo "==> Building Podman container image [$(IMAGE_NAME)]..."
	$(PODMAN) build -t $(IMAGE_NAME) -f Containerfile .

shell:
	@echo "==> Launching interactive shell inside [$(IMAGE_NAME)]..."
	$(PODMAN) run --rm -it -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME) /bin/bash

# ------------------------------------------------------------------------------
# Testing & Code Quality Targets
# ------------------------------------------------------------------------------
test:
	@echo "==> Running pytest test suite inside Podman..."
	$(CONTAINER_RUN) uv run pytest -v

test-cov:
	@echo "==> Running pytest with coverage reporting inside Podman..."
	$(CONTAINER_RUN) uv run pytest -v --cov=sign_translator --cov-report=term-missing

lint:
	@echo "==> Running Ruff linter check..."
	$(CONTAINER_RUN) uv run ruff check .

lint-fix:
	@echo "==> Running Ruff linter with autofix..."
	$(CONTAINER_RUN) uv run ruff check --fix .

format:
	@echo "==> Formatting code with Ruff..."
	$(CONTAINER_RUN) uv run ruff format .

format-check:
	@echo "==> Checking code formatting with Ruff..."
	$(CONTAINER_RUN) uv run ruff format --check .

check: format-check lint

# ------------------------------------------------------------------------------
# Training, Export & Benchmark Targets
# ------------------------------------------------------------------------------
train:
	@echo "==> Running sequence model training pipeline..."
	$(CONTAINER_RUN) uv run python -m sign_translator.train

train-gpu:
	@echo "==> Running sequence model training with CUDA acceleration..."
	$(CONTAINER_RUN) uv run python -c "from sign_translator.train import train_model; from sign_translator.schemas import TrainingConfig; train_model(TrainingConfig(device='cuda', mixed_precision=True))"

export-onnx:
	@echo "==> Exporting PyTorch model to ONNX format..."
	$(CONTAINER_RUN) uv run python -m sign_translator.export_onnx

benchmark:
	@echo "==> Benchmarking ONNX inference latency on CPU..."
	$(CONTAINER_RUN) uv run python -c "from sign_translator.live_inference import ONNXModelRunner; from sign_translator.export_onnx import benchmark_inference; runner = ONNXModelRunner('models/exported/gesture_model.onnx'); res = benchmark_inference(runner); print('Mean Latency:', round(res.mean_latency_ms, 4), 'ms | Throughput:', round(res.throughput_fps), 'FPS | Target < 5.0ms:', res.meets_target)"

benchmark-gpu:
	@echo "==> Benchmarking ONNX inference latency on GPU (CUDAExecutionProvider)..."
	$(CONTAINER_RUN) uv run python -c "from sign_translator.live_inference import ONNXModelRunner; from sign_translator.export_onnx import benchmark_inference; from sign_translator.schemas import BenchmarkConfig; runner = ONNXModelRunner('models/exported/gesture_model.onnx', device='cuda'); res = benchmark_inference(runner, config=BenchmarkConfig(device='cuda', execution_provider='CUDAExecutionProvider')); print('Backend:', res.backend, '| Device:', res.device, '| Provider:', res.execution_provider, '| Mean Latency:', round(res.mean_latency_ms, 4), 'ms | FPS:', round(res.throughput_fps))"

smoke-test:
	@echo "==> Running real-time inference synthetic stream smoke test..."
	$(CONTAINER_RUN) uv run python -c "from sign_translator.live_inference import RealtimeGesturePipeline; from sign_translator.collect_data import SyntheticCaptureSource; p = RealtimeGesturePipeline(capture_source=SyntheticCaptureSource(max_frames=35), gesture_classes=['hello', 'thank_you', 'yes', 'no']); res = p.run(max_frames=35); print('Processed', len(res), 'frames successfully')"

run:
	@echo "==> Running gesture translator CLI (synthetic stream)..."
	$(CONTAINER_RUN) uv run python -m sign_translator.cli --source synthetic --max-frames 50

run-gpu:
	@echo "==> Running gesture translator CLI with CUDA acceleration..."
	$(CONTAINER_RUN) uv run python -m sign_translator.cli --source synthetic --max-frames 50 --device cuda

VIDEO ?= data/sample_gesture.mp4

run-video:
	@echo "==> Running gesture translator on video file [$(VIDEO)]..."
	$(CONTAINER_RUN) uv run python -m sign_translator.cli --source $(VIDEO)

run-camera:
	@echo "==> Running gesture translator on webcam..."
	$(CONTAINER_RUN) uv run python -m sign_translator.cli --source 0

# ------------------------------------------------------------------------------
# Cleanup Target
# ------------------------------------------------------------------------------
clean:
	@echo "==> Cleaning cache directories and temporary test artifacts..."
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
