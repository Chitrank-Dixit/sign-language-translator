# Real-Time Sign Language & Gesture Translator

A real-time spatial-temporal sign language and gesture recognition pipeline capturing 30-frame temporal sequences via video or webcam, extracting 3D hand keypoints via MediaPipe, normalizing spatial coordinates for scale/distance invariance, classifying gestures with PyTorch / ONNX Runtime (LSTM or Transformer), and outputting debounced text and Text-to-Speech (TTS) audio at 30+ FPS.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │                 Video Input                   │
                      │   (Synthetic Stream / Video File / Webcam)    │
                      └──────────────────────┬────────────────────────┘
                                             │ (BGR Frames)
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │           HandLandmarkExtractor               │
                      │  - MediaPipe 21 3D Landmarks per Hand (126-D) │
                      │  - Deterministic Handedness Slot Assignment   │
                      │  - Wrist-Relative Centering & Scale Norm      │
                      └──────────────────────┬────────────────────────┘
                                             │ (126-D Normalized Features)
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │          Sliding Window Buffer (deque)        │
                      │         (B=1, T=30 Frames, D=126 Dim)         │
                      └──────────────────────┬────────────────────────┘
                                             │ (30-Frame Sequence Tensor)
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │          ONNXModelRunner / PyTorch            │
                      │   - Dynamic Batch Axis (B=1, 4, 16)           │
                      │   - Single-Sequence CPU Latency < 0.1 ms      │
                      │   - Numerically Stable Softmax Probabilities  │
                      └──────────────────────┬────────────────────────┘
                                             │ (Raw Class Probabilities)
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │              PredictionSmoother               │
                      │   - M-of-K Majority Voting (M=6, K=10)        │
                      │   - Confidence Filtering (tau >= 0.85)        │
                      │   - Per-Label Cooldown Debouncing (2.0s)      │
                      └──────────────┬─────────────────┬──────────────┘
                                     │                 │
              (Stabilized Gesture)   │                 │ (Audio Trigger)
                                     ▼                 ▼
                      ┌──────────────────────┐  ┌─────────────────────┐
                      │   Console UI / API   │  │   AsyncTTSWorker    │
                      │ (GesturePrediction)  │  │ (Non-blocking queue)│
                      └──────────────────────┘  └─────────────────────┘
```

---

## Universal Stack Standards

This project adheres to the universal Spec-Driven Development (SDD) standard:
- **Container Runtime**: Podman rootless container (`Containerfile`, `compose.yaml`).
- **Package Management**: `uv` package manager with `pyproject.toml` and locked dependencies (`uv.lock`).
- **Data Contracts**: Pydantic v2 schemas (`BaseModel`) with `frozen=True` and `@field_validator` / `@model_validator`.
- **Code Quality**: Ruff linter and formatter (`ruff check`, `ruff format`).
- **Testing**: Containerized Pytest suite with mock frame synthesis (`tests/test_*.py`).

---

## How to Run the Application

The application supports **three execution modes**:

### Mode 1: Synthetic Stream (Instant Test / Zero Setup)
Run the real-time inference pipeline using the built-in synthetic frame generator without needing any video file or physical webcam:

```bash
# Using Makefile
make run

# Or directly via Podman in WSL
podman.exe run --rm -v "$(wslpath -w .):/app:Z" sign_language_translator-app uv run python -m sign_translator.cli --source synthetic --max-frames 50
```

### Mode 2: Run with an Uploaded Video File
If you have a pre-recorded sign language video file (`.mp4`, `.avi`, `.mov`, etc.):

1. Place your video file inside the project directory (e.g., `data/my_sign.mp4`).
2. Run:
```bash
# Using Makefile
make run-video VIDEO=data/my_sign.mp4

# Or directly via Podman in WSL
podman.exe run --rm -v "$(wslpath -w .):/app:Z" sign_language_translator-app uv run python -m sign_translator.cli --source data/my_sign.mp4
```

### Mode 3: Run with a Live Webcam Stream
Run live inference continuously using your connected webcam:

```bash
# Using Makefile (defaults to camera index 0)
make run-camera

# Or directly via Podman in WSL
podman.exe run --rm -v "$(wslpath -w .):/app:Z" sign_language_translator-app uv run python -m sign_translator.cli --source 0
```

---

## CLI Options Reference (`sign_translator.cli`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | `synthetic` | Input source: `'synthetic'`, video file path (e.g. `data/gesture.mp4`), or camera device index (e.g. `0`) |
| `--model` | `str` | `models/exported/gesture_model.onnx` | Path to trained model: exported `.onnx` graph or PyTorch `.pt` checkpoint |
| `--max-frames` | `int` | `None` | Upper limit on frames to process (default: runs until stream ends or `Ctrl+C`) |
| `--no-tts` | `flag` | `False` | Disables audio speech synthesis (text-only console output) |
| `--cooldown` | `float` | `2.0` | Minimum elapsed time in seconds between speech audio triggers |

---

## Model Training & ONNX Export Workflow

### 1. Train Sequence Model
Train the spatial-temporal model (Bi-LSTM or Transformer) on synthetic or recorded sequences:
```bash
make train
# Saves best checkpoint to models/weights/best_model.pt
```

### 2. Export Model to ONNX
Serialize the PyTorch checkpoint to ONNX with a dynamic batch axis and verify numerical parity (`diff < 1e-5`):
```bash
make export-onnx
# Saves graph to models/exported/gesture_model.onnx
```

### 3. Profile CPU Inference Latency
Benchmark the ONNX model to verify it meets the sub-5ms CPU latency acceptance criteria:
```bash
make benchmark
# Emits mean latency (~0.07 ms), throughput (>13,000 FPS), and p95/p99 percentiles
```

---

## Quickstart & Makefile Commands

A unified [Makefile](file:///d:/Projects/sign_language_translator/Makefile) is provided with full cross-environment auto-detection (WSL using `podman.exe` and native Linux/CI using `podman`):

```bash
make help          # View all available targets and descriptions
make build         # Build the Podman container image
make test          # Run the full 141-test suite inside Podman
make test-cov      # Run tests with code coverage reporting
make lint          # Run Ruff linter
make lint-fix      # Run Ruff linter with autofix
make format        # Format codebase with Ruff
make format-check  # Verify code formatting
make check         # Run both format-check and lint
make train         # Train sequence classification model
make export-onnx   # Export PyTorch checkpoint to ONNX with dynamic batching
make benchmark     # Benchmark ONNX CPU latency (target < 5.0ms)
make smoke-test    # Run real-time synthetic stream smoke test
make run           # Run CLI translator on synthetic video stream
make run-video     # Run CLI translator on video file (VIDEO=path/to/video.mp4)
make run-camera    # Run CLI translator on live webcam stream
make shell         # Open interactive bash shell inside container
make clean         # Remove caches and temporary test artifacts
```

---

## Project Milestones Completed

- [x] **Milestone 1**: Environment & Scaffold (`pyproject.toml`, `Containerfile`, `compose.yaml`, `schemas.py`, `test_schemas.py`)
- [x] **Milestone 2**: Data Collection & Landmark Normalization (`collect_data.py`, `normalize.py`, `test_normalize.py`, `test_collect_data.py`)
- [x] **Milestone 3**: Spatial-Temporal Sequence Model & Training (`architecture.py`, `train.py`, `test_model.py`)
- [x] **Milestone 4**: Real-Time Inference, Smoothing & Non-Blocking TTS (`live_inference.py`, `test_inference.py`)
- [x] **Milestone 5**: ONNX Export & 30+ FPS Edge Optimization (`export_onnx.py`, `test_onnx.py`, `cli.py`, `Makefile`)
