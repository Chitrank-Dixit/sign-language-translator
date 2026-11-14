# SPEC: Real-Time Sign Language & Gesture Translator

## 1. System Overview
A real-time spatial-temporal sign language recognition pipeline capturing 30-frame sequences via webcam, extracting 3D hand keypoints via MediaPipe, normalizing spatial coordinates for distance/scale invariance, classifying sequences via PyTorch (LSTM/Transformer), and outputting debounced text and Text-to-Speech (TTS) audio at 30+ FPS via ONNX Runtime.

## 2. Architecture & Data Contracts (`src/sign_translator/schemas.py`)
- `PipelineConfig(BaseSettings)`: Configurable sequence length (`T=30`), `num_hands=2`, `landmarks_per_hand=21`, `coords=3` (`feature_dim = 126`), gesture vocabulary, model hyperparameters, smoothing window (`K=10`), and confidence threshold (`0.85`).
- `SequenceMetadata(BaseModel)`: Tracks `gesture_label`, `sequence_id`, `num_frames`, `feature_dim`, `normalized: bool`.
- `GesturePrediction(BaseModel)`: Outputs `label`, `confidence`, `is_stable`, and `latency_ms`.

## 3. Functional Milestones & Acceptance Criteria

### Milestone 1: Environment & Scaffold
- **Deliverables:** `pyproject.toml`, `Containerfile`, `podman-compose.yml`, `src/sign_translator/schemas.py`, `tests/test_schemas.py`.
- **Acceptance Criteria:** Podman container builds cleanly via `uv`, `ruff check` passes, and Pydantic schema validation tests pass.

### Milestone 2: Data Collection & Landmark Normalization
- **Deliverables:** `src/sign_translator/collect_data.py`, `src/sign_translator/normalize.py`, `tests/test_normalize.py`.
- **Requirements:**
  - Extract 21 `(x, y, z)` landmarks per hand (zero-padded to 126 features if one hand is absent).
  - **Normalization Math:** Translate all landmarks relative to each hand's wrist (`landmark[0]`), then divide by the maximum L2 Euclidean distance from the wrist to scale-normalize into `[-1, 1]`.
- **Acceptance Criteria:**
  - Unit tests prove that scaling coordinates by `2.5x` or shifting `(x, y, z)` by `+100` produces identical normalized tensors (`np.allclose` within `1e-6`).
  - Missing/all-zero hands do not cause `ZeroDivisionError` / `NaN`.

### Milestone 3: Spatial-Temporal Sequence Model & Training
- **Deliverables:** `src/sign_translator/architecture.py`, `src/sign_translator/train.py`, `tests/test_model.py`.
- **Requirements:**
  - Support selectable `LSTMClassifier` (2-layer bidirectional/unidirectional LSTM + dropout + linear head) and `TransformerClassifier` (linear projection + positional encoding + TransformerEncoder).
  - Stratified train/val split, Adam optimizer, CrossEntropyLoss, saving best checkpoint to `models/weights/best_model.pt`.
- **Acceptance Criteria:**
  - Forward-pass unit tests pass for dummy tensor `(batch=4, seq=30, feat=126)` -> `(4, num_classes)` for both architectures.
  - Overfit sanity test achieves >95% accuracy on synthetic separable sequence data within 20 epochs.

### Milestone 4: Real-Time Inference, Smoothing & Non-Blocking TTS
- **Deliverables:** `src/sign_translator/live_inference.py`, `tests/test_inference.py`.
- **Requirements:**
  - Maintain a `collections.deque(maxlen=30)` sliding frame buffer.
  - **Smoothing & Debouncing:** Require at least `M` of the last `K` frames to exceed `confidence_threshold` and implement a cooldown timer so the same word isn't spoken repeatedly every frame.
  - **Async TTS:** Run `pyttsx3` / `gTTS` playback in a background `threading.Thread` or `queue.Queue` worker so audio synthesis never drops OpenCV video FPS.
- **Acceptance Criteria:**
  - Unit tests verify sliding-window debouncing suppresses single-frame false spikes and prevents duplicate TTS triggers during cooldown.

### Milestone 5: ONNX Export & 30+ FPS Edge Optimization
- **Deliverables:** `src/sign_translator/export_onnx.py`, `tests/test_onnx.py`.
- **Requirements:**
  - Export PyTorch checkpoint to `models/exported/gesture_model.onnx` with dynamic batch axis.
  - Integrate `onnxruntime.InferenceSession` option into `live_inference.py`.
- **Acceptance Criteria:**
  - Automated parity test confirms `np.allclose(pytorch_logits, onnx_logits, atol=1e-5)`.
  - Benchmark test verifies single-sequence inference latency `< 5 ms` on CPU.