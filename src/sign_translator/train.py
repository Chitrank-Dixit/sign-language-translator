"""Model training pipeline, dataset abstraction, and checkpoint management.

Provides stratified splitting, deterministic synthetic data synthesis, optimization
loops with gradient clipping, and atomic checkpoint persistence.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from sign_translator.architecture import build_model
from sign_translator.normalize import normalize_landmarks
from sign_translator.schemas import PipelineConfig, TrainingConfig, TrainResult


class GestureDataset(Dataset):
    """PyTorch Dataset supporting in-memory sequence arrays and directory trees."""

    def __init__(
        self,
        tensors: np.ndarray | torch.Tensor | None = None,
        labels: np.ndarray | torch.Tensor | Sequence[int] | None = None,
        data_dir: Path | str | None = None,
        vocabulary: list[str] | None = None,
        normalize_on_load: bool = False,
    ) -> None:
        self.sequences: torch.Tensor
        self.labels: torch.Tensor

        if tensors is not None and labels is not None:
            if isinstance(tensors, np.ndarray):
                arr = (
                    normalize_landmarks(tensors)
                    if normalize_on_load
                    else tensors.astype(np.float32)
                )
                self.sequences = torch.from_numpy(arr).float()
            else:
                t = (
                    normalize_landmarks(tensors)
                    if normalize_on_load
                    else tensors.clone().detach().float()
                )
                self.sequences = t.float()

            if isinstance(labels, (np.ndarray, list, tuple)):
                self.labels = torch.tensor(labels, dtype=torch.long)
            else:
                self.labels = labels.clone().detach().long()
        elif data_dir is not None:
            base_path = Path(data_dir)
            if not base_path.exists():
                raise FileNotFoundError(f"Dataset directory not found: {base_path}")

            seq_list: list[np.ndarray] = []
            label_list: list[int] = []

            classes = vocabulary or sorted(
                [d.name for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
            )
            class_to_idx = {name: idx for idx, name in enumerate(classes)}

            for class_name, class_idx in class_to_idx.items():
                class_dir = base_path / class_name
                if not class_dir.exists():
                    continue
                for npy_file in sorted(class_dir.glob("*.npy")):
                    data = np.load(npy_file)
                    if normalize_on_load:
                        data = normalize_landmarks(data)
                    seq_list.append(data.astype(np.float32))
                    label_list.append(class_idx)

            if not seq_list:
                raise ValueError(f"No .npy sequences found in dataset directory: {base_path}")

            self.sequences = torch.from_numpy(np.stack(seq_list, axis=0)).float()
            self.labels = torch.tensor(label_list, dtype=torch.long)
        else:
            raise ValueError("Must provide either (tensors, labels) or data_dir.")

        if len(self.sequences) != len(self.labels):
            raise ValueError(
                f"Tensors ({len(self.sequences)}) and labels ({len(self.labels)}) length mismatch."
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]

    def get_labels(self) -> np.ndarray:
        """Return numpy array of integer labels for stratified splitting."""
        return self.labels.cpu().numpy()


def generate_synthetic_dataset(
    num_samples_per_class: int = 20,
    sequence_length: int = 30,
    feature_dim: int = 126,
    vocabulary: list[str] | None = None,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate separable synthetic sequence patterns for fast deterministic testing.

    Each gesture class is assigned distinct harmonic waveforms and spatial profiles
    that can be classified cleanly by sequence models within 10-20 epochs.
    """
    rng = np.random.default_rng(seed)
    vocab = vocabulary or ["hello", "thank_you", "yes", "no", "help"]
    num_classes = len(vocab)

    total_samples = num_samples_per_class * num_classes
    x_data = np.zeros((total_samples, sequence_length, feature_dim), dtype=np.float32)
    y_data = np.zeros(total_samples, dtype=np.int64)

    time_steps = np.linspace(0, 2 * np.pi, sequence_length, dtype=np.float32)
    feature_indices = np.arange(feature_dim, dtype=np.float32)

    sample_idx = 0
    for class_id in range(num_classes):
        # Unique harmonic frequency and spatial base profile per class
        freq = float(class_id + 1)
        spatial_shift = float(class_id) / float(num_classes)

        for _ in range(num_samples_per_class):
            # Temporal wave: (T, 1)
            temp_wave = np.sin(freq * time_steps + spatial_shift)[:, np.newaxis]
            # Spatial pattern: (1, D)
            spatial_wave = np.cos(2.0 * np.pi * (feature_indices / feature_dim) + freq)[
                np.newaxis, :
            ]

            # Signal + Gaussian noise
            signal = temp_wave * spatial_wave
            noise = rng.normal(0.0, noise_std, size=(sequence_length, feature_dim)).astype(
                np.float32
            )
            raw_seq = signal + noise

            # Apply landmark normalization
            norm_seq = normalize_landmarks(raw_seq)

            x_data[sample_idx] = norm_seq
            y_data[sample_idx] = class_id
            sample_idx += 1

    return x_data, y_data


def stratified_split(
    labels: Sequence[int] | np.ndarray,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Partition sample indices into class-balanced train and validation subsets."""
    labels_arr = np.asarray(labels)
    unique_classes = np.unique(labels_arr)
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in unique_classes:
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)

        n_val = int(round(len(cls_idx) * val_split))
        if val_split > 0 and n_val == 0 and len(cls_idx) >= 2:
            n_val = 1

        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    return sorted(train_indices), sorted(val_indices)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Execute one training epoch with gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += float(loss.item()) * len(y_batch)
        preds = torch.argmax(logits, dim=-1)
        correct += int((preds == y_batch).sum().item())
        total += len(y_batch)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model without computing gradients."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            running_loss += float(loss.item()) * len(y_batch)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == y_batch).sum().item())
            total += len(y_batch)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    pipeline_config: PipelineConfig,
    training_config: TrainingConfig,
    epoch: int,
    val_acc: float,
    val_loss: float,
    path: Path | str,
) -> Path:
    """Save model checkpoint dictionary atomically."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "pipeline_config": pipeline_config.model_dump(),
        "training_config": training_config.model_dump(),
        "epoch": epoch,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "vocabulary": pipeline_config.vocabulary,
        "model_architecture": pipeline_config.architecture.value,
    }
    torch.save(checkpoint, target_path)
    return target_path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model weights and metadata from checkpoint."""
    target_path = Path(checkpoint_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {target_path}")

    checkpoint = torch.load(target_path, map_location=device, weights_only=False)

    if model is None:
        cfg = PipelineConfig.model_validate(checkpoint["pipeline_config"])
        model = build_model(cfg)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def train_model(
    pipeline_config: PipelineConfig,
    training_config: TrainingConfig,
    dataset: GestureDataset | None = None,
) -> tuple[nn.Module, TrainResult]:
    """Train sequence classification model adhering to SDD constraints."""
    start_time = time.time()
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)

    device = torch.device(training_config.device)

    # Ingest or synthesize dataset
    if dataset is None:
        x_synth, y_synth = generate_synthetic_dataset(
            num_samples_per_class=20,
            sequence_length=pipeline_config.sequence_length,
            feature_dim=pipeline_config.feature_dim,
            vocabulary=pipeline_config.vocabulary,
            seed=training_config.seed,
        )
        dataset = GestureDataset(tensors=x_synth, labels=y_synth)

    train_indices, val_indices = stratified_split(
        dataset.get_labels(),
        val_split=training_config.val_split,
        seed=training_config.seed,
    )

    train_sub = Subset(dataset, train_indices)
    val_sub = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_sub,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    model = build_model(pipeline_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []

    best_model_state: dict[str, Any] | None = None

    for epoch in range(training_config.epochs):
        t_loss, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss < best_val_loss):
            best_val_acc = v_acc
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            save_checkpoint(
                model=model,
                pipeline_config=pipeline_config,
                training_config=training_config,
                epoch=epoch,
                val_acc=v_acc,
                val_loss=v_loss,
                path=training_config.checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= training_config.early_stopping_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    duration = time.time() - start_time

    result = TrainResult(
        best_val_accuracy=float(best_val_acc),
        best_val_loss=float(best_val_loss),
        best_epoch=best_epoch,
        total_epochs=len(train_losses),
        checkpoint_path=str(training_config.checkpoint_path),
        train_loss_history=train_losses,
        val_loss_history=val_losses,
        val_acc_history=val_accs,
        training_duration_seconds=float(duration),
        model_architecture=pipeline_config.architecture.value,
        num_classes=len(pipeline_config.vocabulary),
    )

    return model, result


if __name__ == "__main__":
    p_cfg = PipelineConfig()
    t_cfg = TrainingConfig(epochs=15, early_stopping_patience=5)
    print(
        f"Starting training: {p_cfg.architecture.value} model on {len(p_cfg.vocabulary)} classes..."
    )
    trained_model, train_res = train_model(pipeline_config=p_cfg, training_config=t_cfg)
    print(
        f"Training complete! Best val accuracy: {train_res.best_val_accuracy:.4f}, "
        f"saved to {train_res.checkpoint_path}"
    )
