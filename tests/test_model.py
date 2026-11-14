"""Unit tests for sequence classification models, training loops, and checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from sign_translator.architecture import (
    LSTMClassifier,
    TransformerClassifier,
    build_model,
)
from sign_translator.schemas import (
    ModelArchitecture,
    PipelineConfig,
    TrainingConfig,
    TrainResult,
)
from sign_translator.train import (
    GestureDataset,
    generate_synthetic_dataset,
    load_checkpoint,
    save_checkpoint,
    stratified_split,
    train_model,
)


@pytest.mark.unit
class TestModelArchitectures:
    """Forward-pass and shape verification for neural sequence models."""

    def test_lstm_forward_pass_unidirectional(self) -> None:
        """TC-MOD-01: Verify unidirectional LSTM maps (4, 30, 126) to (4, 5)."""
        model = LSTMClassifier(
            input_dim=126,
            hidden_dim=64,
            num_layers=2,
            num_classes=5,
            bidirectional=False,
        )
        x = torch.randn(4, 30, 126)
        out = model(x)

        assert out.shape == (4, 5)
        assert torch.isfinite(out).all()

    def test_lstm_forward_pass_bidirectional(self) -> None:
        """TC-MOD-02: Verify bidirectional LSTM maps (4, 30, 126) to (4, 5)."""
        model = LSTMClassifier(
            input_dim=126,
            hidden_dim=64,
            num_layers=2,
            num_classes=5,
            bidirectional=True,
        )
        x = torch.randn(4, 30, 126)
        out = model(x)

        assert out.shape == (4, 5)
        assert torch.isfinite(out).all()

    def test_transformer_forward_pass(self) -> None:
        """TC-MOD-03: Verify TransformerClassifier maps (4, 30, 126) to (4, 5)."""
        model = TransformerClassifier(
            input_dim=126,
            hidden_dim=64,
            num_layers=2,
            num_classes=5,
            nhead=4,
        )
        x = torch.randn(4, 30, 126)
        out = model(x)

        assert out.shape == (4, 5)
        assert torch.isfinite(out).all()

    def test_variable_batch_sizes(self) -> None:
        """TC-MOD-04: Test forward pass across non-standard batch sizes."""
        lstm = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=5)
        transformer = TransformerClassifier(input_dim=126, hidden_dim=32, nhead=2, num_classes=5)

        for b in (1, 3, 7, 16):
            x = torch.randn(b, 30, 126)
            out_lstm = lstm(x)
            out_tf = transformer(x)

            assert out_lstm.shape == (b, 5)
            assert out_tf.shape == (b, 5)

    def test_degenerate_all_zero_input(self) -> None:
        """TC-MOD-05: Missing hand / all-zero tensors compute without NaN or Inf."""
        lstm = LSTMClassifier(input_dim=126, hidden_dim=32, num_classes=5)
        transformer = TransformerClassifier(input_dim=126, hidden_dim=32, nhead=2, num_classes=5)

        zeros = torch.zeros(2, 30, 126)
        out_lstm = lstm(zeros)
        out_tf = transformer(zeros)

        assert torch.isfinite(out_lstm).all()
        assert torch.isfinite(out_tf).all()

    def test_build_model_factory(self) -> None:
        """Verify model factory instantiates requested architecture."""
        cfg_lstm = PipelineConfig(architecture=ModelArchitecture.LSTM)
        model_lstm = build_model(cfg_lstm)
        assert isinstance(model_lstm, LSTMClassifier)

        cfg_tf = PipelineConfig(
            architecture=ModelArchitecture.TRANSFORMER,
            hidden_dim=128,
            transformer_heads=4,
        )
        model_tf = build_model(cfg_tf)
        assert isinstance(model_tf, TransformerClassifier)

    def test_transformer_head_divisibility_validation(self) -> None:
        """TC-MOD-07: Validation error when hidden_dim not divisible by transformer_heads."""
        with pytest.raises(ValidationError, match="must be divisible by transformer_heads"):
            PipelineConfig(
                architecture=ModelArchitecture.TRANSFORMER,
                hidden_dim=128,
                transformer_heads=5,
            )


@pytest.mark.unit
class TestDatasetAndSplitting:
    """Tests for dataset loading, synthetic synthesis, and stratified splitting."""

    def test_in_memory_dataset(self) -> None:
        """TC-MOD-08: GestureDataset initialized from in-memory arrays."""
        x = np.random.randn(10, 30, 126).astype(np.float32)
        y = np.arange(10) % 5

        ds = GestureDataset(tensors=x, labels=y)
        assert len(ds) == 10

        item_x, item_y = ds[0]
        assert item_x.shape == (30, 126)
        assert item_x.dtype == torch.float32
        assert item_y.dtype == torch.long

    def test_directory_dataset(self, tmp_path: Path) -> None:
        """TC-MOD-09: GestureDataset loading from nested directory tree."""
        classes = ["hello", "thank_you"]
        for idx, cls in enumerate(classes):
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(3):
                arr = np.ones((30, 126), dtype=np.float32) * (idx + 1)
                np.save(cls_dir / f"sample_{i}.npy", arr)

        ds = GestureDataset(data_dir=tmp_path, vocabulary=classes)
        assert len(ds) == 6
        assert len(np.unique(ds.get_labels())) == 2

    def test_stratified_split_balance(self) -> None:
        """TC-MOD-10: Stratified split maintains equal class distribution."""
        labels = [0] * 10 + [1] * 10 + [2] * 10
        train_idx, val_idx = stratified_split(labels, val_split=0.2, seed=42)

        # 2 per class in validation -> 6 total val, 24 total train
        assert len(val_idx) == 6
        assert len(train_idx) == 24
        assert len(set(train_idx).intersection(set(val_idx))) == 0

        # Check each class has 2 in val
        val_labels = [labels[i] for i in val_idx]
        assert val_labels.count(0) == 2
        assert val_labels.count(1) == 2
        assert val_labels.count(2) == 2

    def test_synthetic_generator_determinism(self) -> None:
        """TC-MOD-11: Synthetic generator produces deterministic, normalized tensors."""
        x1, y1 = generate_synthetic_dataset(num_samples_per_class=5, seed=123)
        x2, y2 = generate_synthetic_dataset(num_samples_per_class=5, seed=123)

        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert x1.shape == (25, 30, 126)


@pytest.mark.unit
class TestTrainingPipeline:
    """Overfit convergence and checkpoint persistence tests."""

    def test_overfit_sanity_lstm(self, tmp_path: Path) -> None:
        """TC-MOD-12: Overfit sanity test achieves >95% accuracy on synthetic data (LSTM)."""
        pipe_cfg = PipelineConfig(
            architecture=ModelArchitecture.LSTM,
            sequence_length=30,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            vocabulary=["hello", "thank_you", "yes", "no", "help"],
        )
        ckpt_file = tmp_path / "lstm_best.pt"
        train_cfg = TrainingConfig(
            epochs=20,
            batch_size=16,
            learning_rate=0.01,
            val_split=0.2,
            seed=42,
            checkpoint_path=str(ckpt_file),
            early_stopping_patience=10,
        )

        _, result = train_model(pipeline_config=pipe_cfg, training_config=train_cfg)

        assert isinstance(result, TrainResult)
        assert result.best_val_accuracy >= 0.95, (
            f"LSTM overfit sanity failed: {result.best_val_accuracy}"
        )
        assert ckpt_file.exists()

    def test_overfit_sanity_transformer(self, tmp_path: Path) -> None:
        """TC-MOD-13: Overfit sanity test achieves >95% accuracy on synthetic data (Transformer)."""
        pipe_cfg = PipelineConfig(
            architecture=ModelArchitecture.TRANSFORMER,
            sequence_length=30,
            hidden_dim=64,
            num_layers=2,
            transformer_heads=4,
            dropout=0.1,
            vocabulary=["hello", "thank_you", "yes", "no", "help"],
        )
        ckpt_file = tmp_path / "transformer_best.pt"
        train_cfg = TrainingConfig(
            epochs=20,
            batch_size=16,
            learning_rate=0.005,
            val_split=0.2,
            seed=42,
            checkpoint_path=str(ckpt_file),
            early_stopping_patience=10,
        )

        _, result = train_model(pipeline_config=pipe_cfg, training_config=train_cfg)

        assert isinstance(result, TrainResult)
        assert result.best_val_accuracy >= 0.95, (
            f"Transformer overfit sanity failed: {result.best_val_accuracy}"
        )
        assert ckpt_file.exists()

    def test_checkpoint_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """TC-MOD-14: Verify checkpoint state dict roundtrip and forward output parity."""
        pipe_cfg = PipelineConfig(
            architecture=ModelArchitecture.LSTM,
            hidden_dim=32,
            num_layers=1,
            vocabulary=["a", "b", "c"],
        )
        train_cfg = TrainingConfig(epochs=1)
        model = build_model(pipe_cfg)

        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model=model,
            pipeline_config=pipe_cfg,
            training_config=train_cfg,
            epoch=0,
            val_acc=0.88,
            val_loss=0.25,
            path=ckpt_path,
        )
        assert ckpt_path.exists()

        restored_model, meta = load_checkpoint(ckpt_path)
        assert meta["val_accuracy"] == 0.88
        assert meta["vocabulary"] == ["a", "b", "c"]

        x = torch.randn(2, 30, 126)
        model.eval()
        with torch.no_grad():
            orig_out = model(x)
            rest_out = restored_model(x)

        assert torch.allclose(orig_out, rest_out, atol=1e-5)

    def test_load_checkpoint_missing_file(self, tmp_path: Path) -> None:
        """TC-MOD-15: Loading a nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_dataset_initialization_errors(self) -> None:
        """TC-MOD-16: Dataset constructor validation for missing arguments and length mismatches."""
        with pytest.raises(ValueError, match="Must provide either"):
            GestureDataset()

        with pytest.raises(ValueError, match="length mismatch"):
            GestureDataset(
                tensors=np.zeros((5, 30, 126), dtype=np.float32),
                labels=np.array([0, 1, 2]),
            )

    def test_dataset_directory_errors(self, tmp_path: Path) -> None:
        """TC-MOD-17: Dataset directory validation for missing paths or empty directories."""
        with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
            GestureDataset(data_dir=tmp_path / "nonexistent_dir")

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .npy sequences found"):
            GestureDataset(data_dir=empty_dir)

    def test_dataset_normalize_on_load(self) -> None:
        """TC-MOD-18: Verify normalize_on_load normalizes raw landmark sequences."""
        raw_x = np.ones((2, 30, 126), dtype=np.float32) * 100.0
        y = np.array([0, 1])
        ds = GestureDataset(tensors=raw_x, labels=y, normalize_on_load=True)

        seq, _ = ds[0]
        # Because all coordinates in hand 0 were identical (100.0), after wrist subtraction they become 0.0
        assert torch.all(seq == 0.0)

    def test_stratified_split_zero_val(self) -> None:
        """TC-MOD-19: Stratified split with val_split=0.0 allocates all to train."""
        labels = [0, 0, 1, 1, 2, 2]
        train_idx, val_idx = stratified_split(labels, val_split=0.0, seed=42)
        assert len(val_idx) == 0
        assert len(train_idx) == 6
