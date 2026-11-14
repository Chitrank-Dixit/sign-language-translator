"""Unit tests for spatial landmark normalization and invariance guarantees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sign_translator.normalize import normalize_landmarks, normalize_single_hand


def _generate_synthetic_hand(seed: int = 42) -> np.ndarray:
    """Generate a realistic synthetic 21-landmark hand centered at arbitrary offset."""
    rng = np.random.default_rng(seed)
    # 21 points with wrist at (10.0, 20.0, 5.0) and spread around it
    wrist = np.array([10.0, 20.0, 5.0], dtype=np.float32)
    offsets = rng.uniform(-5.0, 5.0, size=(21, 3)).astype(np.float32)
    offsets[0] = [0.0, 0.0, 0.0]  # wrist offset is 0
    return wrist + offsets


def _generate_two_hands(seed: int = 42) -> np.ndarray:
    """Generate synthetic 2-hand array of shape (2, 21, 3)."""
    h0 = _generate_synthetic_hand(seed)
    h1 = _generate_synthetic_hand(seed + 1)
    return np.stack([h0, h1], axis=0)


@pytest.mark.unit
class TestLandmarkNormalization:
    """Mathematical and robustness tests for normalize_landmarks and normalize_single_hand."""

    def test_scale_invariance_2_5x(self) -> None:
        """TC-01: Scaling coordinates by 2.5x produces identical normalized tensor."""
        hand = _generate_synthetic_hand(100)
        norm_orig = normalize_single_hand(hand)
        norm_scaled = normalize_single_hand(hand * 2.5)

        assert np.allclose(norm_orig, norm_scaled, atol=1e-6)

    def test_scale_invariance_extreme_factors(self) -> None:
        """TC-02: Extreme scale factors (0.1x and 10x) preserve normalized landmarks."""
        two_hands = _generate_two_hands(101)
        norm_base = normalize_landmarks(two_hands)
        norm_small = normalize_landmarks(two_hands * 0.1)
        norm_large = normalize_landmarks(two_hands * 10.0)

        assert np.allclose(norm_base, norm_small, atol=1e-6)
        assert np.allclose(norm_base, norm_large, atol=1e-6)

    def test_translation_invariance_100(self) -> None:
        """TC-03: Shifting (x, y, z) by +100 produces identical normalized tensor."""
        two_hands = _generate_two_hands(102)
        norm_orig = normalize_landmarks(two_hands)
        shifted = two_hands + 100.0
        norm_shifted = normalize_landmarks(shifted)

        assert np.allclose(norm_orig, norm_shifted, atol=1e-6)

    def test_joint_scale_and_translation(self) -> None:
        """TC-04: Simultaneous scaling (2.5x) and shifting preserves normalized tensor."""
        hand = _generate_synthetic_hand(103)
        norm_orig = normalize_single_hand(hand)

        offset = np.array([50.0, -30.0, 120.0], dtype=np.float32)
        transformed = (hand * 2.5) + offset
        norm_trans = normalize_single_hand(transformed)

        assert np.allclose(norm_orig, norm_trans, atol=1e-6)

    def test_wrist_origin_anchor(self) -> None:
        """TC-05: Wrist landmark (index 0) is strictly normalized to (0, 0, 0)."""
        two_hands = _generate_two_hands(104)
        norm = normalize_landmarks(two_hands)

        # Hand 0 wrist
        assert np.allclose(norm[0, 0, :], [0.0, 0.0, 0.0], atol=1e-7)
        # Hand 1 wrist
        assert np.allclose(norm[1, 0, :], [0.0, 0.0, 0.0], atol=1e-7)

    def test_coordinate_bounds(self) -> None:
        """TC-06: All coordinates are in [-1, 1] and max Euclidean distance is 1.0."""
        hand = _generate_synthetic_hand(105)
        norm = normalize_single_hand(hand)

        assert np.all(norm >= -1.0 - 1e-6)
        assert np.all(norm <= 1.0 + 1e-6)

        dists = np.linalg.norm(norm, axis=-1)
        assert np.isclose(np.max(dists), 1.0, atol=1e-6)

    def test_single_hand_0_present_hand_1_absent(self) -> None:
        """TC-07: Hand 0 normalized while absent Hand 1 remains exact zeros."""
        h0 = _generate_synthetic_hand(106)
        h1 = np.zeros((21, 3), dtype=np.float32)
        data = np.stack([h0, h1], axis=0)  # (2, 21, 3)

        norm = normalize_landmarks(data)

        # Hand 0 is normalized
        assert np.isclose(np.max(np.linalg.norm(norm[0], axis=-1)), 1.0, atol=1e-6)
        # Hand 1 remains exact zeros
        assert np.all(norm[1] == 0.0)

    def test_single_hand_1_present_hand_0_absent(self) -> None:
        """TC-08: Hand 1 normalized while absent Hand 0 remains exact zeros."""
        h0 = np.zeros((21, 3), dtype=np.float32)
        h1 = _generate_synthetic_hand(107)
        data = np.stack([h0, h1], axis=0)

        norm = normalize_landmarks(data)

        assert np.all(norm[0] == 0.0)
        assert np.isclose(np.max(np.linalg.norm(norm[1], axis=-1)), 1.0, atol=1e-6)

    def test_both_hands_absent_all_zeros(self) -> None:
        """TC-09: All-zero tensor returns exact zeros without ZeroDivisionError or NaN."""
        zeros_126 = np.zeros(126, dtype=np.float32)
        norm = normalize_landmarks(zeros_126)

        assert np.all(norm == 0.0)
        assert not np.isnan(norm).any()
        assert not np.isinf(norm).any()

    def test_collapsed_degenerate_hand(self) -> None:
        """TC-10: Hand with all 21 points identical produces zeros safely."""
        degenerate = np.ones((21, 3), dtype=np.float32) * 42.0
        norm = normalize_single_hand(degenerate)

        assert np.all(norm == 0.0)
        assert not np.isnan(norm).any()

    def test_nan_and_inf_immunity(self) -> None:
        """TC-11: Sanitizes NaN and Inf inputs without raising unhandled exceptions."""
        corrupted = _generate_synthetic_hand(108)
        corrupted[5, 0] = np.nan
        corrupted[10, 1] = np.inf
        corrupted[15, 2] = -np.inf

        norm = normalize_single_hand(corrupted)

        assert not np.isnan(norm).any()
        assert not np.isinf(norm).any()
        assert np.all(norm >= -1.0 - 1e-6)
        assert np.all(norm <= 1.0 + 1e-6)

    def test_shape_support(self) -> None:
        """TC-12: Normalizes correctly across all supported tensor ranks and shapes."""
        two_hands = _generate_two_hands(109)

        # (126,)
        flat = two_hands.reshape(126)
        norm_flat = normalize_landmarks(flat)
        assert norm_flat.shape == (126,)

        # (2, 21, 3)
        norm_2 = normalize_landmarks(two_hands)
        assert norm_2.shape == (2, 21, 3)

        # (21, 3)
        norm_single = normalize_landmarks(two_hands[0])
        assert norm_single.shape == (21, 3)

        # (30, 126)
        seq_flat = np.repeat(flat[np.newaxis, :], 30, axis=0)
        norm_seq = normalize_landmarks(seq_flat)
        assert norm_seq.shape == (30, 126)

        # (30, 2, 21, 3)
        seq_struct = np.repeat(two_hands[np.newaxis, ...], 30, axis=0)
        norm_seq_struct = normalize_landmarks(seq_struct)
        assert norm_seq_struct.shape == (30, 2, 21, 3)

        # (4, 30, 126)
        batch = np.repeat(seq_flat[np.newaxis, ...], 4, axis=0)
        norm_batch = normalize_landmarks(batch)
        assert norm_batch.shape == (4, 30, 126)

    def test_unsupported_shape_raises(self) -> None:
        """TC-13: Raises ValueError when given unsupported tensor dimensions."""
        with pytest.raises(ValueError, match="Unsupported landmarks shape"):
            normalize_landmarks(np.zeros(100))

        with pytest.raises(ValueError, match="Expected single hand array of shape"):
            normalize_single_hand(np.zeros((20, 3)))

    def test_unsupported_type_raises(self) -> None:
        """TC-14: Raises TypeError when given unsupported input type."""
        with pytest.raises(TypeError, match="Unsupported landmarks type"):
            normalize_landmarks([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_idempotency(self) -> None:
        """TC-15: Repeated normalization is idempotent: norm(norm(x)) == norm(x)."""
        hand = _generate_synthetic_hand(110)
        norm1 = normalize_single_hand(hand)
        norm2 = normalize_single_hand(norm1)

        assert np.allclose(norm1, norm2, atol=1e-6)

    def test_pytorch_tensor_parity(self) -> None:
        """TC-16: PyTorch Tensor normalization produces identical results to NumPy."""
        two_hands = _generate_two_hands(111)
        np_norm = normalize_landmarks(two_hands)

        torch_input = torch.from_numpy(two_hands)
        torch_norm = normalize_landmarks(torch_input)

        assert isinstance(torch_norm, torch.Tensor)
        assert torch_norm.shape == torch_input.shape
        assert np.allclose(np_norm, torch_norm.numpy(), atol=1e-6)
