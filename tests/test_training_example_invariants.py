import numpy as np
import torch

from experiments.train_conditional_response_demo import masked_target_sequence
from transformer_disentanglement.epistasis import (
    weighted_double_center,
    weighted_gauge_error,
)


def test_masked_target_sequence_does_not_mutate_numpy_input() -> None:
    sequence = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    original = sequence.copy()
    base, true_target = masked_target_sequence(
        sequence, target=2, device=torch.device("cpu")
    )
    assert np.array_equal(sequence, original)
    assert int(true_target) == 3
    assert int(base[2]) == 21
    assert torch.equal(base[[0, 1, 3, 4]], torch.tensor([1, 2, 4, 5]))


def test_teacher_reprojection_obeys_student_gauge() -> None:
    torch.manual_seed(71)
    blocks = torch.randn(6, 5, 5)
    left = torch.softmax(torch.randn(6, 5), dim=-1)
    right = torch.softmax(torch.randn(6, 5), dim=-1)
    projected = weighted_double_center(blocks, left, right)
    assert float(weighted_gauge_error(projected, left, right)) < 2e-6
