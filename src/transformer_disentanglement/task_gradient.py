from __future__ import annotations

import torch
import torch.nn.functional as F


def marginal_orthogonal_task_gradient(
    background_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Project the negative categorical CE gradient out of the marginal space."""
    probabilities = background_logits.softmax(dim=-1).detach()
    one_hot = F.one_hot(targets, background_logits.shape[-1]).to(
        background_logits.dtype
    )
    residual = one_hot - probabilities
    marginal_mean = torch.sum(probabilities * residual, dim=-1, keepdim=True)
    return residual - marginal_mean


def normalized_task_gradient_target(
    background_logits: torch.Tensor,
    targets: torch.Tensor,
    target_rms: float = 0.1,
) -> torch.Tensor:
    residual = marginal_orthogonal_task_gradient(background_logits, targets)
    normalized = residual / residual.square().mean(
        dim=-1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    return normalized * target_rms
