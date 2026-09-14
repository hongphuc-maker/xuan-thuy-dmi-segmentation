from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_counts(
    counts: Sequence[int] | torch.Tensor,
    method: str = "sqrt_inverse_frequency",
    minimum: float = 0.5,
    maximum: float = 5.0,
    beta: float = 0.9999,
) -> torch.Tensor:
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or torch.any(values <= 0):
        raise ValueError("Class counts must be a positive one-dimensional sequence")
    if method == "sqrt_inverse_frequency":
        weights = torch.sqrt(torch.median(values) / values)
    elif method == "effective_number":
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, values))
    elif method == "explicit":
        weights = values.clone()
    else:
        raise ValueError(f"Unknown class weighting method: {method}")
    weights = torch.clamp(weights, min=float(minimum), max=float(maximum))
    return (weights / weights.mean()).to(torch.float32)


def deterministic_masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE without CUDA nll_loss2d; supports optional per-class weights."""
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected logits [B,C,H,W] and target [B,H,W]")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("Logit/target shape mismatch")
    valid = target != ignore_index
    if not bool(valid.any()):
        raise ValueError("CE received zero valid pixels")
    valid_targets = target[valid]
    if int(valid_targets.min()) < 0 or int(valid_targets.max()) >= logits.shape[1]:
        raise ValueError("Target class index is outside [0, num_classes)")

    safe_target = torch.where(valid, target, torch.zeros_like(target)).long()
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    per_pixel = -(
        one_hot.to(logits.dtype) * F.log_softmax(logits, dim=1)
    ).sum(dim=1)
    pixel_weights = valid.to(logits.dtype)
    if class_weights is not None:
        if class_weights.numel() != logits.shape[1]:
            raise ValueError("class_weights length must equal num_classes")
        pixel_weights = pixel_weights * class_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )[safe_target]
    return (per_pixel * pixel_weights).sum() / pixel_weights.sum()


class MaskedCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().clone().float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return deterministic_masked_cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            class_weights=self.class_weights,
        )
