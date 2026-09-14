from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DMIDiagnostics:
    n_samples: int
    classes_present: int
    sign: float
    loss: float
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    rank: int
    finite_loss: bool


def dmi_from_probabilities(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    matrix_jitter: float = 0.0,
    rank_rtol: float = 1e-12,
) -> tuple[torch.Tensor, DMIDiagnostics]:
    if probabilities.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected probabilities [B,C,H,W] and target [B,H,W]")
    if probabilities.shape[0] != target.shape[0] or probabilities.shape[2:] != target.shape[1:]:
        raise ValueError("Probability/target shape mismatch")
    if torch.any(probabilities < 0):
        raise ValueError("DMI probabilities cannot be negative")
    sums = probabilities.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-5):
        raise ValueError("DMI probabilities must sum to one across classes")

    num_classes = probabilities.shape[1]
    labels = target.reshape(-1)
    flat_probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, num_classes)
    return dmi_from_flat_probabilities(
        flat_probabilities,
        labels,
        ignore_index=ignore_index,
        matrix_jitter=matrix_jitter,
        rank_rtol=rank_rtol,
    )


def dmi_from_flat_probabilities(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    matrix_jitter: float = 0.0,
    rank_rtol: float = 1e-12,
) -> tuple[torch.Tensor, DMIDiagnostics]:
    """Compute one DMI matrix from flat samples ``[N,C]``.

    This form is also used by spatial validation after overlapping patch
    probabilities have been averaged into one unique-pixel validation mosaic.
    """
    if probabilities.ndim != 2 or target.ndim != 1:
        raise ValueError("Expected probabilities [N,C] and target [N]")
    if probabilities.shape[0] != target.shape[0]:
        raise ValueError("Probability/target sample count mismatch")
    if torch.any(probabilities < 0):
        raise ValueError("DMI probabilities cannot be negative")
    sums = probabilities.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-5):
        raise ValueError("DMI probabilities must sum to one across classes")

    num_classes = probabilities.shape[1]
    labels = target
    flat_probabilities = probabilities
    valid = labels != ignore_index
    labels = labels[valid].long()
    flat_probabilities = flat_probabilities[valid].to(torch.float64)
    if labels.numel() == 0:
        raise ValueError("DMI received zero valid pixels")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("Target class index is outside [0, num_classes)")

    one_hot = F.one_hot(labels, num_classes=num_classes).to(torch.float64)
    joint = (one_hot.T @ flat_probabilities) / labels.numel()
    if matrix_jitter:
        joint = joint + float(matrix_jitter) * torch.eye(
            num_classes,
            dtype=joint.dtype,
            device=joint.device,
        )
    sign, logabsdet = torch.linalg.slogdet(joint)
    loss = -logabsdet

    with torch.no_grad():
        singular_values = torch.linalg.svdvals(joint.detach())
        largest = singular_values.max()
        smallest = singular_values.min()
        rank = int((singular_values > largest * rank_rtol).sum().cpu())
        condition = largest / smallest if smallest > 0 else torch.tensor(float("inf"))
        diagnostics = DMIDiagnostics(
            n_samples=int(labels.numel()),
            classes_present=int(torch.unique(labels).numel()),
            sign=float(sign.cpu()),
            loss=float(loss.detach().cpu()),
            min_singular_value=float(smallest.cpu()),
            max_singular_value=float(largest.cpu()),
            condition_number=float(condition.cpu()),
            rank=rank,
            finite_loss=bool(torch.isfinite(loss).cpu()),
        )
    return loss, diagnostics


def exact_dmi_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    matrix_jitter: float = 0.0,
) -> tuple[torch.Tensor, DMIDiagnostics]:
    probabilities = torch.softmax(logits, dim=1)
    return dmi_from_probabilities(
        probabilities,
        target,
        ignore_index=ignore_index,
        matrix_jitter=matrix_jitter,
    )


class DMILoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 255,
        matrix_jitter: float = 0.0,
        return_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.matrix_jitter = float(matrix_jitter)
        self.return_diagnostics = bool(return_diagnostics)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        loss, diagnostics = exact_dmi_loss(
            logits,
            target,
            ignore_index=self.ignore_index,
            matrix_jitter=self.matrix_jitter,
        )
        return (loss, diagnostics) if self.return_diagnostics else loss


class DMIProbabilityLoss(nn.Module):
    """DMI for a model whose declared output is already probabilities."""

    def __init__(
        self,
        ignore_index: int = 255,
        matrix_jitter: float = 0.0,
        return_diagnostics: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.matrix_jitter = float(matrix_jitter)
        self.return_diagnostics = bool(return_diagnostics)

    def forward(self, probabilities: torch.Tensor, target: torch.Tensor):
        loss, diagnostics = dmi_from_probabilities(
            probabilities,
            target,
            ignore_index=self.ignore_index,
            matrix_jitter=self.matrix_jitter,
        )
        return (loss, diagnostics) if self.return_diagnostics else loss


def build_probability_dmi_loss(**parameters: object) -> DMIProbabilityLoss:
    """Factory usable directly from a loss YAML via ``module:callable``."""
    return DMIProbabilityLoss(**parameters)
