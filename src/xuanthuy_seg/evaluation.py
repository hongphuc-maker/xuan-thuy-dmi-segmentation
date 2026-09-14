from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.cp2 import CP2Data
from .losses.dmi import dmi_from_flat_probabilities


def cross_entropy_from_flat_probabilities(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE on unique mosaic pixels, with the same weighted-mean convention as training."""
    if probabilities.ndim != 2 or target.ndim != 1:
        raise ValueError("Expected probabilities [N,C] and target [N]")
    if probabilities.shape[0] != target.shape[0]:
        raise ValueError("Probability/target sample count mismatch")
    if torch.any(probabilities < 0):
        raise ValueError("CE probabilities cannot be negative")
    sums = probabilities.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-5):
        raise ValueError("CE probabilities must sum to one across classes")

    valid = target != ignore_index
    labels = target[valid].long()
    values = probabilities[valid].to(torch.float64)
    if labels.numel() == 0:
        raise ValueError("CE received zero valid pixels")
    if int(labels.min()) < 0 or int(labels.max()) >= probabilities.shape[1]:
        raise ValueError("Target class index is outside [0, num_classes)")
    selected = values.gather(1, labels[:, None]).squeeze(1)
    # Mosaic probabilities originate as float32 model outputs. This floor only
    # handles exact underflow to zero; ordinary values are not clipped.
    floor = torch.finfo(torch.float32).tiny
    per_pixel = -torch.log(selected.clamp_min(floor))
    if class_weights is None:
        return per_pixel.mean()
    weights = class_weights.detach().to(device=values.device, dtype=torch.float64)
    if weights.ndim != 1 or weights.numel() != probabilities.shape[1]:
        raise ValueError("class_weights length must equal num_classes")
    pixel_weights = weights[labels]
    return (per_pixel * pixel_weights).sum() / pixel_weights.sum()


def metrics_from_confusion(
    confusion: np.ndarray,
    class_map: dict[int, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    confusion = confusion.astype(np.int64, copy=False)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    union = support + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )
    codes = list(sorted(class_map))
    summary: dict[str, Any] = {
        "validation_oa": float(true_positive.sum() / max(confusion.sum(), 1)),
        "validation_macro_f1_11": float(f1.mean()),
        "validation_macro_iou_11": float(iou.mean()),
        "validation_pixels": int(confusion.sum()),
        "predicted_classes": int((predicted > 0).sum()),
        "recall_by_code": {str(code): float(recall[index]) for index, code in enumerate(codes)},
    }
    per_class = pd.DataFrame(
        {
            "code": codes,
            "class": [class_map[code] for code in codes],
            "support_pixels": support.astype(np.int64),
            "predicted_pixels": predicted.astype(np.int64),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
    )
    return summary, per_class


@torch.inference_mode()
def evaluate_spatial_validation(
    model: torch.nn.Module,
    data: CP2Data,
    device: torch.device,
    batch_size: int = 16,
    output_type: str = "logits",
    include_dmi: bool = False,
    dmi_matrix_jitter: float = 0.0,
    dmi_rank_rtol: float = 1e-12,
    include_ce: bool = False,
    ce_class_weights: torch.Tensor | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Reproduce Notebook 4's overlap-averaged spatial validation mosaic."""
    was_training = model.training
    model.eval()
    loader = DataLoader(
        data.validation_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    num_classes = len(data.class_map)
    probability_sum = np.zeros((num_classes, *data.label.shape), dtype=np.float32)
    probability_count = np.zeros(data.label.shape, dtype=np.uint16)
    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        outputs = model(inputs)
        if output_type == "logits":
            outputs = torch.softmax(outputs, dim=1)
        elif output_type != "probabilities":
            raise ValueError(f"Unsupported model output type: {output_type}")
        probabilities = outputs.cpu().numpy()
        valid_batch = batch["image_valid"].numpy()
        for index in range(probabilities.shape[0]):
            row0, col0 = int(batch["row"][index]), int(batch["col"][index])
            height, width = int(batch["height"][index]), int(batch["width"][index])
            valid = valid_batch[index, :height, :width]
            window = np.s_[row0 : row0 + height, col0 : col0 + width]
            probability_sum[:, window[0], window[1]] += probabilities[index, :, :height, :width] * valid[None]
            probability_count[window] += valid.astype(np.uint16)
        del inputs, probabilities

    metric_mask = data.common_valid & (data.split_mask == 2)
    counts = probability_count[metric_mask]
    expected_pixels = int(data.contract["pixel_counts"]["validation_core"])
    if int(metric_mask.sum()) != expected_pixels or not np.all(counts > 0):
        raise ValueError("Validation windows do not fully cover the CP2 validation core")
    probabilities = probability_sum[:, metric_mask] / counts[None]
    prediction = probabilities.argmax(axis=0)
    truth = data.label[metric_mask].astype(np.int64) - 1
    confusion = np.bincount(
        truth * num_classes + prediction,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    summary, per_class = metrics_from_confusion(confusion, data.class_map)
    flat_probabilities = torch.from_numpy(np.ascontiguousarray(probabilities.T))
    flat_truth = torch.from_numpy(np.ascontiguousarray(truth))
    if include_ce:
        ce_loss = cross_entropy_from_flat_probabilities(
            flat_probabilities,
            flat_truth,
            ignore_index=int(data.ignore_index),
            class_weights=ce_class_weights,
        )
        summary.update(
            {
                "validation_ce_loss": float(ce_loss.cpu()),
                "validation_ce_weighted": ce_class_weights is not None,
            }
        )
    if include_dmi:
        dmi_loss, diagnostics = dmi_from_flat_probabilities(
            flat_probabilities,
            flat_truth,
            ignore_index=int(data.ignore_index),
            matrix_jitter=float(dmi_matrix_jitter),
            rank_rtol=float(dmi_rank_rtol),
        )
        summary.update(
            {
                "validation_dmi_loss": float(dmi_loss.cpu()),
                "validation_dmi_log_abs_det": -float(dmi_loss.cpu()),
                "validation_dmi_abs_det": float(torch.exp(-dmi_loss).cpu()),
                "validation_dmi_rank": diagnostics.rank,
                "validation_dmi_condition": diagnostics.condition_number,
                "validation_dmi_min_singular_value": diagnostics.min_singular_value,
                "validation_dmi_sign": diagnostics.sign,
                "validation_dmi_finite": diagnostics.finite_loss,
                "validation_dmi_classes_present": diagnostics.classes_present,
            }
        )
    if was_training:
        model.train()
    return summary, per_class, confusion
