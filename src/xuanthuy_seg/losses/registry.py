from __future__ import annotations

import importlib
from typing import Any

import torch

from .cross_entropy import MaskedCrossEntropy, class_weights_from_counts
from .dmi import DMILoss


def build_loss(config: dict[str, Any]):
    loss_type = str(config["type"])
    parameters = dict(config.get("parameters", {}))
    ignore_index = int(parameters.get("ignore_index", 255))
    if loss_type == "ce_unweighted":
        return MaskedCrossEntropy(ignore_index=ignore_index)
    if loss_type == "ce_weighted":
        weighting = dict(parameters["weighting"])
        method = str(weighting.pop("method"))
        counts = weighting.pop("class_counts")
        weighting.pop("source", None)
        if method == "explicit":
            weights = torch.as_tensor(counts, dtype=torch.float32)
            weights = weights / weights.mean()
        else:
            weights = class_weights_from_counts(counts, method=method, **weighting)
        return MaskedCrossEntropy(ignore_index=ignore_index, class_weights=weights)
    if loss_type in {"dmi_exact", "dmi_regularized"}:
        jitter = float(parameters.get("matrix_jitter", 0.0))
        if loss_type == "dmi_exact" and jitter != 0.0:
            raise ValueError("dmi_exact requires matrix_jitter=0")
        return DMILoss(ignore_index=ignore_index, matrix_jitter=jitter)
    factory_path = config.get("factory")
    if factory_path:
        module_name, separator, attribute = str(factory_path).partition(":")
        if not separator:
            raise ValueError("loss.factory must use 'module:callable'")
        factory = getattr(importlib.import_module(module_name), attribute)
        return factory(**parameters)
    raise ValueError(f"Unknown loss type: {loss_type}")
