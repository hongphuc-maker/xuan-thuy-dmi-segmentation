from __future__ import annotations

from typing import Any, Iterable

import torch


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config: dict[str, Any]):
    optimizer_type = str(config["type"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    if optimizer_type == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
