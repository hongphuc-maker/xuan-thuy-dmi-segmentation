from __future__ import annotations

import importlib
from typing import Any

from .unet_bn import UNetBN


def build_model(config: dict[str, Any]):
    architecture = str(config["architecture"])
    parameters = dict(config["parameters"])
    if architecture == "unet_bn":
        return UNetBN(**parameters)
    factory_path = config.get("factory")
    if factory_path:
        module_name, separator, attribute = str(factory_path).partition(":")
        if not separator:
            raise ValueError("model.factory must use 'module:callable'")
        factory = getattr(importlib.import_module(module_name), attribute)
        return factory(**parameters)
    raise ValueError(f"Unknown architecture: {architecture}")
