from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdditiveClosing2d(nn.Module):
    """Depthwise differentiable grayscale closing on per-class logits.

    The learnable non-flat structuring element is additive, matching grayscale
    morphology: dilation uses ``x + b`` and erosion uses ``x - b``.  A smooth
    log-sum-exp approximation keeps both operations differentiable.
    """

    def __init__(
        self,
        num_classes: int = 11,
        kernel_size: int = 3,
        beta: float = 10.0,
        padding_mode: str = "replicate",
        se_initialization: str = "zeros",
    ) -> None:
        super().__init__()
        if int(kernel_size) != 3:
            raise ValueError("LogitAdditiveClosing2d currently requires kernel_size=3")
        if float(beta) <= 0:
            raise ValueError("beta must be positive")
        if padding_mode != "replicate":
            raise ValueError("LogitAdditiveClosing2d currently requires replicate padding")
        if se_initialization != "zeros":
            raise ValueError("LogitAdditiveClosing2d currently requires zero/flat SE initialization")

        self.num_classes = int(num_classes)
        self.kernel_size = int(kernel_size)
        self.padding_mode = str(padding_mode)
        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32))
        self.structuring_element = nn.Parameter(
            torch.zeros(self.num_classes, 1, self.kernel_size, self.kernel_size)
        )

    def _windows(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if inputs.ndim != 4 or inputs.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected [B,{self.num_classes},H,W], received {tuple(inputs.shape)}"
            )
        batch, channels, height, width = inputs.shape
        radius = self.kernel_size // 2
        # Avoid CUDA replicate-pad backward kernels, which can conflict with
        # torch.use_deterministic_algorithms(True).  For the locked 3x3 SE,
        # concatenating boundary slices is the same replicate padding.
        if radius != 1:
            raise AssertionError("The locked 3x3 morphology contract requires radius=1")
        padded_width = torch.cat((inputs[..., :1], inputs, inputs[..., -1:]), dim=3)
        padded = torch.cat(
            (padded_width[:, :, :1, :], padded_width, padded_width[:, :, -1:, :]),
            dim=2,
        )
        windows = F.unfold(padded, kernel_size=self.kernel_size)
        windows = windows.reshape(
            batch,
            channels,
            self.kernel_size * self.kernel_size,
            height * width,
        )
        return windows, (batch, channels, height, width)

    def smooth_dilation(self, inputs: torch.Tensor) -> torch.Tensor:
        windows, shape = self._windows(inputs)
        se = self.structuring_element.reshape(
            1, self.num_classes, self.kernel_size * self.kernel_size, 1
        ).to(inputs.dtype)
        beta = self.beta.to(inputs.dtype)
        return (torch.logsumexp(beta * (windows + se), dim=2) / beta).reshape(shape)

    def smooth_erosion(self, inputs: torch.Tensor) -> torch.Tensor:
        windows, shape = self._windows(inputs)
        se = self.structuring_element.reshape(
            1, self.num_classes, self.kernel_size * self.kernel_size, 1
        ).to(inputs.dtype)
        beta = self.beta.to(inputs.dtype)
        return (-torch.logsumexp(-beta * (windows - se), dim=2) / beta).reshape(shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        closed_logits = self.smooth_erosion(self.smooth_dilation(inputs))
        if not torch.isfinite(closed_logits).all():
            raise FloatingPointError("LogitAdditiveClosing2d produced non-finite logits")
        return closed_logits


class UNetWithLogitAdditiveClosing(nn.Module):
    """Canonical U-Net followed by one trainable closing-only logit layer."""

    def __init__(
        self,
        unet: nn.Module,
        num_classes: int = 11,
        kernel_size: int = 3,
        beta: float = 10.0,
        padding_mode: str = "replicate",
        se_initialization: str = "zeros",
    ) -> None:
        super().__init__()
        self.unet = unet
        self.logit_closing = LogitAdditiveClosing2d(
            num_classes=num_classes,
            kernel_size=kernel_size,
            beta=beta,
            padding_mode=padding_mode,
            se_initialization=se_initialization,
        )

    def forward(self, inputs: torch.Tensor, return_intermediates: bool = False):
        raw_logits = self.unet(inputs)
        closed_logits = self.logit_closing(raw_logits)
        if return_intermediates:
            return {"raw_logits": raw_logits, "closed_logits": closed_logits}
        return closed_logits


def build_unet_with_logit_additive_closing(
    in_channels: int = 10,
    num_classes: int = 11,
    base_channels: int = 64,
    kernel_size: int = 3,
    beta: float = 10.0,
    padding_mode: str = "replicate",
    se_initialization: str = "zeros",
) -> UNetWithLogitAdditiveClosing:
    """Factory referenced by the morphology model YAML."""
    from .unet_bn import UNetBN

    unet = UNetBN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base_channels),
    )
    return UNetWithLogitAdditiveClosing(
        unet,
        num_classes=int(num_classes),
        kernel_size=int(kernel_size),
        beta=float(beta),
        padding_mode=str(padding_mode),
        se_initialization=str(se_initialization),
    )
