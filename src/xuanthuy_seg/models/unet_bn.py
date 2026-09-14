from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        diff_y = skip.size(2) - inputs.size(2)
        diff_x = skip.size(3) - inputs.size(3)
        if diff_x or diff_y:
            inputs = F.pad(
                inputs,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, inputs], dim=1))


class UNetBN(nn.Module):
    """The single canonical U-Net used by the existing Notebook 4 experiments."""

    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 11,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, 2 * base)
        self.down2 = Down(2 * base, 4 * base)
        self.down3 = Down(4 * base, 8 * base)
        self.down4 = Down(8 * base, 16 * base)
        self.up1 = Up(16 * base, 8 * base, 8 * base)
        self.up2 = Up(8 * base, 4 * base, 4 * base)
        self.up3 = Up(4 * base, 2 * base, 2 * base)
        self.up4 = Up(2 * base, base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))
