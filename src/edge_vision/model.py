"""Compact MobileNet-style CNN for CIFAR-10 classification.

Architecture:
  Input image 3x32x32
  -> standard convolution (stem)
  -> depthwise-separable blocks (spatial + channel mixing, fewer operations)
  -> global average pooling
  -> classification layer (10 class scores)

A depthwise-separable convolution splits a convolution into a per-channel
spatial filter (depthwise) and a 1x1 channel-mixing filter (pointwise).
"""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial convolution followed by pointwise channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class SmallMobileNet(nn.Module):
    """Small MobileNet-style classifier with fixed input shape 3x32x32."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv(16, 32, stride=2)  # 16x16
        self.block2 = DepthwiseSeparableConv(32, 64, stride=2)   # 8x8
        self.block3 = DepthwiseSeparableConv(64, 128, stride=2)  # 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(num_classes: int = 10) -> nn.Module:
    return SmallMobileNet(num_classes=num_classes)
