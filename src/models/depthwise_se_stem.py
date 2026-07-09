"""
Depthwise-separable CNN stem with per-block squeeze-and-excitation.

Each block applies a depthwise temporal convolution, a pointwise channel-mixing
convolution, batch normalization, activation, squeeze-and-excitation gating,
and optional pooling.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class SEBlock1d(nn.Module):
    """
    Squeeze-and-excitation block for 1D feature maps.

    Uses global average pooling over time to produce a channel-wise gate.
    """

    def __init__(self, channels: int, reduction_ratio: int, activation_cls):
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced),
            activation_cls(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.squeeze(x).squeeze(-1)      # [B, C]
        gate = self.excite(gate).unsqueeze(-1)  # [B, C, 1]
        return x * gate


class DepthwiseSeparableBlock1d(nn.Module):
    """
    Depthwise-separable convolution block with squeeze-and-excitation.

    Applies depthwise convolution, pointwise projection, normalization,
    activation, SE gating, and optional pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation_cls,
        se_reduction_ratio: int,
        pool_cls,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")
        
        conv_stride = 1 if pool_cls is not None else 2

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation_cls()
        self.se = SEBlock1d(out_channels, se_reduction_ratio, activation_cls)
        self.pool = pool_cls(kernel_size=2) if pool_cls is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.se(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class DepthwiseSeparableSEStem(nn.Module):
    """
    CNN stem composed of stacked depthwise-separable SE blocks.

    The stem expects 12-channel ECG input and produces a feature sequence for the
    transformer encoder.
    """

    def __init__(self, cfg):
        super().__init__()

        activation_cls = _ACTIVATIONS[cfg.cnn_activation]

        kernel_sizes = (
            cfg.dwsep_kernel_sizes
            if cfg.dwsep_kernel_sizes is not None
            else [cfg.cnn_kernel_size] * len(cfg.dwsep_channels)
        )
        if len(kernel_sizes) != len(cfg.dwsep_channels):
            raise ValueError(
                f"dwsep_kernel_sizes ({len(kernel_sizes)}) must match "
                f"dwsep_channels ({len(cfg.dwsep_channels)}) in length."
            )

        if cfg.cnn_pooling == "max":
            pool_cls = nn.MaxPool1d
        elif cfg.cnn_pooling == "avg":
            pool_cls = nn.AvgPool1d
        elif cfg.cnn_pooling == "none":
            pool_cls = None
        else:
            raise ValueError(f"Unknown cnn_pooling: {cfg.cnn_pooling!r}")

        in_channels = 12
        blocks = []
        for out_ch, kernel_size in zip(cfg.dwsep_channels, kernel_sizes):
            blocks.append(
                DepthwiseSeparableBlock1d(
                    in_channels,
                    out_ch,
                    kernel_size,
                    activation_cls,
                    cfg.se_reduction_ratio,
                    pool_cls,
                )
            )
            in_channels = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = cfg.dwsep_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)