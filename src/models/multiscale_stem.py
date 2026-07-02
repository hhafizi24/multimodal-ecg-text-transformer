"""
Multi-scale CNN stem for ECG signal encoding.

Runs parallel 1D convolution branches with different kernel sizes, then
concatenates their outputs along the channel dimension. Smaller kernels capture
sharper waveform features, while wider kernels capture slower morphology.

Each branch applies Conv1d, batch normalization, an activation function, and
three stages of average pooling before concatenation.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class _Branch(nn.Module):
    """Single branch of the multi-scale CNN stem."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation_cls,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {kernel_size}.")

        # Mirror the downsampling performed by the sequential CNN stem so both
        # architectures produce sequences of comparable length.
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            activation_cls(),
            nn.AvgPool1d(kernel_size=2),
            nn.AvgPool1d(kernel_size=2),
            nn.AvgPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one scale of the input signal."""
        return self.net(x)


class MultiScaleStem(nn.Module):
    """
    Parallel multi-scale CNN stem.

    Processes the input ECG using multiple convolution branches with different
    kernel sizes. The branch outputs are concatenated along the channel dimension
    to produce a shared feature representation for the signal encoder.

    Args:
        cfg: Model configuration containing the multiscale stem parameters.
        in_channels: Number of ECG input channels.
    """

    def __init__(self, cfg, in_channels: int = 12):
        super().__init__()

        branch_channels = cfg.multiscale_branch_channels
        kernel_sizes = cfg.multiscale_kernel_sizes

        if len(branch_channels) != len(kernel_sizes):
            raise ValueError(
                f"multiscale_branch_channels ({len(branch_channels)}) and "
                f"multiscale_kernel_sizes ({len(kernel_sizes)}) must match."
            )

        if cfg.cnn_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown cnn_activation: {cfg.cnn_activation!r}")

        activation_cls = _ACTIVATIONS[cfg.cnn_activation]

        self.branches = nn.ModuleList(
            [
                _Branch(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    activation_cls=activation_cls,
                )
                for out_ch, kernel_size in zip(branch_channels, kernel_sizes)
            ]
        )

        # Total channel dimension after concatenating all branches.
        self.out_channels = sum(branch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all convolution branches and concatenate their outputs.

        Args:
            x: Input tensor of shape [batch, channels, time].

        Returns:
            Tensor of shape [batch, sum(branch_channels), sequence_length].
        """
        outputs = [branch(x) for branch in self.branches]

        # Branch outputs that differ by one timestep. Trim all 
        # branches to the shortest length before concatenation.
        min_len = min(out.size(-1) for out in outputs)
        outputs = [out[..., :min_len] for out in outputs]

        return torch.cat(outputs, dim=1)