"""
CNN-Transformer encoder for fixed-length ECG signal embeddings.
"""

import torch
import torch.nn as nn

from src.models.multiscale_stem import MultiScaleStem
from src.models.depthwise_se_stem import DepthwiseSeparableSEStem

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class SignalEncoder(nn.Module):
    """
    Encode ECG signals with a configurable CNN stem and Transformer encoder.
    """

    def __init__(self, cfg):
        super().__init__()

        activation_cls = _ACTIVATIONS[cfg.cnn_activation]

        if cfg.cnn_stem == "multiscale":
            self.cnn = MultiScaleStem(cfg)
            cnn_out_channels = self.cnn.out_channels

        elif cfg.cnn_stem == "depthwise_se":
            self.cnn = DepthwiseSeparableSEStem(cfg)
            cnn_out_channels = self.cnn.out_channels

        elif cfg.cnn_stem == "sequential":
            kernel_sizes = (
                cfg.cnn_kernel_sizes
                if cfg.cnn_kernel_sizes is not None
                else [cfg.cnn_kernel_size] * len(cfg.cnn_channels)
            )

            if len(kernel_sizes) != len(cfg.cnn_channels):
                raise ValueError(
                    f"cnn_kernel_sizes ({len(kernel_sizes)}) must match "
                    f"cnn_channels ({len(cfg.cnn_channels)}) in length."
                )

            if cfg.cnn_pooling == "max":
                pool_cls = nn.MaxPool1d
            elif cfg.cnn_pooling == "avg":
                pool_cls = nn.AvgPool1d
            elif cfg.cnn_pooling == "none":
                pool_cls = None
            else:
                raise ValueError(f"Unknown cnn_pooling: {cfg.cnn_pooling!r}")

            conv_stride = 1 if pool_cls is not None else 2

            in_channels = 12
            cnn_layers = []

            # Use stride-2 convolutions when pooling is disabled to keep downsampling consistent
            for out_ch, kernel_size in zip(cfg.cnn_channels, kernel_sizes):
                block = [
                    nn.Conv1d(
                        in_channels,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=conv_stride,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_ch),
                    activation_cls(),
                ]

                if pool_cls is not None:
                    block.append(pool_cls(kernel_size=2))

                cnn_layers.extend(block)
                in_channels = out_ch

            self.cnn = nn.Sequential(*cnn_layers)
            cnn_out_channels = cfg.cnn_channels[-1]

        else:
            raise ValueError(f"Unknown cnn_stem: {cfg.cnn_stem!r}")

        self.cnn_dropout = nn.Dropout(p=cfg.cnn_dropout)
        self.input_proj = nn.Linear(cnn_out_channels, cfg.transformer_hidden_dim)

        # Leave room for the longest sequence produced by any supported CNN stem
        self.pos_embedding = nn.Embedding(200, cfg.transformer_hidden_dim)
        self.pos_drop = nn.Dropout(p=cfg.transformer_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_hidden_dim,
            nhead=cfg.transformer_num_heads,
            dim_feedforward=cfg.transformer_hidden_dim * 4,
            dropout=cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map ECG tensors from [batch, time, channels] to [batch, hidden_dim].
        """
        x = x.permute(0, 2, 1)           # [B, 12, 1000]
        x = self.cnn(x)                  # [B, C, ~125]
        x = self.cnn_dropout(x)
        x = x.permute(0, 2, 1)           # [B, ~125, C]
        x = self.input_proj(x)           # [B, ~125, hidden_dim]

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.pos_drop(x + self.pos_embedding(positions))

        x = self.transformer(x)

        return x.mean(dim=1)
