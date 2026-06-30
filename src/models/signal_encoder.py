"""
Signal encoder: CNN stem followed by a transformer encoder.

The CNN reduces the 1000-timestep sequence to ~125 tokens, either through
stride-2 convolutions or explicit pooling blocks, then projects to
transformer_hidden_dim. A transformer encoder contextualizes the token sequence.
Mean pooling over time produces the final fixed-size signal embedding.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class SignalEncoder(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig. Relevant fields:
                cnn_channels, cnn_kernel_size, cnn_kernel_sizes,
                cnn_activation, cnn_pooling, cnn_dropout,
                transformer_hidden_dim, transformer_num_heads,
                transformer_num_layers, transformer_dropout.
        """
        super().__init__()

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

        activation_cls = _ACTIVATIONS[cfg.cnn_activation]

        if cfg.cnn_pooling == "max":
            pool_cls = nn.MaxPool1d
        elif cfg.cnn_pooling == "avg":
            pool_cls = nn.AvgPool1d
        elif cfg.cnn_pooling == "none":
            pool_cls = None
        else:
            raise ValueError(f"Unknown cnn_pooling: {cfg.cnn_pooling!r}")

        conv_stride = 1 if pool_cls is not None else 2

        in_channels = 12   # one channel per ECG lead
        cnn_layers = []

        # If pooling is enabled, convolution extracts features before downsampling.
        # Otherwise, stride-2 convolution preserves the original baseline behavior.
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

            cnn_layers += block
            in_channels = out_ch

        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_dropout = nn.Dropout(p=cfg.cnn_dropout)

        self.input_proj = nn.Linear(cfg.cnn_channels[-1], cfg.transformer_hidden_dim)

        # Positional embedding table sized with headroom above the ~125 tokens
        # produced by the CNN stem.
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
        Args:
            x: [batch, 1000, 12]

        Returns:
            embedding: [batch, transformer_hidden_dim]
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