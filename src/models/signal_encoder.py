"""
Signal encoder: CNN stem followed by a transformer encoder.

The CNN reduces the input sequence before projecting it to
`transformer_hidden_dim`. A transformer encoder contextualizes the resulting
token sequence, and mean pooling produces a fixed-length signal embedding.
"""

import torch
import torch.nn as nn

from src.models.multiscale_stem import MultiScaleStem

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class SignalEncoder(nn.Module):
    """
    Encodes ECG signals into fixed-length embeddings using a configurable CNN
    stem followed by a transformer encoder.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Model configuration controlling the CNN stem and transformer encoder.
        """
        super().__init__()

        activation_cls = _ACTIVATIONS[cfg.cnn_activation]

        # Build either the original sequential CNN stem or the parallel
        # multi-scale stem. Both produce feature sequences that are projected
        # into the shared transformer embedding space.
        if cfg.cnn_stem == "multiscale":
            self.cnn = MultiScaleStem(cfg)
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

            # If pooling is enabled, convolution extracts features before
            # downsampling. Otherwise, stride-2 convolution preserves the
            # original baseline behavior.
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

        # Positional embedding table sized with headroom above the sequence
        # lengths produced by either CNN stem.
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
        Encode an ECG recording into a fixed-length embedding.

        Args:
            x: Input tensor of shape [batch, time, channels].

        Returns:
            Signal embedding of shape [batch, transformer_hidden_dim].
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

        # Mean pooling over the sequence dimension produces a single embedding
        # per ECG recording.
        return x.mean(dim=1)