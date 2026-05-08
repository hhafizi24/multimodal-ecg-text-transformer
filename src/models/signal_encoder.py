"""
Signal encoder: CNN stem followed by a transformer encoder.

The CNN reduces the 1000-timestep sequence to ~125 tokens via three stride-2
conv layers, then projects to transformer_hidden_dim. A standard transformer
encoder then contextualizes the token sequence. Mean pooling over the time
dimension produces the final fixed-size signal embedding.
"""

import torch
import torch.nn as nn


class SignalEncoder(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig. Relevant fields:
                cnn_channels, cnn_kernel_size, transformer_hidden_dim,
                transformer_num_heads, transformer_num_layers, transformer_dropout.
        """
        super().__init__()

        in_channels = 12  # one per ECG lead
        channels = cfg.cnn_channels  # e.g. [32, 64, 128]

        # CNN stem — each layer halves the time dimension via stride 2.
        # BatchNorm before activation keeps training stable on ECG amplitudes.
        cnn_layers = []
        for out_ch in channels:
            cnn_layers += [
                nn.Conv1d(
                    in_channels, out_ch,
                    kernel_size=cfg.cnn_kernel_size,
                    stride=2,
                    padding=cfg.cnn_kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_channels = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Project CNN output channels to transformer hidden dim
        self.input_proj = nn.Linear(channels[-1], cfg.transformer_hidden_dim)

        # Learnable positional embeddings — max_len set with headroom above the
        # ~125 tokens produced after 3 stride-2 layers over 1000 timesteps.
        self.pos_embedding = nn.Embedding(200, cfg.transformer_hidden_dim)
        self.pos_drop = nn.Dropout(p=cfg.transformer_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_hidden_dim,
            nhead=cfg.transformer_num_heads,
            dim_feedforward=cfg.transformer_hidden_dim * 4,
            dropout=cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.transformer_num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1000, 12]
        Returns:
            embedding: [batch, transformer_hidden_dim]
        """
        # Conv1d expects [batch, channels, time]
        x = x.permute(0, 2, 1)          # [B, 12, 1000]
        x = self.cnn(x)                  # [B, C, ~125]
        x = x.permute(0, 2, 1)          # [B, ~125, C]
        x = self.input_proj(x)           # [B, ~125, hidden_dim]

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.pos_drop(x + self.pos_embedding(positions))  # [B, ~125, hidden_dim]

        x = self.transformer(x)          # [B, ~125, hidden_dim]
        return x.mean(dim=1)             # [B, hidden_dim]
