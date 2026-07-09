"""
Cross-attention fusion over fixed-size signal and text embeddings.

The signal embedding is used as the query, while the text embedding provides
the key and value. The attended text representation is added to the original
signal embedding through a residual connection, followed by layer normalization.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig. Relevant fields:
                transformer_hidden_dim, fusion_num_heads.
        """
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.transformer_hidden_dim,
            num_heads=cfg.fusion_num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.transformer_hidden_dim)

    def forward(
        self,
        signal_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            signal_emb: [batch, hidden_dim] — query
            text_emb:   [batch, hidden_dim] — key and value
        Returns:
            fused: [batch, hidden_dim]
        """
        # Convert pooled embeddings to the 3D shape expected by MultiheadAttention
        q = signal_emb.unsqueeze(1)  # [B, 1, hidden_dim]
        k = text_emb.unsqueeze(1)    # [B, 1, hidden_dim]

        attended, _ = self.cross_attn(query=q, key=k, value=k)
        attended = attended.squeeze(1)  # [B, hidden_dim]

        # Residual update anchored on the signal representation
        return self.norm(signal_emb + attended)
