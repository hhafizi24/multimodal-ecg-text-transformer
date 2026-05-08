"""
Late embedding-level cross-attention fusion module.

Both modalities are fully encoded to fixed-size embeddings before fusion. 
The signal embedding queries the projected text embedding, allowing the 
ECG representation to attend to the clinical language context. A residual 
connection preserves the original signal embedding in case the text branch
contributes little for a given sample.
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
        # MultiheadAttention expects [batch, seq_len, dim] — add sequence dim of 1
        q = signal_emb.unsqueeze(1)  # [B, 1, hidden_dim]
        k = text_emb.unsqueeze(1)    # [B, 1, hidden_dim]

        attended, _ = self.cross_attn(query=q, key=k, value=k)
        attended = attended.squeeze(1)  # [B, hidden_dim]

        # Residual + layer norm
        return self.norm(signal_emb + attended)
