"""
Cross-attention fusion for pooled signal and text embeddings.

The signal embedding attends over the signal-text pair, with optional
text masking for modality-dropout training and deterministic ablation.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if not 0.0 <= cfg.text_modality_dropout_p <= 1.0:
            raise ValueError("text_modality_dropout_p must be between 0 and 1.")

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.transformer_hidden_dim,
            num_heads=cfg.fusion_num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.transformer_hidden_dim)
        self.text_modality_dropout_p = cfg.text_modality_dropout_p

    def forward(
        self,
        signal_emb: torch.Tensor,
        text_emb: torch.Tensor,
        text_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fuse signal and text embeddings.

        Args:
            signal_emb:     Signal embeddings with shape [batch, hidden_dim].
            text_emb:       Text embeddings with shape [batch, hidden_dim].
            text_available: Boolean mask indicating available text inputs.
                             
        Returns:
            Fused embeddings with shape [batch, hidden_dim].
        """
        batch_size = signal_emb.size(0)
        query = signal_emb.unsqueeze(1)
        kv = torch.stack([signal_emb, text_emb], dim=1)  # [B, 2, hidden_dim]

        if text_available is not None:
            drop_text = ~text_available.to(
                device=signal_emb.device,
                dtype=torch.bool,
            )
        elif self.training and self.text_modality_dropout_p > 0:
            drop_text = (
                torch.rand(batch_size, device=signal_emb.device)
                < self.text_modality_dropout_p
            )
        else:
            drop_text = None

        key_padding_mask = None
        if drop_text is not None:
            key_padding_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=signal_emb.device)
            key_padding_mask[:, 1] = drop_text

        attended, _ = self.cross_attn(
            query,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = attended.squeeze(1)

        return self.norm(signal_emb + attended)