"""
Text encoder consisting of a frozen MedBERT.de backbone and a trainable
projection head.

The frozen language model produces a CLS-position representation, which is
projected into the shared embedding space.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig. Relevant fields:
                text_model_name, text_projection_dim.
        """
        super().__init__()

        self.text_model = AutoModel.from_pretrained(cfg.text_model_name)

        # Freeze all MedBERT.de parameters
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.text_model.eval()

        # Project CLS-position representation into the shared embedding space
        self.projection = nn.Linear(
            self.text_model.config.hidden_size,  
            cfg.text_projection_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep the frozen backbone deterministic while allowing the
        # projection layer to follow the parent module's mode.
        self.text_model.eval()
        return self

    def extract_frozen_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Runs the frozen backbone and returns the raw CLS-position feature."""
        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state[:, 0, :].clone()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cached_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:        [batch, seq_len] — required unless cached_embedding is given
            attention_mask:   [batch, seq_len] — required unless cached_embedding is given
            cached_embedding: [batch, hidden_size] precomputed CLS feature, bypasses the backbone
        Returns:
            embedding: [batch, text_projection_dim]
        """
        if cached_embedding is not None:
            features = cached_embedding
        else:
            if input_ids is None or attention_mask is None:
                raise ValueError("Provide cached_embedding or tokenized text inputs.")
            features = self.extract_frozen_features(input_ids, attention_mask)

        return self.projection(features)
