"""
Text encoder: frozen MedBERTde clinical language encoder with a trainable projection head.

MedBERT.de remains frozen during training, while a linear projection maps the
768-dimensional CLS-position representation into the shared embedding space.
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

        # Freeze all MedBERTde parameters
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Project CLS-position representation into the shared embedding space
        self.projection = nn.Linear(
            self.text_model.config.hidden_size,  
            cfg.text_projection_dim,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            embedding: [batch, text_projection_dim]
        """
        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        cls_token = outputs.last_hidden_state[:, 0, :].clone()  # [B, 768] — CLS-position representation
        return self.projection(cls_token)                       # [B, text_projection_dim]
