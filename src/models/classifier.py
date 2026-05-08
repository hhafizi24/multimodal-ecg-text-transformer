"""
Classification head.

A single linear layer mapping from the shared embedding space to class logits.
"""

import torch
import torch.nn as nn
 
 
class ClassificationHead(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig. Relevant fields:
                transformer_hidden_dim, num_classes.
        """
        super().__init__()
        self.linear = nn.Linear(cfg.transformer_hidden_dim, cfg.num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim]
        Returns:
            logits: [batch, num_classes]
        """
        return self.linear(x)
