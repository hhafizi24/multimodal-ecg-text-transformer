"""
Configurable classification head for unimodal and fused embeddings.
"""

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class ClassificationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.classifier_hidden_dim is None:
            self.net = nn.Linear(cfg.transformer_hidden_dim, cfg.num_classes)
        else:
            if cfg.classifier_activation not in _ACTIVATIONS:
                raise ValueError(
                    f"Unknown classifier_activation: {cfg.classifier_activation!r}"
                )

            activation_cls = _ACTIVATIONS[cfg.classifier_activation]

            self.net = nn.Sequential(
                nn.Linear(cfg.transformer_hidden_dim, cfg.classifier_hidden_dim),
                activation_cls(),
                nn.Dropout(p=cfg.classifier_dropout),
                nn.Linear(cfg.classifier_hidden_dim, cfg.num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map embeddings to class logits.
        """
        return self.net(x)
