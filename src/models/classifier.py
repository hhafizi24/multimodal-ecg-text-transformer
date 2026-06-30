"""
Classification head.

Uses a single linear layer by default. When classifier_hidden_dim is set, adds
one hidden layer with configurable activation and dropout before the final
projection.
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
        """
        Args:
            cfg: ModelConfig. Relevant fields:
                transformer_hidden_dim, classifier_hidden_dim,
                classifier_activation, classifier_dropout, num_classes.
        """
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
        Args:
            x: [batch, hidden_dim]

        Returns:
            logits: [batch, num_classes]
        """
        return self.net(x)