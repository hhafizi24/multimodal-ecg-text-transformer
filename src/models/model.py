"""
Unified multimodal ECG classifier.

A single model class covering all three experimental stages. Active
components are controlled by cfg.mode, and unused branches are not
instantiated for a given configuration.
"""

import torch
import torch.nn as nn

from src.models.classifier import ClassificationHead
from src.models.fusion import CrossAttentionFusion
from src.models.signal_encoder import SignalEncoder
from src.models.text_encoder import TextEncoder


class MultimodalECGClassifier(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig. cfg.mode controls which branches are active:
                "signal_only" — ECG encoder + classifier
                "text_only"   — text encoder + classifier
                "fusion"      — both encoders + cross-attention + classifier
        """
        super().__init__()

        self.mode = cfg.mode

        valid_modes = ("signal_only", "text_only", "fusion")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {valid_modes}.")

        if self.mode in ("signal_only", "fusion"):
            self.signal_encoder = SignalEncoder(cfg)

        if self.mode in ("text_only", "fusion"):
            self.text_encoder = TextEncoder(cfg)

        if self.mode == "fusion":
            self.fusion = CrossAttentionFusion(cfg)

        self.classifier = ClassificationHead(cfg)

    def forward(
        self,
        signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        All three inputs are always passed in regardless of mode — unused
        tensors are simply ignored. This keeps the training loop and
        DataLoader uniform across all stages.

        Args:
            signal:         [batch, 1000, 12]
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        if self.mode == "signal_only":
            emb = self.signal_encoder(signal)

        elif self.mode == "text_only":
            emb = self.text_encoder(input_ids, attention_mask)

        else:  # fusion
            sig_emb = self.signal_encoder(signal)
            txt_emb = self.text_encoder(input_ids, attention_mask)
            emb = self.fusion(sig_emb, txt_emb)

        return self.classifier(emb)
