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

        # Freeze pretrained encoders during fusion training
        self.freeze_encoders = cfg.freeze_encoders
        if self.freeze_encoders:
            if hasattr(self, "signal_encoder"):
                for p in self.signal_encoder.parameters():
                    p.requires_grad = False
            if hasattr(self, "text_encoder"):
                for p in self.text_encoder.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            if hasattr(self, "signal_encoder"):
                self.signal_encoder.eval()
            if hasattr(self, "text_encoder"):
                self.text_encoder.eval()
        return self

    def forward(
        self,
        signal: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cached_embedding: torch.Tensor | None = None,
        signal_embedding: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            signal:           [batch, 1000, 12]
            input_ids:        [batch, seq_len]
            attention_mask:   [batch, seq_len]
            cached_embedding: [batch, hidden_size], bypasses the text backbone
            signal_embedding: [batch, hidden_dim], bypasses the signal encoder
            text_embedding:   [batch, hidden_dim], bypasses the text encoder
            text_available:   [batch] bool, forces deterministic text ablation
        Returns:
            logits: [batch, num_classes]
        """
        if self.mode == "fusion" and (
            (signal_embedding is None) != (text_embedding is None)
        ):
            raise ValueError(
                "signal_embedding and text_embedding must be provided together."
            )
        
        if self.mode == "signal_only":
            emb = signal_embedding if signal_embedding is not None else self.signal_encoder(signal)

        elif self.mode == "text_only":
            emb = text_embedding if text_embedding is not None else self.text_encoder(input_ids, attention_mask, cached_embedding)

        else:  # fusion
            sig_emb = signal_embedding if signal_embedding is not None else self.signal_encoder(signal)
            txt_emb = text_embedding if text_embedding is not None else self.text_encoder(input_ids, attention_mask, cached_embedding)
            emb = self.fusion(
                sig_emb,
                txt_emb,
                text_available=text_available,
            )

        return self.classifier(emb)