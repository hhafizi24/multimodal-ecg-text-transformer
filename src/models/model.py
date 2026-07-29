"""
Unified classifier for signal-only, text-only, and fused ECG inputs.
"""

import torch
import torch.nn as nn

from src.models.classifier import ClassificationHead
from src.models.fusion import CrossAttentionFusion
from src.models.signal_encoder import SignalEncoder
from src.models.text_encoder import TextEncoder


class MultimodalECGClassifier(nn.Module):
    """
    Configure signal-only, text-only, or cross-attention fusion classification.
    """

    def __init__(self, cfg):
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

        self.freeze_encoders = cfg.freeze_encoders
        if self.freeze_encoders:
            if hasattr(self, "signal_encoder"):
                for p in self.signal_encoder.parameters():
                    p.requires_grad = False
            if hasattr(self, "text_encoder"):
                for p in self.text_encoder.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        """
        Set training mode while keeping frozen encoders in evaluation mode.
        """
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
        Return class logits from raw inputs or precomputed embeddings.

        Args:
            signal: Raw ECG tensor with shape [batch, samples, leads].
            input_ids: Tokenized report IDs.
            attention_mask: Token mask for the report encoder.
            cached_embedding: Cached text-backbone features for text-only
                inference.
            signal_embedding: Precomputed signal features for fusion.
            text_embedding: Precomputed text features for fusion.
            text_available: Per-sample mask indicating whether report text is
                available.

        Returns:
            Class logits with shape [batch, num_classes].

        Notes:
            Fusion embeddings must be supplied together.
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

        else:
            sig_emb = signal_embedding if signal_embedding is not None else self.signal_encoder(signal)
            txt_emb = text_embedding if text_embedding is not None else self.text_encoder(input_ids, attention_mask, cached_embedding)
            emb = self.fusion(
                sig_emb,
                txt_emb,
                text_available=text_available,
            )

        return self.classifier(emb)
