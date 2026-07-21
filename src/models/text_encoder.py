"""
Text encoder consisting of a MedBERT.de backbone and a trainable projection
head with optional LoRA adaptation.

The backbone's CLS representation is projected into the shared embedding
space used by the classifier and fusion module.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Encode clinical text with a frozen or LoRA-adapted MedBERT.de backbone."""
    
    def __init__(self, cfg):
        super().__init__()

        self.use_lora = cfg.use_lora
        backbone = AutoModel.from_pretrained(cfg.text_model_name)
        hidden_size = backbone.config.hidden_size

        if self.use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            )
            self.text_model = get_peft_model(backbone, lora_cfg)
        else:
            self.text_model = backbone
            for param in self.text_model.parameters():
                param.requires_grad = False
            self.text_model.eval()

        # Project the CLS representation into the shared embedding space
        self.projection = nn.Linear(hidden_size, cfg.text_projection_dim)

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep dropout disabled when the backbone is fully frozen.
        if not self.use_lora:
            self.text_model.eval()
        return self

    def extract_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the backbone's CLS representation before projection.

        Gradient tracking is disabled when the backbone is fully frozen.
        """
        if self.use_lora:
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
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
        Project either tokenized text or a cached backbone representation.

        Args:
            input_ids: Token IDs with shape [batch, seq_len].
            attention_mask: Attention mask with shape [batch, seq_len].
            cached_embedding: Precomputed CLS features with shape
                [batch, hidden_size].

        Returns:
            Projected text features with shape [batch, text_projection_dim].
        """
        if cached_embedding is not None:
            if self.use_lora:
                raise ValueError(
                    "Cached text features cannot be used when LoRA is enabled."
                )
            features = cached_embedding
        else:
            if input_ids is None or attention_mask is None:
                raise ValueError("Provide cached_embedding or tokenized text inputs.")
            features = self.extract_features(input_ids, attention_mask)

        return self.projection(features)