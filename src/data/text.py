"""
Tokenization utilities for German clinical report text.
"""

from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "GerMedBERT/medbert-512"):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize(
    texts: list[str],
    tokenizer,
    max_length: int = 128,
) -> dict:
    """
    Tokenize and pad report text to fixed-length PyTorch tensors.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
