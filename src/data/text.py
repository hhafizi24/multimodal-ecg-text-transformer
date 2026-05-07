"""
Tokenization helpers for the clinical report text branch.

Wraps the multilingual DistilBERT tokenizer with project-specific defaults.
The tokenizer is loaded once at the call site and reused — don't instantiate
it inside the Dataset __getitem__.
"""

from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "distilbert-base-multilingual-cased"):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize(
    texts: list[str],
    tokenizer,
    max_length: int = 128,
) -> dict:
    """
    Tokenize a batch of report strings.

    Returns a dict with 'input_ids' and 'attention_mask' as PyTorch tensors,
    truncated and padded to max_length.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )