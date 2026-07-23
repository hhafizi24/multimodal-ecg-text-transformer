import torch

from src.training.evaluate import metrics_from_logits


def fit_logit_bias(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    bias_range: tuple[float, float] = (-1.0, 1.0),
    step: float = 0.05,
    num_passes: int = 5,
) -> torch.Tensor:
    """Fit a per-class logit bias vector via coordinate descent on macro F1."""
    num_steps = round((bias_range[1] - bias_range[0]) / step) + 1
    candidates = torch.linspace(bias_range[0], bias_range[1], num_steps)

    bias = torch.zeros(num_classes)
    best_f1 = metrics_from_logits(logits, labels, bias)["macro_f1"]

    for _ in range(num_passes):
        for c in range(num_classes):
            best_value = bias[c].item()
            for candidate in candidates:
                trial_bias = bias.clone()
                trial_bias[c] = candidate
                f1 = metrics_from_logits(logits, labels, trial_bias)["macro_f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_value = candidate.item()
            bias[c] = best_value

    return bias