"""
Load pretrained submodule weights into an assembled model.
"""

from pathlib import Path

import torch


def load_encoder_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    submodule_name: str,
    map_location: str = "cpu",
) -> None:
    """
    Load a submodule's weights from a full-model checkpoint by prefix.

    Args:
        model:           model containing the target submodule as an attribute.
        checkpoint_path: path to a checkpoint saved by train.py.
        submodule_name:  attribute name on model, e.g. "signal_encoder".
        map_location:    torch.load device mapping.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    submodule = getattr(model, submodule_name)

    prefix = f"{submodule_name}."
    state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }

    submodule.load_state_dict(state, strict=True)