"""
Register a PyTorch checkpoint with the MLflow Model Registry.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient

from configs.config import ModelConfig
from src.models.model import MultimodalECGClassifier


def register(
    run_id: str,
    checkpoint_path: Path,
    registered_model_name: str,
    architecture_tag: str,
    alias: str,
    extra_pip_requirements: list[str],
) -> None:
    """Log and register a checkpoint under its original MLflow run."""
    client = MlflowClient()
    source_run = client.get_run(run_id)
    mlflow.set_experiment(experiment_id=source_run.info.experiment_id)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ModelConfig(**checkpoint["model_cfg"])
    model = MultimodalECGClassifier(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            serialization_format="pickle",
            extra_pip_requirements=extra_pip_requirements,
            metadata={
                "selection_metric": "validation_macro_f1",
                "selected_checkpoint_epoch": checkpoint["epoch"],
            },
        )

    version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=registered_model_name,
    )

    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "architecture",
        architecture_tag,
    )
    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "validation_macro_f1",
        str(checkpoint["val_macro_f1"]),
    )
    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "validation_macro_auc",
        str(checkpoint["val_macro_auc"]),
    )
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version.version,
    )

    print(
        f"Registered {registered_model_name} version "
        f"{version.version} with alias @{alias}"
    )


def main() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--architecture-tag", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--extra-pip-requirements", nargs="*", default=["transformers>=4.38.0", "peft>=0.10.0"])
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    register(
        run_id=args.run_id,
        checkpoint_path=Path(args.checkpoint_path),
        registered_model_name=args.registered_model_name,
        architecture_tag=args.architecture_tag,
        alias=args.alias,
        extra_pip_requirements=args.extra_pip_requirements,
    )


if __name__ == "__main__":
    main()