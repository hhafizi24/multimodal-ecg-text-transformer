"""
Hydra-Zen configuration for the multimodal ECG-text transformer.

Configuration objects for data, model, training, export, and benchmarking.
"""
from dataclasses import dataclass, field
from typing import Literal
from hydra_zen import store

@dataclass
class DataConfig:
    processed_data_dir: str = "data/processed"
    norm_stats_path: str = "data/processed/norm_stats.json"
    batch_size: int = 32
    num_workers: int = 2
    max_text_length: int = 128

@dataclass
class ModelConfig:
    # Active model branch: signal-only, text-only, or fusion
    mode: Literal["signal_only", "text_only", "fusion"] = "fusion"
    cnn_channels: list[int] = field(default_factory=lambda: [32, 64, 128])

    # Used for every CNN layer when cnn_kernel_sizes is None
    cnn_kernel_size: int = 7

    # Optional per-layer kernel sizes. Must match len(cnn_channels)
    cnn_kernel_sizes: list[int] | None = None

    cnn_activation: Literal["relu", "gelu", "silu", "leaky_relu"] = "gelu"
    cnn_pooling: Literal["none", "max", "avg"] = "none"
    cnn_dropout: float = 0.0

    # CNN stem implementation used by the signal encoder
    cnn_stem: Literal["sequential", "multiscale"] = "sequential"

    # Used only when cnn_stem="multiscale"
    multiscale_branch_channels: list[int] = field(
        default_factory=lambda: [64, 64, 64]
    )
    multiscale_kernel_sizes: list[int] = field(
        default_factory=lambda: [11, 21, 41]
    )

    transformer_hidden_dim: int = 256
    transformer_num_heads: int = 8
    transformer_num_layers: int = 3
    transformer_dropout: float = 0.1

    text_model_name: str = "GerMedBERT/medbert-512"
    
    # Must match transformer_hidden_dim
    text_projection_dim: int = 256

    fusion_num_heads: int = 8

    # Optional hidden layer in the classification head
    classifier_hidden_dim: int | None = None
    classifier_activation: Literal["relu", "gelu", "silu", "leaky_relu"] = "gelu"
    classifier_dropout: float = 0.0

    num_classes: int = 5

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 30
    weight_decay: float = 1e-4

    # Learning rate schedule
    scheduler: Literal["cosine", "step"] = "cosine"
    checkpoint_dir: str = "models"
    experiment_name: str = "ecg_multimodal"
    use_class_weights: bool = True
    class_weights_path: str = "data/processed/config_snapshot.json"
    
    # Set to None to disable early stopping
    early_stopping_patience: int | None = 5

    loss_fn: Literal["cross_entropy", "focal"] = "cross_entropy"
    # Used only when loss_fn="focal"
    focal_gamma: float = 2.0

    # Controls initialization, shuffling, augmentation, and dropout RNGs
    seed: int = 42

@dataclass
class ExportConfig:
    checkpoint_path: str = "models/best_model.pt"
    onnx_path: str = "models/model.onnx"
    quantized_onnx_path: str = "models/model_quantized.onnx"
    opset_version: int = 14

@dataclass
class BenchmarkConfig:
    n_warmup_runs: int = 20
    n_benchmark_runs: int = 200
    api_url: str = "http://localhost:8000/predict"
    test_data_path: str = "data/processed/test"
    output_json: str = "results/benchmark.json"
    output_figure: str = "results/figures/benchmark_comparison.png"

# Register configs for optional Hydra CLI use
ecg_store = store(group="ecg")
ecg_store(DataConfig, name="default_data")
ecg_store(ModelConfig, name="signal_only", mode="signal_only")
ecg_store(ModelConfig, name="text_only", mode="text_only")
ecg_store(ModelConfig, name="fusion", mode="fusion")
ecg_store(TrainingConfig, name="default_training")
ecg_store(ExportConfig, name="default_export")
ecg_store(BenchmarkConfig, name="default_benchmark")
store.add_to_hydra_store()