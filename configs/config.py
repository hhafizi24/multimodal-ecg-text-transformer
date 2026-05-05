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
    # Controls which branches are active in the model
    mode: Literal["signal_only", "text_only", "fusion"] = "fusion"
    cnn_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 7
    transformer_hidden_dim: int = 256
    transformer_num_heads: int = 8
    transformer_num_layers: int = 3
    transformer_dropout: float = 0.1
    text_model_name: str = "distilbert-base-multilingual-cased"
    # Must match transformer_hidden_dim
    text_projection_dim: int = 256
    fusion_num_heads: int = 8
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

# Register configs for optional Hydra CLI use.
ecg_store = store(group="ecg")
ecg_store(DataConfig, name="default_data")
ecg_store(ModelConfig, name="signal_only", mode="signal_only")
ecg_store(ModelConfig, name="text_only", mode="text_only")
ecg_store(ModelConfig, name="fusion", mode="fusion")
ecg_store(TrainingConfig, name="default_training")
ecg_store(ExportConfig, name="default_export")
ecg_store(BenchmarkConfig, name="default_benchmark")
store.add_to_hydra_store()