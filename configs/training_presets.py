"""Hydra-Zen presets for signal, text, and fusion training."""

from hydra_zen import builds, make_config, store

from configs.config import DataConfig, ModelConfig, TrainingConfig


SignalOnlyTrainConfig = make_config(
    model_cfg=builds(
        ModelConfig,
        mode="signal_only",
        cnn_channels=[64, 64, 64, 128, 128],
        cnn_kernel_sizes=[7, 7, 7, 5, 5],
        cnn_activation="gelu",
        cnn_pooling="avg",
        cnn_dropout=0.1,
        transformer_hidden_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=3,
        transformer_dropout=0.2,
        classifier_hidden_dim=256,
        classifier_activation="gelu",
        classifier_dropout=0.3,
        num_classes=5,
        populate_full_signature=True,
    ),
    data_cfg=builds(
        DataConfig,
        batch_size=64,
        num_workers=2,
        max_text_length=128,
        populate_full_signature=True,
    ),
    train_cfg=builds(
        TrainingConfig,
        learning_rate=3e-5,
        num_epochs=50,
        weight_decay=5e-3,
        scheduler="cosine",
        experiment_name="signal_only",
        run_name=None,
        use_class_weights=True,
        early_stopping_patience=8,
        loss_fn="focal",
        focal_gamma=1.5,
        seed=42,
        populate_full_signature=True,
    ),
    device="auto",
)


TextOnlyTrainConfig = make_config(
    model_cfg=builds(
        ModelConfig,
        mode="text_only",
        text_model_name="GerMedBERT/medbert-512",
        text_projection_dim=256,
        transformer_hidden_dim=256,
        classifier_hidden_dim=256,
        classifier_activation="gelu",
        classifier_dropout=0.3,
        num_classes=5,
        use_lora=True,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["query", "value"],
        populate_full_signature=True,
    ),
    data_cfg=builds(
        DataConfig,
        batch_size=64,
        num_workers=2,
        max_text_length=128,
        populate_full_signature=True,
    ),
    train_cfg=builds(
        TrainingConfig,
        learning_rate=3e-4,
        num_epochs=20,
        weight_decay=1e-4,
        scheduler="cosine",
        experiment_name="text_only",
        run_name=None,
        use_class_weights=True,
        early_stopping_patience=5,
        loss_fn="cross_entropy",
        seed=42,
        populate_full_signature=True,
    ),
    device="auto",
)


FusionTrainConfig = make_config(
    model_cfg=builds(
        ModelConfig,
        mode="fusion",
        freeze_encoders=True,
        text_modality_dropout_p=0.3,
        fusion_num_heads=16,
        cnn_channels=[64, 64, 64, 128, 128],
        cnn_kernel_sizes=[7, 7, 7, 5, 5],
        cnn_activation="gelu",
        cnn_pooling="avg",
        cnn_dropout=0.1,
        transformer_hidden_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=3,
        transformer_dropout=0.2,
        text_model_name="GerMedBERT/medbert-512",
        text_projection_dim=256,
        use_lora=True,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["query", "value"],
        classifier_hidden_dim=256,
        classifier_activation="gelu",
        classifier_dropout=0.3,
        num_classes=5,
        populate_full_signature=True,
    ),
    data_cfg=builds(
        DataConfig,
        batch_size=64,
        num_workers=2,
        max_text_length=128,
        populate_full_signature=True,
    ),
    train_cfg=builds(
        TrainingConfig,
        learning_rate=1e-4,
        num_epochs=30,
        weight_decay=1e-4,
        scheduler="cosine",
        experiment_name="fusion",
        run_name=None,
        use_class_weights=True,
        early_stopping_patience=8,
        loss_fn="focal",
        focal_gamma=1.5,
        label_smoothing=0.0,
        seed=42,
        populate_full_signature=True,
    ),
    fusion_cache_dir="",
    signal_checkpoint="",
    text_checkpoint="",
    device="auto",
)


store(SignalOnlyTrainConfig, name="signal_only")
store(TextOnlyTrainConfig, name="text_only")
store(FusionTrainConfig, name="fusion")
store.add_to_hydra_store()