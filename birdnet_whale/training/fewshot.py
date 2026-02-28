import numpy as np
import tensorflow as tf

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.data.fewshot_augmentation import embedding_mixup, cross_snr_knowledge_transfer
from birdnet_whale.data.sample_weighting import compute_sample_weight
from birdnet_whale.models.adaptive_refiner import build_adaptive_refinement_model
from birdnet_whale.models.losses import FocalLossWithLabelSmoothing
from birdnet_whale.utils.progress import ProgressTracker

SPECIES_FOCAL_WEIGHTS = {
    "NorthernRightWhale": 3.0,
    "BottlenoseDolphin": 2.5,
    "BowheadWhale": 2.0,
    "ClymeneDolphin": 1.5,
    "HumpbackWhale": 1.5,
    "Frasherdolphin": 1.2,
    "KillerWhale": 0.8,
}


def _current_alpha(model) -> float | None:
    try:
        layer = model.get_layer("residual_blend")
    except Exception:
        return None
    alpha_getter = getattr(layer, "current_alpha", None)
    if callable(alpha_getter):
        return float(tf.keras.backend.get_value(alpha_getter()))
    return None


class _AlphaTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.values: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        alpha = _current_alpha(self.model)
        if alpha is not None:
            self.values.append(float(alpha))


class _StageProgressTracker(tf.keras.callbacks.Callback):
    def __init__(self, stage_name: str, total_epochs: int, enabled: bool, percent_step: int):
        super().__init__()
        self._progress = ProgressTracker(
            f"FewShot/{stage_name}",
            total_epochs,
            enabled=enabled and total_epochs > 0,
            percent_step=max(1, percent_step),
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss_val = logs.get("loss")
        alpha = _current_alpha(self.model)
        extra_parts = []
        if loss_val is not None:
            extra_parts.append(f"loss={float(loss_val):.4f}")
        if alpha is not None:
            extra_parts.append(f"alpha={float(alpha):.4f}")
        self._progress.update(extra=" ".join(extra_parts) if extra_parts else None)

    def on_train_end(self, logs=None):
        self._progress.close()


def _clone_encoder(pretrained_encoder, pretrained_weights=None):
    encoder_clone = tf.keras.models.clone_model(pretrained_encoder)
    weights = pretrained_weights if pretrained_weights is not None else pretrained_encoder.get_weights()
    encoder_clone.set_weights(weights)
    return encoder_clone


def get_alpha_init(k: int | None) -> float:
    if k == 2:
        return 0.5
    if k is not None and k <= 8:
        return 0.6
    return 0.7


def _batch_size_for_k(k: int | None, n_samples: int) -> int:
    if k == 2:
        base = 8
    elif k is not None and k <= 8:
        base = 16
    else:
        base = 32
    return max(1, min(base, int(n_samples)))


def compute_training_epochs(n_samples: int, batch_size: int, k: int | None, base_epochs: int, target_steps: int) -> int:
    if n_samples <= 0:
        return int(base_epochs)
    steps_per_epoch = max(1, int(np.ceil(float(n_samples) / float(batch_size))))
    needed_epochs = int(np.ceil(float(target_steps) / float(steps_per_epoch)))
    if k is not None and k <= 4:
        return max(int(base_epochs), needed_epochs)
    if k is not None and k <= 8:
        return max(int(base_epochs), int(np.ceil(0.7 * needed_epochs)))
    return int(base_epochs)


def _to_snr_array(snr_value, n_samples: int, default: float = 0.0):
    if snr_value is None:
        return np.full((n_samples,), float(default), dtype=np.float32)
    arr = np.asarray(snr_value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full((n_samples,), float(arr), dtype=np.float32)
    if arr.shape[0] != n_samples:
        if arr.shape[0] == 1:
            return np.full((n_samples,), float(arr[0]), dtype=np.float32)
        raise ValueError(f"SNR shape mismatch: expected {n_samples}, got {arr.shape[0]}")
    return arr.astype(np.float32)


def _apply_k2_augmentation(
    x_train,
    y_labels,
    effective_snrs,
    target_snrs,
    full_train_data,
    config: ExperimentConfig,
):
    rng = np.random.default_rng(config.split_seed)
    target_arr = None if target_snrs is None else _to_snr_array(target_snrs, len(y_labels))
    mixup_data = embedding_mixup(
        x_train,
        y_labels,
        effective_snrs=effective_snrs,
        target_snrs=target_arr,
        n_mixup=config.mixup_n_per_pair,
        alpha=config.mixup_alpha,
        rng=rng,
    )

    x_out = mixup_data["embeddings"]
    y_out = mixup_data["labels"]
    eff_out = mixup_data["effective_snrs"]
    tgt_out = mixup_data["target_snrs"] if target_snrs is not None else None
    transfer_multiplier = np.ones((len(y_out),), dtype=np.float32)

    avg_target_snr = None if tgt_out is None else float(np.mean(tgt_out))
    if avg_target_snr is not None and avg_target_snr < 0.0:
        high_embs = np.asarray(full_train_data["embeddings"], dtype=np.float32)
        high_labels = np.asarray(full_train_data["labels"], dtype=np.int32)
        high_eff = np.full((len(high_labels),), 10.0, dtype=np.float32)
        high_tgt = np.full((len(high_labels),), 10.0, dtype=np.float32)
        transfer = cross_snr_knowledge_transfer(
            target_embs=x_out,
            target_labels=y_out,
            high_snr_embs=high_embs,
            high_snr_labels=high_labels,
            target_effective_snrs=eff_out,
            target_target_snrs=tgt_out,
            high_snr_effective_snrs=high_eff,
            high_snr_target_snrs=high_tgt,
            transfer_ratio=float(getattr(config, "k2_transfer_ratio", 0.2)),
            transfer_weight=float(getattr(config, "k2_transfer_weight", 0.3)),
            rng=rng,
        )
        x_out = transfer["embeddings"]
        y_out = transfer["labels"]
        eff_out = transfer["effective_snrs"]
        tgt_out = transfer["target_snrs"] if target_snrs is not None else None
        transfer_multiplier = transfer["sample_weight_multiplier"]

    return x_out, y_out, eff_out, tgt_out, transfer_multiplier


def _collect_gate_activation_sample(
    model: tf.keras.Model,
    x_data: np.ndarray,
    max_sample_values: int = 2048,
    max_input_rows: int = 256,
    seed: int = 42,
) -> np.ndarray:
    if x_data is None or len(x_data) == 0:
        return np.zeros((0,), dtype=np.float32)
    rng = np.random.default_rng(seed)
    n_rows = len(x_data)
    take_rows = min(max_input_rows, n_rows)
    row_idx = rng.choice(n_rows, size=take_rows, replace=n_rows < take_rows)
    gate_layer_name = "gate_values"
    try:
        gate_output = model.get_layer(gate_layer_name).output
    except Exception:
        gate_layer_name = "gate_mul"
        gate_output = model.get_layer(gate_layer_name).output
    gate_probe = tf.keras.Model(inputs=model.input, outputs=gate_output)
    gate_values = gate_probe.predict(x_data[row_idx], verbose=0).reshape(-1).astype(np.float32)
    if gate_values.size > max_sample_values:
        pick = rng.choice(gate_values.size, size=max_sample_values, replace=False)
        gate_values = gate_values[pick]
    return gate_values


def _fit_stage(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: tf.Tensor,
    sample_weights: np.ndarray,
    epochs: int,
    batch_size: int,
    stage_name: str,
    show_progress: bool,
    progress_step: int,
):
    alpha_tracker = _AlphaTracker()
    stage_progress = _StageProgressTracker(
        stage_name=stage_name,
        total_epochs=epochs,
        enabled=show_progress,
        percent_step=progress_step,
    )
    history = model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[alpha_tracker, stage_progress],
    )
    losses = [float(v) for v in history.history.get("loss", [])]
    alphas = [float(v) for v in alpha_tracker.values]
    return losses, alphas


def _build_species_class_weights(full_train_data: dict, num_classes: int) -> np.ndarray:
    class_names = [str(c) for c in full_train_data.get("class_names", [])]
    weights = np.ones((num_classes,), dtype=np.float32)
    if not class_names:
        return weights

    lower_lookup = {k.lower(): float(v) for k, v in SPECIES_FOCAL_WEIGHTS.items()}
    for i in range(min(num_classes, len(class_names))):
        class_name = class_names[i]
        weights[i] = float(
            SPECIES_FOCAL_WEIGHTS.get(
                class_name,
                lower_lookup.get(class_name.lower(), 1.0),
            )
        )
    return weights


def _build_focal_loss(config: ExperimentConfig, class_weights: np.ndarray | None = None) -> FocalLossWithLabelSmoothing:
    return FocalLossWithLabelSmoothing(
        alpha=float(getattr(config, "focal_alpha", 1.0)),
        gamma=float(getattr(config, "focal_gamma", 1.5)),
        label_smoothing=float(getattr(config, "focal_label_smoothing", 0.1)),
        num_classes=config.num_classes,
        class_weights=class_weights,
    )


def train_for_dataset(
    x_train,
    y_labels,
    effective_snrs,
    target_snrs,
    pretrained_encoder,
    full_train_data,
    config: ExperimentConfig,
    pretrained_weights=None,
    k: int | None = None,
    return_diagnostics: bool = False,
    diagnostics_seed: int | None = None,
):
    x_train = np.asarray(x_train, dtype=np.float32)
    y_labels = np.asarray(y_labels, dtype=np.int32)
    if effective_snrs is None:
        effective_snrs = np.zeros(len(y_labels), dtype=np.float32)
    else:
        effective_snrs = np.asarray(effective_snrs, dtype=np.float32)

    transfer_multiplier = np.ones((len(y_labels),), dtype=np.float32)
    if k == 2:
        x_train, y_labels, effective_snrs, target_snrs, transfer_multiplier = _apply_k2_augmentation(
            x_train=x_train,
            y_labels=y_labels,
            effective_snrs=effective_snrs,
            target_snrs=target_snrs,
            full_train_data=full_train_data,
            config=config,
        )

    class_counts = {int(c): int(np.sum(full_train_data["labels"] == c)) for c in np.unique(full_train_data["labels"])}
    sample_weights = compute_sample_weight(
        effective_snrs,
        target_snrs,
        y_labels,
        class_counts,
        gamma_snr=config.gamma_snr,
        gamma_class=config.gamma_class,
    )
    if len(transfer_multiplier) == len(sample_weights):
        sample_weights = sample_weights * transfer_multiplier
        sample_weights = sample_weights / (np.mean(sample_weights) + 1e-8)

    encoder = _clone_encoder(pretrained_encoder, pretrained_weights=pretrained_weights)
    model = build_adaptive_refinement_model(
        encoder,
        num_classes=config.num_classes,
        alpha=get_alpha_init(k),
        use_learnable_alpha=True,
        dropout_rate=float(getattr(config, "dropout", 0.4)),
        use_dynamic_alpha=bool(getattr(config, "use_dynamic_alpha", True)),
        alpha_ema_momentum=float(getattr(config, "alpha_ema_momentum", 0.9)),
        gate_temperature_init=float(getattr(config, "gate_temperature_init", 1.5)),
        gate_target_mean=float(getattr(config, "gate_target_mean", 0.45)),
        gate_balance_weight=float(getattr(config, "gate_balance_weight", 1e-3)),
        gate_min_std=float(getattr(config, "gate_min_std", 0.08)),
        gate_std_weight=float(getattr(config, "gate_std_weight", 5e-4)),
    )
    class_focal_weights = _build_species_class_weights(full_train_data, config.num_classes)
    y_train = tf.one_hot(y_labels, depth=config.num_classes)
    batch_size = _batch_size_for_k(k, len(y_labels))
    warmup_epochs = compute_training_epochs(len(y_labels), batch_size, k, config.warmup_epochs, target_steps=100)
    joint_epochs = compute_training_epochs(len(y_labels), batch_size, k, config.joint_epochs, target_steps=300)
    finetune_epochs = compute_training_epochs(len(y_labels), batch_size, k, config.finetune_epochs, target_steps=200)
    diagnostics = {
        "stage_history": [],
        "final_alpha": None,
        "gate_activation_sample": np.zeros((0,), dtype=np.float32),
        "gate_activation_summary": {},
    }
    show_progress = bool(getattr(config, "enable_detailed_progress", True))
    progress_step = int(getattr(config, "progress_percent_step", 5))

    # Stage 1: Warm-up (classifier only)
    for layer in model.layers:
        layer.trainable = False
    model.get_layer("classifier").trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="categorical_crossentropy")
    warmup_losses, warmup_alphas = _fit_stage(
        model=model,
        x_train=x_train,
        y_train=y_train,
        sample_weights=sample_weights,
        epochs=warmup_epochs,
        batch_size=batch_size,
        stage_name="warmup",
        show_progress=show_progress,
        progress_step=progress_step,
    )

    # Stage 2: Joint training (refiner only)
    for layer in model.layers:
        layer.trainable = True
    encoder.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=_build_focal_loss(config, class_weights=class_focal_weights))
    joint_losses, joint_alphas = _fit_stage(
        model=model,
        x_train=x_train,
        y_train=y_train,
        sample_weights=sample_weights,
        epochs=joint_epochs,
        batch_size=batch_size,
        stage_name="joint",
        show_progress=show_progress,
        progress_step=progress_step,
    )

    # Stage 3: Fine-tuning (all layers)
    encoder.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=_build_focal_loss(config, class_weights=class_focal_weights))
    finetune_losses, finetune_alphas = _fit_stage(
        model=model,
        x_train=x_train,
        y_train=y_train,
        sample_weights=sample_weights,
        epochs=finetune_epochs,
        batch_size=batch_size,
        stage_name="finetune",
        show_progress=show_progress,
        progress_step=progress_step,
    )

    if return_diagnostics:
        for epoch_idx, loss_val in enumerate(warmup_losses, start=1):
            diagnostics["stage_history"].append(
                {
                    "stage": "warmup",
                    "epoch": epoch_idx,
                    "loss": float(loss_val),
                    "alpha": float(warmup_alphas[epoch_idx - 1]) if epoch_idx - 1 < len(warmup_alphas) else np.nan,
                }
            )
        for epoch_idx, loss_val in enumerate(joint_losses, start=1):
            diagnostics["stage_history"].append(
                {
                    "stage": "joint",
                    "epoch": epoch_idx,
                    "loss": float(loss_val),
                    "alpha": float(joint_alphas[epoch_idx - 1]) if epoch_idx - 1 < len(joint_alphas) else np.nan,
                }
            )
        for epoch_idx, loss_val in enumerate(finetune_losses, start=1):
            diagnostics["stage_history"].append(
                {
                    "stage": "finetune",
                    "epoch": epoch_idx,
                    "loss": float(loss_val),
                    "alpha": float(finetune_alphas[epoch_idx - 1]) if epoch_idx - 1 < len(finetune_alphas) else np.nan,
                }
            )

        diagnostics["final_alpha"] = _current_alpha(model)
        gate_sample = _collect_gate_activation_sample(
            model=model,
            x_data=x_train,
            max_sample_values=int(getattr(config, "paper_gate_sample_values", 2048)),
            max_input_rows=int(getattr(config, "paper_gate_input_rows", 256)),
            seed=int(diagnostics_seed if diagnostics_seed is not None else config.split_seed),
        )
        diagnostics["gate_activation_sample"] = gate_sample
        if gate_sample.size > 0:
            diagnostics["gate_activation_summary"] = {
                "mean": float(np.mean(gate_sample)),
                "std": float(np.std(gate_sample)),
                "p05": float(np.percentile(gate_sample, 5)),
                "p50": float(np.percentile(gate_sample, 50)),
                "p95": float(np.percentile(gate_sample, 95)),
            }

    if return_diagnostics:
        return model, diagnostics
    return model


def train_for_k_shot_snr(k, target_snr_db, pretrained_encoder, full_train_data, generator, config: ExperimentConfig, seed: int):
    k_shot_indices = generator.sample_k_shot_indices(k, seed=seed)
    soundscape_data = generator.generate_soundscape_embeddings(k_shot_indices, target_snr_db, seed=seed)

    return train_for_dataset(
        soundscape_data["embeddings"],
        soundscape_data["labels"],
        soundscape_data["effective_snrs"],
        soundscape_data.get("target_snrs", target_snr_db),
        pretrained_encoder,
        full_train_data,
        config,
        k=k,
        return_diagnostics=False,
    )
