import math
import numpy as np
import tensorflow as tf

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.models.shared_encoder import build_shared_encoder
from birdnet_whale.models.contrastive_head import build_projection_head
from birdnet_whale.models.losses import SupervisedContrastiveLoss
from birdnet_whale.utils.progress import ProgressTracker

HARD_NEGATIVE_PAIRS = [
    ("NorthernRightWhale", "BowheadWhale"),
    ("BottlenoseDolphin", "Frasherdolphin"),
    ("HumpbackWhale", "ClymeneDolphin"),
]


def cosine_annealing(step, total_steps, max_lr, min_lr):
    if total_steps <= 1:
        return min_lr
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * step / total_steps))


def _build_label_indices(labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(cls): np.where(labels == cls)[0] for cls in np.unique(labels)}


def _build_snr_buckets(indices_by_class: dict[int, np.ndarray], snrs: np.ndarray | None) -> dict[int, dict[float, np.ndarray]]:
    if snrs is None:
        return {}
    snrs = np.asarray(snrs, dtype=np.float32)
    buckets: dict[int, dict[float, np.ndarray]] = {}
    for cls, cls_indices in indices_by_class.items():
        cls_snrs = snrs[cls_indices]
        cls_bucket: dict[float, np.ndarray] = {}
        for snr_value in np.unique(cls_snrs):
            snr_mask = np.isclose(cls_snrs, snr_value)
            cls_bucket[float(snr_value)] = cls_indices[snr_mask]
        if cls_bucket:
            buckets[cls] = cls_bucket
    return buckets


def _sample_indices_for_class(
    cls_indices: np.ndarray,
    per_class_count: int,
    rng: np.random.Generator,
    cls_snr_buckets: dict[float, np.ndarray] | None = None,
) -> list[int]:
    if per_class_count <= 0 or cls_indices.size == 0:
        return []

    if not cls_snr_buckets or len(cls_snr_buckets) <= 1:
        replace = cls_indices.size < per_class_count
        return rng.choice(cls_indices, size=per_class_count, replace=replace).astype(np.int64).tolist()

    selected: list[int] = []
    snr_keys = list(cls_snr_buckets.keys())
    rng.shuffle(snr_keys)

    while len(selected) < per_class_count:
        added = False
        for snr_key in snr_keys:
            bucket = cls_snr_buckets[snr_key]
            if bucket.size == 0:
                continue
            pick = int(rng.choice(bucket, size=1, replace=bucket.size == 1)[0])
            selected.append(pick)
            added = True
            if len(selected) >= per_class_count:
                break
        if not added:
            break

    if len(selected) < per_class_count:
        fallback = rng.choice(cls_indices, size=per_class_count - len(selected), replace=True).astype(np.int64).tolist()
        selected.extend(fallback)
    return selected


def sample_multi_snr_balanced_batch(
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    snrs: np.ndarray | None = None,
    indices_by_class: dict[int, np.ndarray] | None = None,
    snr_buckets: dict[int, dict[float, np.ndarray]] | None = None,
) -> np.ndarray:
    if indices_by_class is None:
        indices_by_class = _build_label_indices(labels)
    classes = np.array(sorted(indices_by_class.keys()), dtype=np.int32)
    if classes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    classes_per_batch = min(classes.size, batch_size)
    chosen_classes = rng.choice(classes, size=classes_per_batch, replace=classes.size < classes_per_batch)
    base = batch_size // classes_per_batch
    remainder = batch_size % classes_per_batch

    if snr_buckets is None:
        snr_buckets = _build_snr_buckets(indices_by_class, snrs)
    batch_indices: list[int] = []
    for i, cls in enumerate(chosen_classes):
        per_class_count = base + (1 if i < remainder else 0)
        class_id = int(np.asarray(cls).item())
        cls_indices = indices_by_class[class_id]
        selected = _sample_indices_for_class(
            cls_indices=cls_indices,
            per_class_count=per_class_count,
            rng=rng,
            cls_snr_buckets=snr_buckets.get(class_id),
        )
        batch_indices.extend(selected)

    if len(batch_indices) > batch_size:
        batch_indices = batch_indices[:batch_size]
    elif len(batch_indices) < batch_size:
        pad = rng.choice(labels.shape[0], size=batch_size - len(batch_indices), replace=True).astype(np.int64).tolist()
        batch_indices.extend(pad)

    rng.shuffle(batch_indices)
    return np.asarray(batch_indices, dtype=np.int64)


def compute_hard_negative_loss(embeddings, labels, class_names, hard_pairs, margin=0.5):
    """
    Apply additional margin constraints for difficult negative species pairs:
    d(class_a, class_b) > margin.
    """
    if class_names is None or not hard_pairs:
        return tf.constant(0.0, dtype=tf.float32)

    labels = tf.cast(tf.reshape(labels, (-1,)), tf.int32)
    class_names = [str(n) for n in class_names]
    total_loss = tf.constant(0.0, dtype=tf.float32)
    active_pairs = tf.constant(0.0, dtype=tf.float32)
    margin_t = tf.constant(float(margin), dtype=tf.float32)

    for sp1, sp2 in hard_pairs:
        if sp1 not in class_names or sp2 not in class_names:
            continue
        idx1 = int(class_names.index(sp1))
        idx2 = int(class_names.index(sp2))

        mask1 = tf.equal(labels, idx1)
        mask2 = tf.equal(labels, idx2)
        has_pair = tf.logical_and(tf.reduce_any(mask1), tf.reduce_any(mask2))

        def _pair_loss():
            emb1 = tf.reduce_mean(tf.boolean_mask(embeddings, mask1), axis=0)
            emb2 = tf.reduce_mean(tf.boolean_mask(embeddings, mask2), axis=0)
            dist = tf.norm(emb1 - emb2)
            return tf.nn.relu(margin_t - dist)

        pair_loss = tf.cond(has_pair, _pair_loss, lambda: tf.constant(0.0, dtype=tf.float32))
        total_loss = total_loss + pair_loss
        active_pairs = active_pairs + tf.cast(has_pair, tf.float32)

    return tf.where(active_pairs > 0.0, total_loss / (active_pairs + 1e-8), tf.constant(0.0, dtype=tf.float32))


def train_contrastive_pretraining(
    train_embeddings,
    train_labels,
    config: ExperimentConfig,
    output_path: str,
    train_snrs=None,
    class_names=None,
    return_history: bool = False,
):
    encoder = build_shared_encoder(input_dim=config.embedding_dim, feature_dim=config.feature_dim)
    model = build_projection_head(encoder, projection_dim=config.projection_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = SupervisedContrastiveLoss(temperature=config.contrastive_temperature)

    rng = np.random.default_rng(config.split_seed)
    total_steps = int(config.pretrain_steps)
    n_samples = train_embeddings.shape[0]
    if train_snrs is not None and len(train_snrs) != n_samples:
        train_snrs = None
    indices_by_class = _build_label_indices(train_labels)
    snr_buckets = _build_snr_buckets(indices_by_class, train_snrs)
    history = {"step": [], "loss": [], "supcon_loss": [], "hard_negative_loss": [], "lr": []}
    show_progress = bool(getattr(config, "enable_detailed_progress", True))
    progress_step = int(getattr(config, "progress_percent_step", 5))
    hard_pairs = getattr(config, "hard_negative_pairs", HARD_NEGATIVE_PAIRS)
    hard_margin = float(getattr(config, "hard_negative_margin", 0.5))
    hard_weight = float(getattr(config, "hard_negative_weight", 0.3))
    use_hard_neg = bool(getattr(config, "enable_hard_negative_mining", True))
    tracker = ProgressTracker(
        "Contrastive Pretrain",
        total_steps,
        enabled=show_progress and total_steps > 0,
        percent_step=progress_step,
    )
    patience_counter = 0

    for step in range(total_steps):
        batch_size = min(config.batch_size, n_samples)
        idx = sample_multi_snr_balanced_batch(
            labels=train_labels,
            batch_size=batch_size,
            rng=rng,
            snrs=train_snrs,
            indices_by_class=indices_by_class,
            snr_buckets=snr_buckets,
        )
        batch_embs = train_embeddings[idx]
        batch_labels = train_labels[idx]

        with tf.GradientTape() as tape:
            _, z = model(batch_embs, training=True)
            supcon_loss = loss_fn(z, batch_labels)
            hard_neg_loss = tf.constant(0.0, dtype=tf.float32)
            if use_hard_neg:
                hard_neg_loss = compute_hard_negative_loss(
                    z,
                    batch_labels,
                    class_names=class_names,
                    hard_pairs=hard_pairs,
                    margin=hard_margin,
                )
            loss = supcon_loss + hard_weight * hard_neg_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        lr = cosine_annealing(step, total_steps, max_lr=0.001, min_lr=1e-5)
        optimizer.learning_rate.assign(lr)
        history["step"].append(int(step))
        history["loss"].append(float(loss.numpy()))
        history["supcon_loss"].append(float(supcon_loss.numpy()))
        history["hard_negative_loss"].append(float(hard_neg_loss.numpy()))
        history["lr"].append(float(lr))
        tracker.update(
            extra=(
                f"loss={float(loss.numpy()):.4f} "
                f"sup={float(supcon_loss.numpy()):.4f} "
                f"hard={float(hard_neg_loss.numpy()):.4f} "
                f"lr={float(lr):.6f}"
            )
        )
        if not show_progress and step % 200 == 0:
            print(f"[Pretrain] Step {step}/{total_steps} - loss={loss.numpy():.4f} - lr={lr:.6f}")

        if step > 200 and float(loss.numpy()) < 0.3:
            patience_counter += 1
            if patience_counter > 50:
                if not show_progress:
                    print(f"[Pretrain] Early stopping at step {step}: loss < 0.3 for {patience_counter} steps")
                break
        else:
            patience_counter = 0

    encoder_only = tf.keras.Model(inputs=model.input, outputs=model.get_layer("features").output)
    encoder_only.save(output_path)
    tracker.close(extra=f"saved={output_path}")
    if return_history:
        return encoder_only, history
    return encoder_only
