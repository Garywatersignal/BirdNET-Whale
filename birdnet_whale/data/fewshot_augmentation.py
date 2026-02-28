import numpy as np


def embedding_mixup(
    embeddings,
    labels,
    *,
    effective_snrs=None,
    target_snrs=None,
    n_mixup: int = 3,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
    max_pairs_per_class: int = 64,
):
    if rng is None:
        rng = np.random.default_rng()

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n_samples = labels.shape[0]

    if n_samples == 0:
        return {
            "embeddings": embeddings,
            "labels": labels,
            "effective_snrs": None if effective_snrs is None else np.asarray(effective_snrs, dtype=np.float32),
            "target_snrs": None if target_snrs is None else np.asarray(target_snrs, dtype=np.float32),
        }

    effective_arr = None if effective_snrs is None else np.asarray(effective_snrs, dtype=np.float32)
    target_arr = None if target_snrs is None else np.asarray(target_snrs, dtype=np.float32)

    emb_list = [embeddings]
    label_list = [labels]
    eff_list = [] if effective_arr is not None else None
    tgt_list = [] if target_arr is not None else None
    if eff_list is not None:
        eff_list.append(effective_arr)
    if tgt_list is not None:
        tgt_list.append(target_arr)

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size < 2:
            continue

        all_pairs = []
        for i in range(cls_idx.size):
            for j in range(i + 1, cls_idx.size):
                all_pairs.append((int(cls_idx[i]), int(cls_idx[j])))

        if not all_pairs:
            continue
        if len(all_pairs) > max_pairs_per_class:
            chosen = rng.choice(len(all_pairs), size=max_pairs_per_class, replace=False)
            pairs = [all_pairs[int(i)] for i in chosen]
        else:
            pairs = all_pairs

        mixed_embs = []
        mixed_eff = [] if eff_list is not None else None
        mixed_tgt = [] if tgt_list is not None else None
        for left, right in pairs:
            for _ in range(max(1, int(n_mixup))):
                lam = float(rng.beta(alpha, alpha))
                mixed = lam * embeddings[left] + (1.0 - lam) * embeddings[right]
                mixed_embs.append(mixed.astype(np.float32))
                if mixed_eff is not None:
                    mixed_eff.append(np.float32(lam * effective_arr[left] + (1.0 - lam) * effective_arr[right]))
                if mixed_tgt is not None:
                    mixed_tgt.append(np.float32(lam * target_arr[left] + (1.0 - lam) * target_arr[right]))

        if mixed_embs:
            emb_list.append(np.asarray(mixed_embs, dtype=np.float32))
            label_list.append(np.full((len(mixed_embs),), int(cls), dtype=np.int32))
            if mixed_eff is not None:
                eff_list.append(np.asarray(mixed_eff, dtype=np.float32))
            if mixed_tgt is not None:
                tgt_list.append(np.asarray(mixed_tgt, dtype=np.float32))

    out = {
        "embeddings": np.vstack(emb_list),
        "labels": np.concatenate(label_list),
        "effective_snrs": None if eff_list is None else np.concatenate(eff_list),
        "target_snrs": None if tgt_list is None else np.concatenate(tgt_list),
    }
    return out


def cross_snr_knowledge_transfer(
    target_embs,
    target_labels,
    high_snr_embs,
    high_snr_labels,
    *,
    target_effective_snrs=None,
    target_target_snrs=None,
    high_snr_effective_snrs=None,
    high_snr_target_snrs=None,
    transfer_ratio: float = 0.3,
    transfer_weight: float = 0.3,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    target_embs = np.asarray(target_embs, dtype=np.float32)
    target_labels = np.asarray(target_labels, dtype=np.int32)
    high_snr_embs = np.asarray(high_snr_embs, dtype=np.float32)
    high_snr_labels = np.asarray(high_snr_labels, dtype=np.int32)

    n_target = target_embs.shape[0]
    if n_target == 0 or high_snr_embs.shape[0] == 0:
        return {
            "embeddings": target_embs,
            "labels": target_labels,
            "effective_snrs": None if target_effective_snrs is None else np.asarray(target_effective_snrs, dtype=np.float32),
            "target_snrs": None if target_target_snrs is None else np.asarray(target_target_snrs, dtype=np.float32),
            "sample_weight_multiplier": np.ones((n_target,), dtype=np.float32),
        }

    n_transfer = int(round(n_target * float(transfer_ratio)))
    n_transfer = max(0, min(n_transfer, high_snr_embs.shape[0]))
    if n_transfer == 0:
        return {
            "embeddings": target_embs,
            "labels": target_labels,
            "effective_snrs": None if target_effective_snrs is None else np.asarray(target_effective_snrs, dtype=np.float32),
            "target_snrs": None if target_target_snrs is None else np.asarray(target_target_snrs, dtype=np.float32),
            "sample_weight_multiplier": np.ones((n_target,), dtype=np.float32),
        }

    transfer_idx = rng.choice(high_snr_embs.shape[0], size=n_transfer, replace=False)
    transfer_embs = high_snr_embs[transfer_idx]
    transfer_labels = high_snr_labels[transfer_idx]

    combined_embs = np.vstack([target_embs, transfer_embs]).astype(np.float32)
    combined_labels = np.concatenate([target_labels, transfer_labels]).astype(np.int32)

    target_eff = None if target_effective_snrs is None else np.asarray(target_effective_snrs, dtype=np.float32)
    high_eff = None if high_snr_effective_snrs is None else np.asarray(high_snr_effective_snrs, dtype=np.float32)
    if target_eff is None and high_eff is not None:
        target_eff = np.zeros((n_target,), dtype=np.float32)
    if target_eff is not None and high_eff is None:
        high_eff = np.zeros((high_snr_embs.shape[0],), dtype=np.float32)
    if target_eff is not None and high_eff is not None:
        combined_eff = np.concatenate([target_eff, high_eff[transfer_idx]]).astype(np.float32)
    else:
        combined_eff = None

    target_tgt = None if target_target_snrs is None else np.asarray(target_target_snrs, dtype=np.float32)
    high_tgt = None if high_snr_target_snrs is None else np.asarray(high_snr_target_snrs, dtype=np.float32)
    if target_tgt is None and high_tgt is not None:
        target_tgt = np.zeros((n_target,), dtype=np.float32)
    if target_tgt is not None and high_tgt is None:
        high_tgt = np.zeros((high_snr_embs.shape[0],), dtype=np.float32)
    if target_tgt is not None and high_tgt is not None:
        combined_tgt = np.concatenate([target_tgt, high_tgt[transfer_idx]]).astype(np.float32)
    else:
        combined_tgt = None

    sample_weight_multiplier = np.concatenate(
        [
            np.ones((n_target,), dtype=np.float32),
            np.full((n_transfer,), float(transfer_weight), dtype=np.float32),
        ]
    ).astype(np.float32)

    return {
        "embeddings": combined_embs,
        "labels": combined_labels,
        "effective_snrs": combined_eff,
        "target_snrs": combined_tgt,
        "sample_weight_multiplier": sample_weight_multiplier,
    }
