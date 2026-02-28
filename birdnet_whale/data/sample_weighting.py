import numpy as np


def _as_array(value, n_samples):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(n_samples, float(arr), dtype=np.float32)
    if arr.shape[0] != n_samples:
        if arr.shape[0] == 1:
            return np.full(n_samples, float(arr[0]), dtype=np.float32)
        raise ValueError(f"target_snr shape mismatch: expected {n_samples}, got {arr.shape[0]}")
    return arr


def compute_sample_weight(
    effective_snrs,
    target_snr,
    labels,
    class_counts,
    gamma_snr=0.5,
    gamma_class=0.3,
    clip_range=(0.3, 3.0),
):
    n_samples = len(effective_snrs)
    if n_samples == 0:
        return np.zeros((0,), dtype=np.float32)

    target_arr = _as_array(target_snr, n_samples)
    if target_arr is None:
        w_snr = np.ones(n_samples, dtype=np.float32)
    else:
        effective_arr = np.asarray(effective_snrs, dtype=np.float32)
        snr_diff = np.clip(effective_arr - target_arr, -15.0, 15.0)
        difficulty = np.exp(-snr_diff / 5.0)
        w_snr = difficulty ** gamma_snr
        w_snr = w_snr / (np.mean(w_snr) + 1e-8)

    total_samples = float(sum(class_counts.values()))
    w_class = np.zeros(n_samples, dtype=np.float32)
    for i, label in enumerate(labels):
        count = max(int(class_counts.get(int(label), 1)), 1)
        w_class[i] = total_samples / (len(class_counts) * count)
    w_class = w_class ** gamma_class
    w_class = w_class / (np.mean(w_class) + 1e-8)

    w_final = w_snr * w_class
    w_final = np.clip(w_final, clip_range[0], clip_range[1])
    w_final = w_final / (np.mean(w_final) + 1e-8)
    return w_final.astype(np.float32)
