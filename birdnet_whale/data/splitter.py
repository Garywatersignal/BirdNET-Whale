from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def collect_files(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES]


def collect_class_files(root: Path) -> Dict[str, List[Path]]:
    """Collect class-organized audio files, skipping folders that contain 'noise' in name."""
    class_to_files: Dict[str, List[Path]] = {}
    if not root.exists():
        return class_to_files
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if "noise" in name.lower():
            continue
        files = collect_files(sub)
        if files:
            class_to_files[name] = files
    return class_to_files


def split_train_eval_files(
    class_files: Dict[str, List[Path]],
    eval_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Split each class into train/eval subsets with a fixed ratio."""
    eval_ratio = float(np.clip(eval_ratio, 0.01, 0.5))
    rng = np.random.default_rng(seed)
    train_map: Dict[str, List[Path]] = {}
    eval_map: Dict[str, List[Path]] = {}

    for cls, files in class_files.items():
        if not files:
            train_map[cls] = []
            eval_map[cls] = []
            continue

        arr = np.array(files, dtype=object)
        rng.shuffle(arr)

        if len(arr) == 1:
            print(f"[WARN] Class {cls} has only 1 sample; using it for training.")
            train_map[cls] = arr.tolist()
            eval_map[cls] = []
            continue

        split_idx = int(round(len(arr) * (1 - eval_ratio)))
        split_idx = max(1, min(split_idx, len(arr) - 1))
        train_map[cls] = arr[:split_idx].tolist()
        eval_map[cls] = arr[split_idx:].tolist()

    return train_map, eval_map
