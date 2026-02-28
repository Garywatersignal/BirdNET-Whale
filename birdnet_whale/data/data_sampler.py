from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.data.soundscape import SoundscapeGenerator
from birdnet_whale.data.feature_extractor import EmbeddingExtractor
from birdnet_whale.data.noise_mix import (
    build_noise_mix_config,
    build_noise_mix,
    apply_noise_with_snr,
    compute_effective_snr,
)
from birdnet_whale.utils.progress import ProgressTracker


def _coerce_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str,)):
        return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [str(v) for v in value]


class DynamicTrainingSetGenerator:
    def __init__(
        self,
        cached_train_data: Dict,
        noise_dir: str,
        config: ExperimentConfig,
        extractor: EmbeddingExtractor,
        noise_probability: float | None = None,
        noise_max_sources: int | None = None,
    ):
        self.data = cached_train_data
        self.config = config
        self.extractor = extractor
        self.soundscape = SoundscapeGenerator(noise_dir, config.sample_rate, config.audio_duration, use_scaper=config.use_scaper)
        prob = config.train_noise_probability if noise_probability is None else noise_probability
        max_sources = config.train_noise_max_sources if noise_max_sources is None else noise_max_sources
        self.noise_config = build_noise_mix_config(
            Path(noise_dir),
            list(config.snr_values),
            prob,
            max_sources,
            config.sample_rate,
        )

    def sample_k_shot_indices(self, k: int, seed: int) -> Dict[int, List[int]]:
        rng = np.random.default_rng(seed)
        labels = self.data["labels"]
        unique_classes = np.unique(labels)
        indices_by_class = {}
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            replace = len(cls_indices) < k
            selected = rng.choice(cls_indices, size=k, replace=replace)
            indices_by_class[int(cls)] = selected.tolist()
        return indices_by_class

    def _build_k_shot_path_pool(self, k_shot_indices: Dict[int, List[int]]) -> Dict[int, List[str]]:
        paths = _coerce_paths(self.data.get("paths"))
        pool: Dict[int, List[str]] = {}
        for cls, idx_list in k_shot_indices.items():
            cls_paths = []
            for idx in idx_list:
                if idx < len(paths):
                    cls_paths.append(paths[idx])
            pool[int(cls)] = cls_paths
        return pool

    def build_clean_embeddings(self, k_shot_indices: Dict[int, List[int]]) -> Dict:
        embeddings = self.data["embeddings"]
        labels = self.data["labels"]
        selected_embeddings: List[np.ndarray] = []
        selected_labels: List[int] = []

        for cls, idx_list in k_shot_indices.items():
            for idx in idx_list:
                if idx < len(embeddings):
                    selected_embeddings.append(embeddings[idx])
                    selected_labels.append(int(labels[idx]))

        effective_snrs = np.zeros(len(selected_labels), dtype=np.float32)
        target_snrs = np.zeros(len(selected_labels), dtype=np.float32)
        return {
            "embeddings": np.array(selected_embeddings, dtype=np.float32),
            "labels": np.array(selected_labels, dtype=np.int32),
            "effective_snrs": effective_snrs,
            "target_snrs": target_snrs,
        }

    def generate_soundscape_embeddings(
        self,
        k_shot_indices: Dict[int, List[int]],
        target_snr: float,
        seed: int,
        *,
        n_per_class: Optional[int] = None,
        progress_label: str | None = None,
    ) -> Dict:
        if not self.config.use_scaper:
            raise RuntimeError("Soundscape generation requested but use_scaper is disabled.")

        rng = np.random.default_rng(seed)
        per_class = int(n_per_class or self.config.scaper_n_per_class)
        paths_by_class = self._build_k_shot_path_pool(k_shot_indices)
        if not paths_by_class:
            raise RuntimeError("No foreground paths available for soundscape generation.")

        effective_snrs: List[float] = []
        target_snrs: List[float] = []
        augmented_embeddings: List[np.ndarray] = []
        augmented_labels: List[int] = []

        classes = list(paths_by_class.keys())
        valid_classes = [cls for cls in classes if paths_by_class.get(cls)]
        total_samples = max(0, int(per_class) * len(valid_classes))
        show_progress = bool(getattr(self.config, "enable_detailed_progress", True))
        progress_step = int(getattr(self.config, "progress_percent_step", 5))
        tracker = ProgressTracker(
            progress_label or f"Soundscape SNR={float(target_snr):g}dB",
            total_samples,
            enabled=show_progress and total_samples > 0,
            percent_step=progress_step,
        )
        for cls in classes:
            cls_pool = paths_by_class.get(cls, [])
            if not cls_pool:
                continue
            other_classes = [c for c in classes if c != cls and paths_by_class.get(c)]
            for _ in range(per_class):
                fg_path = str(rng.choice(cls_pool))
                distractor_path = None
                if other_classes:
                    d_cls = int(rng.choice(other_classes))
                    d_pool = paths_by_class.get(d_cls, [])
                    if d_pool:
                        distractor_path = str(rng.choice(d_pool))

                sig, eff_snr = self.soundscape.generate(
                    fg_path=fg_path,
                    target_snr=float(target_snr),
                    seed=int(rng.integers(0, 2**31 - 1)),
                    duration_mode="full_or_partial",
                    partial_prob=self.config.partial_prob,
                    partial_dur_range=(self.config.partial_dur_min, self.config.partial_dur_max),
                    full_dur_range=(self.config.full_event_dur_min, self.config.full_event_dur_max),
                    allow_random_event_time=True,
                    distractor_path=distractor_path,
                    distractor_prob=self.config.distractor_prob,
                    distractor_dur_range=(self.config.distractor_dur_min, self.config.distractor_dur_max),
                    distractor_snr_offset_range=(self.config.distractor_snr_offset_min, self.config.distractor_snr_offset_max),
                    bg2_path=None,
                    bg_mix_prob=self.config.bg_mix_prob,
                    bg2_rel_db_range=(self.config.bg2_rel_db_min, self.config.bg2_rel_db_max),
                    compute_eff_snr=self.config.record_effective_snr,
                )
                emb = self.extractor.extract_embedding_from_signal(sig)
                augmented_embeddings.append(emb)
                augmented_labels.append(int(cls))
                effective_snrs.append(float(eff_snr))
                target_snrs.append(float(target_snr))
                tracker.update(extra=f"class={int(cls)}")

        tracker.close(extra=f"samples={len(augmented_labels)}")

        return {
            "embeddings": np.array(augmented_embeddings, dtype=np.float32),
            "labels": np.array(augmented_labels, dtype=np.int32),
            "effective_snrs": np.array(effective_snrs, dtype=np.float32),
            "target_snrs": np.array(target_snrs, dtype=np.float32),
        }

    def generate_mixed_snr_soundscape_embeddings(
        self,
        k_shot_indices: Dict[int, List[int]],
        snr_values: List[float],
        seed: int,
    ) -> Tuple[Dict, Dict[float, Dict]]:
        snr_values = [float(s) for s in snr_values]
        num_snrs = len(snr_values)
        per_snr_data: Dict[float, Dict] = {}
        if num_snrs == 0:
            empty = {
                "embeddings": np.empty((0, self.config.embedding_dim), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
                "effective_snrs": np.empty((0,), dtype=np.float32),
                "target_snrs": np.empty((0,), dtype=np.float32),
            }
            return empty, per_snr_data

        total_budget = int(self.config.scaper_n_per_class) * num_snrs
        if self.config.use_snr_weights and self.config.snr_weights:
            weights = [float(self.config.snr_weights.get(s, 1.0)) for s in snr_values]
            weight_sum = float(sum(weights)) if sum(weights) > 0 else 0.0
            if weight_sum > 0:
                normalized = [w / weight_sum for w in weights]
            else:
                normalized = [1.0 / num_snrs] * num_snrs
            raw_alloc = [total_budget * w for w in normalized]
            allocations = [max(1, int(round(a))) for a in raw_alloc]
            diff = total_budget - sum(allocations)
            if diff != 0:
                max_idx = int(np.argmax(normalized))
                allocations[max_idx] = max(1, allocations[max_idx] + diff)
        else:
            allocations = [int(self.config.scaper_n_per_class)] * num_snrs

        all_embeddings: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_eff: List[np.ndarray] = []
        all_target: List[np.ndarray] = []
        show_progress = bool(getattr(self.config, "enable_detailed_progress", True))
        progress_step = int(getattr(self.config, "progress_percent_step", 5))
        snr_tracker = ProgressTracker(
            "Mixed-SNR Generation",
            num_snrs,
            enabled=show_progress and num_snrs > 0,
            percent_step=max(1, progress_step),
        )

        for idx, snr in enumerate(snr_values):
            snr_seed = int(seed + idx * 10000)
            n_per_class = allocations[idx]
            if show_progress:
                print(
                    f"[Mixed-SNR Generation] SNR {idx + 1}/{num_snrs}: "
                    f"{float(snr):g}dB, n_per_class={int(n_per_class)}"
                )
            data = self.generate_soundscape_embeddings(
                k_shot_indices,
                target_snr=snr,
                seed=snr_seed,
                n_per_class=n_per_class,
                progress_label=f"Soundscape SNR={float(snr):g}dB",
            )
            per_snr_data[snr] = data
            if data["embeddings"].size > 0:
                all_embeddings.append(data["embeddings"])
                all_labels.append(data["labels"])
                all_eff.append(data["effective_snrs"])
                all_target.append(data["target_snrs"])
            snr_tracker.update(extra=f"snr={float(snr):g}dB samples={len(data['labels'])}")

        snr_tracker.close()
        if not all_embeddings:
            empty = {
                "embeddings": np.empty((0, self.config.embedding_dim), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
                "effective_snrs": np.empty((0,), dtype=np.float32),
                "target_snrs": np.empty((0,), dtype=np.float32),
            }
            return empty, per_snr_data

        combined = {
            "embeddings": np.vstack(all_embeddings),
            "labels": np.concatenate(all_labels),
            "effective_snrs": np.concatenate(all_eff),
            "target_snrs": np.concatenate(all_target),
        }
        return combined, per_snr_data

    def apply_few_shot_augmentation(self, soundscape_data: Dict, k: int) -> Dict:
        if k != 2:
            return soundscape_data

        rng = np.random.default_rng(self.config.split_seed)
        embeddings = soundscape_data["embeddings"].tolist()
        labels = soundscape_data["labels"].tolist()
        snrs = soundscape_data["effective_snrs"].tolist()
        target_snrs = soundscape_data.get("target_snrs")
        if target_snrs is None:
            target_snrs = [0.0 for _ in labels]
        else:
            target_snrs = target_snrs.tolist()

        for cls in np.unique(soundscape_data["labels"]):
            cls_idx = [i for i, lab in enumerate(soundscape_data["labels"]) if lab == cls]
            if len(cls_idx) < 2:
                continue
            e1 = soundscape_data["embeddings"][cls_idx[0]]
            e2 = soundscape_data["embeddings"][cls_idx[1]]
            s1 = float(soundscape_data["effective_snrs"][cls_idx[0]])
            s2 = float(soundscape_data["effective_snrs"][cls_idx[1]])
            t1 = float(target_snrs[cls_idx[0]])
            t2 = float(target_snrs[cls_idx[1]])
            for _ in range(self.config.mixup_n_per_pair):
                lam = rng.beta(self.config.mixup_alpha, self.config.mixup_alpha)
                e_mix = lam * e1 + (1.0 - lam) * e2
                embeddings.append(e_mix.astype(np.float32))
                labels.append(int(cls))
                snrs.append(0.5 * (s1 + s2))
                target_snrs.append(0.5 * (t1 + t2))

        return {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int32),
            "effective_snrs": np.array(snrs, dtype=np.float32),
            "target_snrs": np.array(target_snrs, dtype=np.float32),
        }

    def generate_test_sets(self, test_data: Dict, snr_values, seed: int) -> Dict:
        test_sets = {}
        base = test_data
        paths = _coerce_paths(base.get("paths"))
        noise_files = self.soundscape.noise_files or []
        show_progress = bool(getattr(self.config, "enable_detailed_progress", True))
        progress_step = int(getattr(self.config, "progress_percent_step", 5))
        total_items = int(len(snr_values) * len(base.get("labels", [])))
        tracker = ProgressTracker(
            "Test Set Generation",
            total_items,
            enabled=show_progress and total_items > 0,
            percent_step=progress_step,
        )
        for snr in snr_values:
            if show_progress:
                print(f"[Test Set Generation] building SNR={float(snr):g}dB")
            if noise_files and paths and self.noise_config is not None:
                idx_map = {int(cls): np.where(base["labels"] == cls)[0].tolist() for cls in np.unique(base["labels"])}
                data = {
                    "embeddings": [],
                    "labels": [],
                    "effective_snrs": [],
                }
                for cls, idx_list in idx_map.items():
                    for idx in idx_list:
                        if idx >= len(paths):
                            continue
                        sig = self.extractor.load_audio(paths[idx])
                        noise_mix, has_noise = build_noise_mix(sig, self.config.sample_rate, self.noise_config, np.random.default_rng(seed + idx))
                        if has_noise and noise_mix is not None:
                            mixed, noise_component = apply_noise_with_snr(sig, noise_mix, self.config.sample_rate, snr)
                            eff = compute_effective_snr(sig, noise_component)
                        else:
                            mixed = sig
                            eff = float(snr)
                        emb = self.extractor.extract_embedding_from_signal(mixed)
                        data["embeddings"].append(emb)
                        data["labels"].append(cls)
                        data["effective_snrs"].append(eff)
                        tracker.update(extra=f"snr={float(snr):g}dB")
                test_sets[snr] = {
                    "embeddings": np.array(data["embeddings"], dtype=np.float32),
                    "labels": np.array(data["labels"], dtype=np.int32),
                }
            else:
                test_sets[snr] = {
                    "embeddings": base["embeddings"],
                    "labels": base["labels"],
                }
                tracker.update(len(base["labels"]), extra=f"snr={float(snr):g}dB (reuse cached)")
        tracker.close(extra=f"snr_sets={len(test_sets)}")
        return test_sets
