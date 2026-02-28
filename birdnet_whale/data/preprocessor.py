from pathlib import Path
from typing import Dict, List

import numpy as np

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.utils.io import collect_audio_files_by_class
from birdnet_whale.utils.progress import ProgressTracker
from birdnet_whale.data.feature_extractor import EmbeddingExtractor


class DataPreprocessor:
    def __init__(self, audio_dir: str, cache_dir: str, config: ExperimentConfig, extractor: EmbeddingExtractor):
        self.audio_dir = audio_dir
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.extractor = extractor

    def build_embedding_cache(self, cache_path: str) -> None:
        file_map = collect_audio_files_by_class(self.audio_dir)
        class_names = sorted(file_map.keys())
        self.build_embedding_cache_from_file_map(file_map, cache_path, class_names=class_names)

    def build_embedding_cache_from_file_map(self, file_map: Dict[str, List[Path]], cache_path: str, class_names=None) -> None:
        if class_names is None:
            class_names = sorted(file_map.keys())
        class_names = list(class_names)

        embeddings: List[np.ndarray] = []
        labels: List[int] = []
        paths: List[str] = []
        total_files = int(sum(len(file_map.get(class_name, [])) for class_name in class_names))
        show_progress = bool(getattr(self.config, "enable_detailed_progress", True))
        progress_step = int(getattr(self.config, "progress_percent_step", 5))
        tracker = ProgressTracker(
            "Embedding Cache",
            total_files,
            enabled=show_progress and total_files > 0,
            percent_step=progress_step,
        )

        for label_idx, class_name in enumerate(class_names):
            files = file_map.get(class_name, [])
            if show_progress:
                print(
                    f"[Embedding Cache] class {label_idx + 1}/{len(class_names)}: "
                    f"{class_name} ({len(files)} files)"
                )
            for fpath in files:
                emb = self.extractor.extract_embedding_from_audio(str(fpath))
                embeddings.append(emb)
                labels.append(label_idx)
                paths.append(str(fpath))
                tracker.update(extra=f"class={class_name}")

        embeddings_arr = np.array(embeddings, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int32)
        paths_arr = np.array(paths, dtype=object)
        class_arr = np.array(class_names, dtype=object)

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=embeddings_arr, labels=labels_arr, paths=paths_arr, class_names=class_arr)
        tracker.close(extra=f"saved={cache_path}")

    @staticmethod
    def load_cache(cache_path: str) -> Dict[str, np.ndarray]:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "labels": data["labels"],
            "paths": data["paths"],
            "class_names": data["class_names"].tolist(),
        }
