from pathlib import Path
from typing import Dict, List

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.utils.seed import set_global_seed
from birdnet_whale.data.feature_extractor import EmbeddingExtractor
from birdnet_whale.data.preprocessor import DataPreprocessor
from birdnet_whale.data.data_sampler import DynamicTrainingSetGenerator
from birdnet_whale.data.splitter import collect_class_files, split_train_eval_files
from birdnet_whale.training.contrastive import train_contrastive_pretraining
from birdnet_whale.training.fewshot import train_for_k_shot_snr
from birdnet_whale.eval.evaluator import evaluate_model


def _limit_files(file_map: Dict[str, List[Path]], per_class: int) -> Dict[str, List[Path]]:
    limited: Dict[str, List[Path]] = {}
    for cls, files in file_map.items():
        if not files:
            continue
        limited[cls] = files[:per_class]
    return limited


def main() -> None:
    config = ExperimentConfig()
    config.cache_dir = "./cache/smoke_test"
    config.output_dir = "./results/smoke_test"
    config.pretrain_steps = 5
    config.batch_size = 4
    config.k_values = (1,)
    config.snr_values = (0,)
    config.num_seeds = 1
    config.warmup_epochs = 1
    config.joint_epochs = 1
    config.finetune_epochs = 1
    config.ensure_dirs()

    set_global_seed(config.split_seed)

    train_root = Path(config.raw_audio_dir)
    if not train_root.exists():
        raise RuntimeError(f"Train directory not found: {train_root}")

    train_files = collect_class_files(train_root)
    if not train_files:
        raise RuntimeError(f"No training audio found under: {train_root}")

    test_root = Path(config.test_audio_dir)
    test_files = collect_class_files(test_root) if test_root.exists() else {}

    if test_files:
        train_map = train_files
        test_map = test_files
    else:
        train_map, test_map = split_train_eval_files(train_files, config.eval_ratio, config.split_seed)

    train_map = _limit_files(train_map, per_class=2)
    test_map = _limit_files(test_map, per_class=1)

    class_names = sorted(train_map.keys())
    if not class_names:
        raise RuntimeError("No classes found after limiting files.")
    config.num_classes = len(class_names)

    extractor = EmbeddingExtractor(config)
    preprocessor = DataPreprocessor(config.raw_audio_dir, config.cache_dir, config, extractor)

    train_cache_path = Path(config.cache_dir) / "train_embeddings.npz"
    test_cache_path = Path(config.cache_dir) / "test_embeddings.npz"
    preprocessor.build_embedding_cache_from_file_map(train_map, str(train_cache_path), class_names=class_names)
    preprocessor.build_embedding_cache_from_file_map(test_map, str(test_cache_path), class_names=class_names)

    train_cache = DataPreprocessor.load_cache(str(train_cache_path))
    test_cache = DataPreprocessor.load_cache(str(test_cache_path))

    if train_cache["embeddings"].size == 0 or test_cache["embeddings"].size == 0:
        raise RuntimeError("Smoke test failed: empty embeddings.")

    train_data = {
        "embeddings": train_cache["embeddings"],
        "labels": train_cache["labels"],
        "paths": train_cache["paths"],
        "class_names": train_cache["class_names"],
    }
    test_data = {
        "embeddings": test_cache["embeddings"],
        "labels": test_cache["labels"],
        "paths": test_cache["paths"],
        "class_names": test_cache["class_names"],
    }

    pretrain_path = Path(config.cache_dir) / "pretrained_encoder.h5"
    pretrained_encoder = train_contrastive_pretraining(
        train_data["embeddings"],
        train_data["labels"],
        config,
        str(pretrain_path),
        class_names=train_data.get("class_names"),
    )

    generator = DynamicTrainingSetGenerator(train_data, config.background_noise_dir, config, extractor)
    if Path(config.test_noise_dir).exists():
        generator_test = DynamicTrainingSetGenerator(test_data, config.test_noise_dir, config, extractor)
    else:
        generator_test = generator

    test_sets_by_snr = generator_test.generate_test_sets(test_data, config.snr_values, seed=config.split_seed)
    snr = config.snr_values[0]

    model = train_for_k_shot_snr(
        k=1,
        target_snr_db=snr,
        pretrained_encoder=pretrained_encoder,
        full_train_data=train_data,
        generator=generator,
        config=config,
        seed=0,
    )

    metrics = evaluate_model(model, test_sets_by_snr[snr]["embeddings"], test_sets_by_snr[snr]["labels"], num_classes=config.num_classes)
    print("Smoke test passed.")
    print(metrics)


if __name__ == "__main__":
    main()
