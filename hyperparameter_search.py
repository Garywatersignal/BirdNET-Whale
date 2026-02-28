from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.data.data_sampler import DynamicTrainingSetGenerator
from birdnet_whale.data.feature_extractor import EmbeddingExtractor
from birdnet_whale.data.preprocessor import DataPreprocessor
from birdnet_whale.data.splitter import collect_class_files, split_train_eval_files
from birdnet_whale.eval.evaluator import evaluate_model
from birdnet_whale.training.contrastive import train_contrastive_pretraining
from birdnet_whale.training.fewshot import train_for_dataset
from birdnet_whale.utils.seed import set_global_seed


GRID_SPACE = {
    "gamma_snr": [0.3, 0.5, 0.7],
    "gamma_class": [0.2, 0.3, 0.5],
    "mixup_alpha": [0.3, 0.4, 0.6],
    "dropout": [0.3, 0.4, 0.5],
    "focal_gamma": [1.0, 1.5, 2.0],
}


def _load_or_build_split_cache(config: ExperimentConfig, extractor: EmbeddingExtractor):
    train_files = collect_class_files(Path(config.raw_audio_dir))
    if not train_files:
        raise RuntimeError(f"No training data found under: {config.raw_audio_dir}")

    class_names = sorted(train_files.keys())
    config.num_classes = len(class_names)
    train_map, test_map = split_train_eval_files(train_files, config.eval_ratio, config.split_seed)

    preprocessor = DataPreprocessor(config.raw_audio_dir, config.cache_dir, config, extractor)
    train_cache_path = Path(config.cache_dir) / "train_embeddings.npz"
    test_cache_path = Path(config.cache_dir) / "test_embeddings.npz"

    if not train_cache_path.exists():
        preprocessor.build_embedding_cache_from_file_map(train_map, str(train_cache_path), class_names=class_names)
    if not test_cache_path.exists():
        preprocessor.build_embedding_cache_from_file_map(test_map, str(test_cache_path), class_names=class_names)

    train_cache = preprocessor.load_cache(str(train_cache_path))
    test_cache = preprocessor.load_cache(str(test_cache_path))

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
    return train_data, test_data


def _load_or_train_pretrained_encoder(config: ExperimentConfig, train_data: dict):
    pretrain_path = Path(config.cache_dir) / "pretrained_encoder.h5"
    if pretrain_path.exists():
        encoder = tf.keras.models.load_model(str(pretrain_path), compile=False)
        return encoder, encoder.get_weights()

    encoder = train_contrastive_pretraining(
        train_data["embeddings"],
        train_data["labels"],
        config,
        str(pretrain_path),
        class_names=train_data.get("class_names"),
    )
    return encoder, encoder.get_weights()


def _select_test_set(test_sets_by_snr: dict, target_snr: float) -> dict:
    for snr_key, data in test_sets_by_snr.items():
        if np.isclose(float(snr_key), float(target_snr)):
            return data
    raise KeyError(f"Test SNR {target_snr} dB was not generated. Available: {list(test_sets_by_snr.keys())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Birdnet-Whale few-shot (fixed objective: k=2, Mixed train, test -5dB by default)."
    )
    parser.add_argument("--method", type=str, choices=("grid", "optuna"), default="optuna", help="Search method.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--grid-limit", type=int, default=0, help="Optional cap for grid candidates (0 = full grid).")
    parser.add_argument("--k", type=int, default=2, help="k-shot value for objective.")
    parser.add_argument("--train-seed", type=int, default=0, help="Sampling seed for objective run.")
    parser.add_argument("--test-snr", type=float, default=-5.0, help="Evaluation SNR (dB) for objective.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--study-name",
        type=str,
        default="birdnet_whale_hyperparam_search",
        help="Optuna study name (used only when --method optuna).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparam_search",
        help="Directory where search artifacts will be written.",
    )
    return parser.parse_args()


class SearchRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed(int(args.seed))

        self.base_config = ExperimentConfig()
        self.base_config.ensure_dirs()
        self.base_config.enable_detailed_progress = False
        self.base_config.progress_percent_step = 10

        extractor = EmbeddingExtractor(self.base_config)
        self.train_data, test_data = _load_or_build_split_cache(self.base_config, extractor)
        self.generator = DynamicTrainingSetGenerator(
            self.train_data,
            self.base_config.background_noise_dir,
            self.base_config,
            extractor,
            noise_probability=self.base_config.train_noise_probability,
            noise_max_sources=self.base_config.train_noise_max_sources,
        )
        if Path(self.base_config.test_noise_dir).exists():
            generator_test = DynamicTrainingSetGenerator(
                test_data,
                self.base_config.test_noise_dir,
                self.base_config,
                extractor,
                noise_probability=self.base_config.test_noise_probability,
                noise_max_sources=self.base_config.test_noise_max_sources,
            )
        else:
            generator_test = self.generator

        test_sets_by_snr = generator_test.generate_test_sets(
            test_data,
            (float(self.args.test_snr),),
            seed=self.base_config.split_seed,
        )
        self.objective_test_set = _select_test_set(test_sets_by_snr, float(self.args.test_snr))

        k_shot_indices = self.generator.sample_k_shot_indices(int(self.args.k), seed=int(self.args.train_seed))
        mixed_data, _ = self.generator.generate_mixed_snr_soundscape_embeddings(
            k_shot_indices,
            list(self.base_config.train_snr_values),
            seed=int(self.args.train_seed),
        )
        if mixed_data["embeddings"].size == 0:
            raise RuntimeError("Generated mixed training set is empty; cannot run hyperparameter search.")
        self.mixed_data = mixed_data
        self.pretrained_encoder, self.pretrained_weights = _load_or_train_pretrained_encoder(
            self.base_config, self.train_data
        )

    def run_single_experiment(self, params: dict[str, float]) -> dict[str, Any]:
        cfg = replace(self.base_config)
        for key, value in params.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"ExperimentConfig has no attribute '{key}'")
            setattr(cfg, key, float(value))

        model = train_for_dataset(
            self.mixed_data["embeddings"],
            self.mixed_data["labels"],
            self.mixed_data.get("effective_snrs"),
            self.mixed_data.get("target_snrs", self.args.test_snr),
            self.pretrained_encoder,
            self.train_data,
            cfg,
            pretrained_weights=self.pretrained_weights,
            k=int(self.args.k),
            return_diagnostics=False,
        )
        metrics = evaluate_model(
            model,
            self.objective_test_set["embeddings"],
            self.objective_test_set["labels"],
            num_classes=cfg.num_classes,
        )
        result = {
            **params,
            "macro_ap": float(metrics["macro_ap"]),
            "macro_auc": float(metrics["macro_auc"]),
            "accuracy": float(metrics["accuracy"]),
        }

        del model
        tf.keras.backend.clear_session()
        return result

    def run_grid_search(self) -> tuple[dict[str, float], float, pd.DataFrame]:
        candidates = list(ParameterGrid(GRID_SPACE))
        if int(self.args.grid_limit) > 0:
            candidates = candidates[: int(self.args.grid_limit)]
        if not candidates:
            raise RuntimeError("No grid candidates to evaluate.")

        rows: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_params: dict[str, float] | None = None

        print(f"[Grid] evaluating {len(candidates)} candidates")
        for idx, params in enumerate(candidates, start=1):
            print(f"[Grid] {idx}/{len(candidates)} params={params}")
            try:
                row = self.run_single_experiment(params)
                row["trial"] = idx
                row["status"] = "ok"
            except Exception as exc:
                row = {**params, "trial": idx, "status": "failed", "error": str(exc), "macro_ap": np.nan}
            rows.append(row)

            score = row.get("macro_ap")
            if row["status"] == "ok" and score is not None and float(score) > best_score:
                best_score = float(score)
                best_params = {k: float(v) for k, v in params.items()}
                print(f"[Grid] new best macro_ap={best_score:.4f} params={best_params}")

        if best_params is None:
            raise RuntimeError("All grid runs failed. No valid best_params found.")
        df = pd.DataFrame(rows)
        return best_params, best_score, df

    def run_optuna_search(self) -> tuple[dict[str, float], float, pd.DataFrame]:
        try:
            import optuna as optuna_lib
        except ImportError:
            raise RuntimeError("Optuna is not installed. Install with: pip install optuna")

        def objective(optuna_trial: Any) -> float:
            params = {
                "gamma_snr": optuna_trial.suggest_float("gamma_snr", 0.3, 0.7),
                "gamma_class": optuna_trial.suggest_float("gamma_class", 0.2, 0.5),
                "mixup_alpha": optuna_trial.suggest_float("mixup_alpha", 0.3, 0.6),
                "dropout": optuna_trial.suggest_float("dropout", 0.3, 0.5),
                "focal_gamma": optuna_trial.suggest_float("focal_gamma", 1.0, 2.0),
            }
            result = self.run_single_experiment(params)
            optuna_trial.set_user_attr("macro_auc", result["macro_auc"])
            optuna_trial.set_user_attr("accuracy", result["accuracy"])
            return float(result["macro_ap"])

        db_path = (self.output_dir / "optuna_study.db").as_posix()
        study = optuna_lib.create_study(
            direction="maximize",
            study_name=str(self.args.study_name),
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            sampler=optuna_lib.samplers.TPESampler(seed=int(self.args.seed)),
        )
        study.optimize(objective, n_trials=int(self.args.trials), show_progress_bar=True, gc_after_trial=True)

        rows: list[dict[str, Any]] = []
        for completed_trial in study.trials:
            row = {"trial": int(completed_trial.number), "status": str(completed_trial.state).split(".")[-1].lower()}
            row.update({k: completed_trial.params.get(k, np.nan) for k in GRID_SPACE.keys()})
            row["macro_ap"] = float(completed_trial.value) if completed_trial.value is not None else np.nan
            row["macro_auc"] = float(completed_trial.user_attrs.get("macro_auc", np.nan))
            row["accuracy"] = float(completed_trial.user_attrs.get("accuracy", np.nan))
            rows.append(row)
        df = pd.DataFrame(rows)
        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_score = float(study.best_value)
        return best_params, best_score, df

    def save_outputs(self, method: str, best_params: dict[str, float], best_score: float, df: pd.DataFrame) -> None:
        best_payload = {
            "method": method,
            "best_value_macro_ap": float(best_score),
            "best_params": best_params,
            "k": int(self.args.k),
            "test_snr": float(self.args.test_snr),
            "train_seed": int(self.args.train_seed),
        }
        best_path = self.output_dir / "best_params.json"
        best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

        csv_method_path = self.output_dir / f"{method}_search_results.csv"
        csv_common_path = self.output_dir / "hyperparameter_search_results.csv"
        df.to_csv(csv_method_path, index=False)
        df.to_csv(csv_common_path, index=False)

        print("=" * 72)
        print(f"Search method: {method}")
        print(f"Best macro_ap: {best_score:.4f}")
        print(f"Best params : {best_params}")
        print(f"Saved: {best_path}")
        print(f"Saved: {csv_method_path}")
        print(f"Saved: {csv_common_path}")
        print("=" * 72)


def main() -> None:
    args = parse_args()
    runner = SearchRunner(args)

    if args.method == "grid":
        best_params, best_score, df = runner.run_grid_search()
    elif args.method == "optuna":
        best_params, best_score, df = runner.run_optuna_search()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    runner.save_outputs(args.method, best_params, best_score, df)


if __name__ == "__main__":
    main()
