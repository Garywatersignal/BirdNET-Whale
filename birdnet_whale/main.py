from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from birdnet_whale.config import ExperimentConfig
from birdnet_whale.utils.seed import set_global_seed
from birdnet_whale.data.feature_extractor import EmbeddingExtractor
from birdnet_whale.data.preprocessor import DataPreprocessor
from birdnet_whale.data.data_sampler import DynamicTrainingSetGenerator
from birdnet_whale.data.splitter import collect_class_files, split_train_eval_files
from birdnet_whale.training.contrastive import train_contrastive_pretraining
from birdnet_whale.training.fewshot import train_for_dataset
from birdnet_whale.eval.evaluator import evaluate_model
from birdnet_whale.eval.visualizer import plot_performance_vs_k_and_snr
from birdnet_whale.eval.paper_visualizer import generate_paper_visualizations
from birdnet_whale.utils.io import AUDIO_EXTS
from birdnet_whale.utils.progress import ProgressTracker


def _run_hyperparameter_search_and_apply(config: ExperimentConfig) -> None:
    if not bool(getattr(config, "auto_run_hparam_search", False)):
        return

    project_root = Path(__file__).resolve().parent.parent
    search_script = project_root / "hyperparameter_search.py"
    if not search_script.exists():
        print(f"Step 0/6: Hyperparameter search script not found, skip: {search_script}")
        return

    method = str(getattr(config, "hparam_search_method", "grid")).strip().lower()
    if method not in {"grid", "optuna"}:
        raise ValueError(f"Invalid hparam_search_method: {method}. Expected 'grid' or 'optuna'.")

    output_dir = Path(getattr(config, "hparam_search_output_dir", str(project_root / "results" / "hyperparam_search"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_params.json"
    force_rerun = bool(getattr(config, "hparam_search_force_rerun", True))

    if force_rerun or not best_path.exists():
        cmd = [
            sys.executable,
            str(search_script),
            "--method",
            method,
            "--k",
            str(int(getattr(config, "hparam_search_k", 2))),
            "--train-seed",
            str(int(getattr(config, "hparam_search_seed", 0))),
            "--test-snr",
            str(float(getattr(config, "hparam_search_test_snr", -5.0))),
            "--output-dir",
            str(output_dir),
        ]
        if method == "optuna":
            cmd.extend(["--trials", str(int(getattr(config, "hparam_search_trials", 50)))])
        else:
            grid_limit = int(getattr(config, "hparam_search_grid_limit", 0))
            if grid_limit > 0:
                cmd.extend(["--grid-limit", str(grid_limit)])
        print(f"Step 0/6: Running hyperparameter search ({method})...")
        subprocess.run(cmd, check=True, cwd=str(project_root))
    else:
        print(f"Step 0/6: Reusing cached hyperparameter search: {best_path}")

    if not best_path.exists():
        raise RuntimeError(f"Hyperparameter search did not produce best_params.json: {best_path}")

    payload = json.loads(best_path.read_text(encoding="utf-8"))
    best_params = payload.get("best_params", {})
    applied = {}
    for key in ("gamma_snr", "gamma_class", "mixup_alpha", "dropout", "focal_gamma"):
        if key in best_params:
            value = float(best_params[key])
            setattr(config, key, value)
            applied[key] = value

    if applied:
        print(f"Step 0/6: Applied best hyperparameters: {applied}")
    else:
        print("Step 0/6: No compatible hyperparameters found in best_params.json.")


def main():
    config = ExperimentConfig()
    config.ensure_dirs()
    _run_hyperparameter_search_and_apply(config)
    base_output_dir = Path(config.output_dir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_output_dir / f"run_{run_stamp}"
    duplicate_idx = 1
    while run_output_dir.exists():
        duplicate_idx += 1
        run_output_dir = base_output_dir / f"run_{run_stamp}_{duplicate_idx:02d}"
    run_output_dir.mkdir(parents=True, exist_ok=False)
    config.output_dir = str(run_output_dir)
    print(f"Run output directory: {config.output_dir}")
    show_progress = bool(getattr(config, "enable_detailed_progress", True))
    progress_step = int(getattr(config, "progress_percent_step", 5))
    pipeline_tracker = ProgressTracker(
        "Pipeline",
        6,
        enabled=show_progress,
        percent_step=1,
    )

    set_global_seed(config.split_seed)

    extractor = EmbeddingExtractor(config)
    preprocessor = DataPreprocessor(config.raw_audio_dir, config.cache_dir, config, extractor)

    train_files = collect_class_files(Path(config.raw_audio_dir))
    if not train_files:
        raise RuntimeError(f"No training data found under: {config.raw_audio_dir}")
    print("Step 1/6: Splitting Train into train/test (eva2_2_1.py logic).")
    train_map, test_map = split_train_eval_files(train_files, config.eval_ratio, config.split_seed)

    class_names = sorted(train_files.keys())

    train_cache_path = Path(config.cache_dir) / "train_embeddings.npz"
    if not train_cache_path.exists():
        print("Step 1a/6: Building train embedding cache...")
        preprocessor.build_embedding_cache_from_file_map(train_map, str(train_cache_path), class_names=class_names)
    else:
        print("Step 1a/6: Train cache found, skipping embedding extraction.")

    test_cache_path = Path(config.cache_dir) / "test_embeddings.npz"
    if not test_cache_path.exists():
        print("Step 1b/6: Building test embedding cache...")
        preprocessor.build_embedding_cache_from_file_map(test_map, str(test_cache_path), class_names=class_names)
    else:
        print("Step 1b/6: Test cache found, skipping embedding extraction.")

    train_cache = preprocessor.load_cache(str(train_cache_path))
    test_cache = preprocessor.load_cache(str(test_cache_path))

    print(f"Loaded {len(train_cache['embeddings'])} train samples across {len(train_cache['class_names'])} classes.")
    print(f"Loaded {len(test_cache['embeddings'])} test samples.")
    pipeline_tracker.update(extra="Step 1 complete")

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
    class_names = [str(c) for c in train_data.get("class_names", [])]
    pretrain_history_df = pd.DataFrame(columns=["step", "loss", "lr"])

    if config.enable_noise_separation_check and Path(config.test_noise_dir).exists():
        train_noise_files = sorted([p for p in Path(config.background_noise_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTS])
        test_noise_files = sorted([p for p in Path(config.test_noise_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTS])
        train_paths = {p.resolve() for p in train_noise_files}
        test_paths = {p.resolve() for p in test_noise_files}
        overlap = train_paths & test_paths
        if overlap:
            examples = [p.name for p in list(overlap)[:5]]
            raise RuntimeError(
                f"Train/test noise sets have {len(overlap)} overlapping files. "
                f"Examples: {examples}. Please separate train_noise and test_noise."
            )

    generator = DynamicTrainingSetGenerator(
        train_data,
        config.background_noise_dir,
        config,
        extractor,
        noise_probability=config.train_noise_probability,
        noise_max_sources=config.train_noise_max_sources,
    )

    pretrain_path = Path(config.cache_dir) / "pretrained_encoder.h5"
    pretrain_history_path = Path(config.cache_dir) / "pretrain_history.csv"
    if not pretrain_path.exists():
        print("Step 2/6: Contrastive pretraining...")
        pretrain_embeddings = train_data["embeddings"]
        pretrain_labels = train_data["labels"]
        pretrain_snrs = np.zeros(len(pretrain_labels), dtype=np.float32)

        if config.pretrain_use_mixed_snr and config.use_scaper:
            try:
                all_indices = {int(cls): np.where(train_data["labels"] == cls)[0].tolist() for cls in np.unique(train_data["labels"])}
                pretrain_mix_data, _ = generator.generate_mixed_snr_soundscape_embeddings(
                    all_indices,
                    list(config.train_snr_values),
                    seed=config.split_seed + 12345,
                )
                if pretrain_mix_data["embeddings"].size > 0:
                    pretrain_embeddings = np.vstack([pretrain_embeddings, pretrain_mix_data["embeddings"]])
                    pretrain_labels = np.concatenate([pretrain_labels, pretrain_mix_data["labels"]])
                    pretrain_snr_values = pretrain_mix_data.get("target_snrs")
                    if pretrain_snr_values is None or len(pretrain_snr_values) != len(pretrain_mix_data["labels"]):
                        pretrain_snr_values = np.zeros(len(pretrain_mix_data["labels"]), dtype=np.float32)
                    pretrain_snrs = np.concatenate([pretrain_snrs, pretrain_snr_values.astype(np.float32)])
                    print(f"  Added mixed-SNR pretrain samples: {len(pretrain_mix_data['labels'])}")
            except Exception as exc:
                print(f"  Skip mixed-SNR pretrain augmentation: {exc}")

        pretrained_encoder, pretrain_history = train_contrastive_pretraining(
            pretrain_embeddings,
            pretrain_labels,
            config,
            str(pretrain_path),
            train_snrs=pretrain_snrs,
            class_names=train_data.get("class_names"),
            return_history=True,
        )
        pretrain_history_df = pd.DataFrame(pretrain_history)
        pretrain_history_df.to_csv(pretrain_history_path, index=False)
    else:
        print("Step 2/6: Pretrained encoder found, loading.")
        pretrained_encoder = tf.keras.models.load_model(str(pretrain_path), compile=False)
        if pretrain_history_path.exists():
            pretrain_history_df = pd.read_csv(pretrain_history_path)
    pretrained_weights = pretrained_encoder.get_weights()
    pipeline_tracker.update(extra="Step 2 complete")

    print("Step 3/6: Preparing test sets by SNR...")
    if Path(config.test_noise_dir).exists():
        generator_test = DynamicTrainingSetGenerator(
            test_data,
            config.test_noise_dir,
            config,
            extractor,
            noise_probability=config.test_noise_probability,
            noise_max_sources=config.test_noise_max_sources,
        )
    else:
        generator_test = generator
    test_sets_by_snr = generator_test.generate_test_sets(test_data, config.snr_values, seed=config.split_seed)
    pipeline_tracker.update(extra="Step 3 complete")

    print("Step 4/6: Few-shot experiment loop...")
    all_experiment_results = []
    per_class_records = []
    training_history_records = []
    gate_activation_records = []
    conditions_per_task = 3 if len(config.train_snr_values) > 0 else 2
    total_fewshot_tasks = int(config.num_seeds) * len(config.k_values) * conditions_per_task
    fewshot_tracker = ProgressTracker(
        "Few-shot Tasks",
        total_fewshot_tasks,
        enabled=show_progress and total_fewshot_tasks > 0,
        percent_step=progress_step,
    )

    for seed in range(config.num_seeds):
        print(f"  Seed {seed + 1}/{config.num_seeds}")
        for k in config.k_values:
            k_shot_indices = generator.sample_k_shot_indices(k, seed=seed)

            clean_data = generator.build_clean_embeddings(k_shot_indices)
            mixed_data, per_snr_data = generator.generate_mixed_snr_soundscape_embeddings(
                k_shot_indices, list(config.train_snr_values), seed=seed
            )
            mid_snr = float(config.train_snr_values[len(config.train_snr_values) // 2]) if config.train_snr_values else 0.0
            single_data = per_snr_data.get(mid_snr)
            if single_data is None or single_data.get("embeddings", np.array([])).size == 0:
                single_data = None
                for snr in config.train_snr_values:
                    cand = per_snr_data.get(float(snr))
                    if cand is not None and cand.get("embeddings", np.array([])).size > 0:
                        single_data = cand
                        mid_snr = float(snr)
                        break

            train_sets = {
                "C": (clean_data, None),
                "M": (mixed_data, None),
            }
            if single_data is not None:
                train_sets["S"] = (single_data, mid_snr)
            elif conditions_per_task == 3:
                fewshot_tracker.update(extra=f"skip seed={seed} k={k} cond=S(no data)")

            for condition_key, (train_data_pack, train_snr_val) in train_sets.items():
                if train_data_pack["embeddings"].size == 0:
                    fewshot_tracker.update(extra=f"skip seed={seed} k={k} cond={condition_key}")
                    continue
                desc = "clean" if condition_key == "C" else ("mixed-SNR" if condition_key == "M" else f"single-SNR({train_snr_val:g}dB)")
                print(f"    Training: k={k}, condition={condition_key} [{desc}]")
                target_snrs = train_data_pack.get("target_snrs")
                if condition_key == "C":
                    target_snrs = None

                model, train_diag = train_for_dataset(
                    train_data_pack["embeddings"],
                    train_data_pack["labels"],
                    train_data_pack.get("effective_snrs"),
                    target_snrs,
                    pretrained_encoder,
                    train_data,
                    config,
                    pretrained_weights=pretrained_weights,
                    k=k,
                    return_diagnostics=True,
                    diagnostics_seed=int(seed * 100000 + k * 100 + ord(condition_key[0])),
                )
                for row in train_diag.get("stage_history", []):
                    training_history_records.append(
                        {
                            "seed": seed,
                            "k": k,
                            "train_condition": condition_key,
                            "train_snr": train_snr_val,
                            "stage": row.get("stage"),
                            "epoch": int(row.get("epoch", 0)),
                            "loss": float(row.get("loss", np.nan)),
                            "alpha": float(row.get("alpha", np.nan)),
                        }
                    )
                gate_sample = np.asarray(train_diag.get("gate_activation_sample", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
                for val in gate_sample:
                    gate_activation_records.append(
                        {
                            "seed": seed,
                            "k": k,
                            "train_condition": condition_key,
                            "train_snr": train_snr_val,
                            "gate_activation": float(val),
                            "final_alpha": float(train_diag.get("final_alpha", np.nan)),
                        }
                    )

                for eval_idx, (test_snr, test_set) in enumerate(test_sets_by_snr.items(), start=1):
                    if show_progress:
                        print(
                            f"      Eval {eval_idx}/{len(test_sets_by_snr)}: "
                            f"test_snr={float(test_snr):g}dB"
                        )
                    metrics = evaluate_model(model, test_set["embeddings"], test_set["labels"], num_classes=config.num_classes)
                    print(
                        "        Metrics: "
                        f"macro_ap={float(metrics['macro_ap']):.4f}, "
                        f"macro_auc={float(metrics['macro_auc']):.4f}, "
                        f"accuracy={float(metrics['accuracy']):.4f}"
                    )
                    all_experiment_results.append({
                        "seed": seed,
                        "k": k,
                        "train_condition": condition_key,
                        "train_snr": train_snr_val,
                        "test_snr": test_snr,
                        "macro_ap": metrics["macro_ap"],
                        "macro_auc": metrics["macro_auc"],
                        "accuracy": metrics["accuracy"],
                    })
                    cm = metrics.get("confusion_matrix")
                    per_ap = metrics.get("per_class_ap", [])
                    per_auc = metrics.get("per_class_auc", [])
                    for class_idx in range(config.num_classes):
                        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                        ap_val = float(per_ap[class_idx]) if class_idx < len(per_ap) else np.nan
                        auc_val = float(per_auc[class_idx]) if class_idx < len(per_auc) else np.nan
                        tp = float(cm[class_idx, class_idx]) if cm is not None else 0.0
                        fp = float(np.sum(cm[:, class_idx]) - tp) if cm is not None else 0.0
                        fn = float(np.sum(cm[class_idx, :]) - tp) if cm is not None else 0.0
                        precision = tp / (tp + fp + 1e-12)
                        recall = tp / (tp + fn + 1e-12)
                        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
                        per_class_records.append(
                            {
                                "seed": seed,
                                "k": k,
                                "train_condition": condition_key,
                                "train_snr": train_snr_val,
                                "test_snr": test_snr,
                                "class_idx": class_idx,
                                "class_name": class_name,
                                "ap": ap_val,
                                "auc": auc_val,
                                "precision": float(precision),
                                "recall": float(recall),
                                "f1": float(f1),
                            }
                        )
                fewshot_tracker.update(extra=f"seed={seed} k={k} cond={condition_key}")

    fewshot_tracker.close(extra=f"tasks={total_fewshot_tasks}")
    pipeline_tracker.update(extra="Step 4 complete")
    print("Step 5/6: Aggregating results...")
    results_df = pd.DataFrame(all_experiment_results)
    results_csv_path = Path(config.output_dir) / "all_experiment_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    group_cols = ["k", "test_snr"]
    if "train_condition" in results_df.columns:
        group_cols.insert(1, "train_condition")
    else:
        group_cols.insert(1, "train_snr")
    summary_df = results_df.groupby(group_cols).agg({
        "macro_ap": ["mean", "std"],
        "macro_auc": ["mean", "std"],
        "accuracy": ["mean", "std"],
    }).round(4)
    summary_csv_path = Path(config.output_dir) / "results_summary.csv"
    summary_df.to_csv(summary_csv_path)
    pipeline_tracker.update(extra="Step 5 complete")

    print("Step 6/6: Plotting...")
    plot_path = Path(config.output_dir) / "performance_summary.png"
    plot_performance_vs_k_and_snr(results_df, str(plot_path))

    per_class_df = pd.DataFrame(per_class_records)
    if not per_class_df.empty:
        per_class_csv = Path(config.output_dir) / "per_class_metrics.csv"
        per_class_df.to_csv(per_class_csv, index=False)
        per_class_summary_df = (
            per_class_df.groupby(["k", "train_condition", "test_snr", "class_idx", "class_name"], as_index=False)
            .agg(
                ap_mean=("ap", "mean"),
                ap_std=("ap", "std"),
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
            )
            .round(4)
            .fillna(0.0)
        )
        per_class_summary_csv = Path(config.output_dir) / "per_class_summary.csv"
        per_class_summary_df.to_csv(per_class_summary_csv, index=False)

    training_history_df = pd.DataFrame(training_history_records)
    if not training_history_df.empty:
        training_history_csv = Path(config.output_dir) / "training_history.csv"
        training_history_df.to_csv(training_history_csv, index=False)

    gate_samples_df = pd.DataFrame(gate_activation_records)
    if not gate_samples_df.empty:
        gate_samples_csv = Path(config.output_dir) / "gate_activation_samples.csv"
        gate_samples_df.to_csv(gate_samples_csv, index=False)

    generate_paper_visualizations(
        results_df=results_df,
        per_class_df=per_class_df,
        training_history_df=training_history_df,
        gate_samples_df=gate_samples_df,
        pretrain_history_df=pretrain_history_df,
        class_names=class_names,
        output_dir=config.output_dir,
    )
    pipeline_tracker.update(extra="Step 6 complete")
    pipeline_tracker.close()

    print("All experiments complete.")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
