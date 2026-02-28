from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os


@dataclass
class ExperimentConfig:
    # Dataset root (BirdNET-Analyzer layout)
    dataset_root: str = "D:/BirdNET-Analyzer/birdnet_analyzer/DataBase"

    # Paths
    raw_audio_dir: str = ""
    background_noise_dir: str = ""
    test_audio_dir: str = ""
    val_audio_dir: str = ""
    test_noise_dir: str = ""
    cache_dir: str = "./cache"
    output_dir: str = str(Path(__file__).resolve().parent.parent / "results")

    # Data
    num_classes: int = 7
    sample_rate: int = 48000
    audio_duration: float = 3.0
    embedding_dim: int = 1024
    feature_dim: int = 256
    projection_dim: int = 128

    # Few-shot
    k_values: Tuple[int, ...] = (2,)
    snr_values: Tuple[int, ...] = (-5,)
    train_snr_values: Tuple[int, ...] = (-5, 0, 5, 10)
    num_seeds: int = 2
    split_seed: int = 42
    eval_ratio: float = 0.2

    # Pretrain
    pretrain_steps: int = 2000
    contrastive_temperature: float = 0.07
    batch_size: int = 56
    pretrain_use_mixed_snr: bool = True
    enable_hard_negative_mining: bool = True
    hard_negative_margin: float = 0.5
    hard_negative_weight: float = 0.3
    hard_negative_pairs: Tuple[Tuple[str, str], ...] = (
        ("NorthernRightWhale", "BowheadWhale"),
        ("BottlenoseDolphin", "Frasherdolphin"),
        ("HumpbackWhale", "ClymeneDolphin"),
    )

    # Three-stage training
    warmup_epochs: int = 10
    joint_epochs: int = 30
    finetune_epochs: int = 20

    # Sample weighting
    gamma_snr: float = 0.5
    gamma_class: float = 0.3

    # Noise mixing (aligned with eva2_2_1 defaults)
    train_noise_probability: float = 1.0
    train_noise_max_sources: int = 2
    test_noise_probability: float = 1.0
    test_noise_max_sources: int = 2

    # Soundscape (Scaper) settings aligned with eva2_2_1
    scaper_n_per_class: int = 100
    use_snr_weights: bool = False
    snr_weights: dict = None
    record_effective_snr: bool = True

    bg_mix_prob: float = 0.35
    bg2_rel_db_min: float = -10.0
    bg2_rel_db_max: float = -3.0

    distractor_prob: float = 0.25
    distractor_dur_min: float = 0.5
    distractor_dur_max: float = 1.5
    distractor_snr_offset_min: float = -10.0
    distractor_snr_offset_max: float = -3.0

    partial_prob: float = 0.35
    partial_dur_min: float = 1.0
    partial_dur_max: float = 2.5
    full_event_dur_min: float = 2.8
    full_event_dur_max: float = 3.0

    enable_noise_separation_check: bool = True

    # k=2 augmentation
    mixup_alpha: float = 0.4
    mixup_n_per_pair: int = 3
    k2_transfer_ratio: float = 0.2
    k2_transfer_weight: float = 0.3

    # Adaptive refiner / focal loss
    dropout: float = 0.4
    focal_alpha: float = 1.0
    focal_gamma: float = 1.5
    focal_label_smoothing: float = 0.1
    use_dynamic_alpha: bool = True
    alpha_ema_momentum: float = 0.9
    gate_temperature_init: float = 1.5
    gate_target_mean: float = 0.45
    gate_balance_weight: float = 1e-3
    gate_min_std: float = 0.08
    gate_std_weight: float = 5e-4

    # Auto hyperparameter search before main experiment
    auto_run_hparam_search: bool = False
    hparam_search_method: str = "grid"  # grid or optuna
    hparam_search_trials: int = 50
    hparam_search_grid_limit: int = 0  # 0 means full grid
    hparam_search_k: int = 2
    hparam_search_seed: int = 0
    hparam_search_test_snr: float = -5.0
    hparam_search_force_rerun: bool = True
    hparam_search_output_dir: str = str(Path(__file__).resolve().parent.parent / "results" / "hyperparam_search")

    # Embedding backend
    embedding_backend: str = "birdnet_analyzer"  # birdnet_analyzer only
    birdnet_analyzer_path: str = "D:/BirdNET-Analyzer"
    normalize_embeddings: bool = True
    require_real_embeddings: bool = True

    # Soundscape
    use_scaper: bool = True

    # Paper visualization diagnostics
    paper_gate_sample_values: int = 2048
    paper_gate_input_rows: int = 256
    enable_detailed_progress: bool = True
    progress_percent_step: int = 5

    def __post_init__(self) -> None:
        if not self.raw_audio_dir:
            self.raw_audio_dir = os.path.join(self.dataset_root, "Train")
        if not self.background_noise_dir:
            self.background_noise_dir = os.path.join(self.dataset_root, "train_noise")
        if not self.test_audio_dir:
            self.test_audio_dir = os.path.join(self.dataset_root, "Test")
        if not self.val_audio_dir:
            self.val_audio_dir = os.path.join(self.dataset_root, "Val")
        if not self.test_noise_dir:
            self.test_noise_dir = os.path.join(self.dataset_root, "test_noise")
        if self.snr_weights is None:
            self.snr_weights = {-5.0: 0.40, 0.0: 0.25, 5.0: 0.20, 10.0: 0.15}

    def ensure_dirs(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
