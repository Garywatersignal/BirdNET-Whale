# Birdnet Whale Few-Shot (Noise Robust)

This project implements a noise-robust few-shot learning pipeline for 7 marine mammal classes.
It follows a three-stage training flow and uses BirdNET embeddings as input features.

## Key Features
- BirdNET embedding extraction and caching
- K-shot sampling with optional soundscape augmentation (Scaper)
- Supervised contrastive pretraining
- Dual-path adaptive refinement classifier
- Noise-aware sample weighting
- Comprehensive evaluation and visualization

## Data Layout
By default the project points to:
```
D:/BirdNET-Analyzer/birdnet_analyzer/DataBase
```
with subfolders:
```
Train/<class_name>/*.wav
Test/<class_name>/*.wav
train_noise/*.wav
test_noise/*.wav
```
If `DataBase/Test` is missing or empty, the code will split `DataBase/Train`
into train/test per class using `eval_ratio` (default 0.2) and `split_seed`.
You can override these paths in `birdnet_whale/config.py`.

## Quick Start
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Ensure BirdNET-Analyzer is available (required):
- Set `BIRDNET_ANALYZER_PATH` environment variable or edit `birdnet_whale/config.py`.
- If BirdNET-Analyzer cannot be imported, the pipeline will raise an error (no fallback).

3. Run the full experiment:
```
python -m birdnet_whale.main
```

4. Optional: run hyperparameter search (k=2, Mixed, test -5dB):
```
python hyperparameter_search.py --method optuna --trials 50
python hyperparameter_search.py --method grid
```

## Notes
- Outputs:
  - Metrics CSVs: `results/all_experiment_results.csv`, `results/results_summary.csv`
  - Plots: `results/performance_summary.png` and `results/plots/*.png`
- The default config targets 3-second, 48 kHz audio clips.
- Scaper is used when enabled and when noise files are available. If Scaper fails, the code falls back to a simple SNR mixer.
- For large experiments, consider reducing `num_seeds`, `k_values`, or `snr_values` in `birdnet_whale/config.py`.
