from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None


_LUFS_METERS: Dict[int, "pyln.Meter"] = {}
_LUFS_WARNED = False


@dataclass
class NoiseMixConfig:
    probability: float
    snr_values: List[float]
    max_sources: int
    noise_files: List[Path]
    noise_durations: Dict[str, float]

    def is_enabled(self) -> bool:
        return (
            bool(self.noise_files)
            and bool(self.snr_values)
            and self.probability > 0.0
            and self.max_sources > 0
        )


def _get_lufs_meter(sample_rate: int):
    if pyln is None:
        return None
    meter = _LUFS_METERS.get(sample_rate)
    if meter is None:
        meter = pyln.Meter(sample_rate)
        _LUFS_METERS[sample_rate] = meter
    return meter


def _integrated_lufs(signal: np.ndarray, sample_rate: int) -> Optional[float]:
    meter = _get_lufs_meter(sample_rate)
    if meter is None or signal.size == 0:
        return None
    try:
        return float(meter.integrated_loudness(signal.astype(np.float32)))
    except Exception:
        return None


def _load_audio(path: Path, sample_rate: int, duration: float, offset: float) -> np.ndarray:
    try:
        bn_audio = importlib.import_module("birdnet_analyzer.audio")
        sig, _ = bn_audio.openAudioFile(
            str(path),
            sample_rate=sample_rate,
            offset=offset,
            duration=duration,
            fmin=None,
            fmax=None,
        )
        return sig.astype(np.float32)
    except Exception:
        import librosa

        sig, _ = librosa.load(
            str(path),
            sr=sample_rate,
            mono=True,
            offset=offset,
            duration=duration,
            res_type="kaiser_fast",
        )
        return sig.astype(np.float32)


def build_noise_duration_map(noise_files: List[Path], sample_rate: int) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for noise_path in noise_files:
        try:
            bn_audio = importlib.import_module("birdnet_analyzer.audio")
            durations[str(noise_path)] = float(bn_audio.getAudioFileLength(str(noise_path), sample_rate))
        except Exception:
            try:
                import librosa

                durations[str(noise_path)] = float(librosa.get_duration(filename=str(noise_path), sr=sample_rate))
            except Exception:
                durations[str(noise_path)] = 0.0
    return durations


def build_noise_mix_config(
    noise_dir: Path,
    snr_values: List[float],
    probability: float,
    max_sources: int,
    sample_rate: int,
) -> Optional[NoiseMixConfig]:
    if not noise_dir.exists():
        return None
    noise_files = [p for p in sorted(noise_dir.rglob("*")) if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3")]
    if not noise_files:
        return None

    snr_values = [float(v) for v in snr_values if math.isfinite(v)]
    probability = float(np.clip(probability, 0.0, 1.0))
    max_sources = max(1, int(max_sources))
    if not snr_values or probability <= 0.0:
        return None

    durations = build_noise_duration_map(noise_files, sample_rate)
    return NoiseMixConfig(
        probability=probability,
        snr_values=snr_values,
        max_sources=max_sources,
        noise_files=noise_files,
        noise_durations=durations,
    )


def build_noise_mix(
    sig: np.ndarray,
    sample_rate: int,
    config: Optional[NoiseMixConfig],
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], bool]:
    if config is None or not config.is_enabled():
        return None, False
    if sig.size == 0 or rng.random() > config.probability:
        return None, False

    target_len = sig.shape[0]
    duration_sec = max(target_len / float(sample_rate or 1), 1e-3)
    num_sources = int(rng.integers(1, config.max_sources + 1))
    noise_mix = np.zeros(target_len, dtype=np.float32)

    for _ in range(num_sources):
        noise_path = config.noise_files[int(rng.integers(0, len(config.noise_files)))]
        total_duration = max(config.noise_durations.get(str(noise_path), 0.0), 0.0)
        offset = 0.0
        if total_duration > duration_sec:
            offset = float(rng.uniform(0.0, max(total_duration - duration_sec, 1e-3)))
        try:
            noise_sig = _load_audio(noise_path, sample_rate, duration_sec, offset)
        except Exception:
            continue

        if noise_sig.size == 0:
            continue

        if noise_sig.shape[0] < target_len:
            reps = int(np.ceil(target_len / max(noise_sig.shape[0], 1)))
            noise_sig = np.tile(noise_sig, reps)[:target_len]
        elif noise_sig.shape[0] > target_len:
            start = int(rng.integers(0, noise_sig.shape[0] - target_len + 1))
            noise_sig = noise_sig[start:start + target_len]
        else:
            noise_sig = noise_sig.copy()

        noise_mix += noise_sig.astype(np.float32)

    if not np.any(noise_mix):
        return None, False
    return noise_mix, True


def apply_noise_with_snr(
    signal: np.ndarray,
    noise_mix: np.ndarray,
    sample_rate: int,
    target_snr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    signal = signal.astype(np.float32)
    noise_mix = noise_mix.astype(np.float32)

    scale = None
    if pyln is not None:
        signal_lufs = _integrated_lufs(signal, sample_rate)
        noise_lufs = _integrated_lufs(noise_mix, sample_rate)
        if (
            signal_lufs is not None
            and noise_lufs is not None
            and math.isfinite(signal_lufs)
            and math.isfinite(noise_lufs)
        ):
            target_noise_lufs = signal_lufs - float(target_snr)
            gain_db = target_noise_lufs - noise_lufs
            scale = 10 ** (gain_db / 20.0)
    else:
        global _LUFS_WARNED
        if not _LUFS_WARNED:
            _LUFS_WARNED = True

    if scale is None:
        signal_rms = float(np.sqrt(np.mean(signal ** 2) + 1e-12))
        noise_rms = float(np.sqrt(np.mean(noise_mix ** 2) + 1e-12))
        if signal_rms <= 0.0 or noise_rms <= 0.0:
            return signal, np.zeros_like(signal)
        desired_noise_rms = signal_rms / (10 ** (float(target_snr) / 20.0))
        scale = desired_noise_rms / noise_rms

    noise_component = noise_mix * float(scale)
    mixed = signal + noise_component

    max_amp = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if max_amp > 1.0:
        mixed = mixed / max_amp * 0.99

    return mixed.astype(np.float32), noise_component.astype(np.float32)


def compute_effective_snr(signal: np.ndarray, noise_component: np.ndarray) -> float:
    eps = 1e-8
    sig_rms = float(np.sqrt(np.mean(signal ** 2) + eps))
    noise_rms = float(np.sqrt(np.mean(noise_component ** 2) + eps))
    if sig_rms <= 0.0 or noise_rms <= 0.0:
        return 0.0
    return 20.0 * np.log10(sig_rms / (noise_rms + eps))
