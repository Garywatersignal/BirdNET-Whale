from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import librosa

try:
    import scaper
except Exception:
    scaper = None

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None


def _ensure_mono(sig: np.ndarray) -> np.ndarray:
    if sig is None:
        return np.zeros(0, dtype=np.float32)
    if sig.ndim == 1:
        return sig.astype(np.float32)
    if sig.ndim == 2:
        return np.mean(sig, axis=1).astype(np.float32)
    return sig.reshape(-1).astype(np.float32)


def _resample_signal(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or sig.size == 0:
        return sig.astype(np.float32)
    return librosa.resample(sig.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def _pad_or_trim(sig: np.ndarray, target_len: int) -> np.ndarray:
    if sig.shape[0] < target_len:
        return np.pad(sig, (0, target_len - sig.shape[0]), mode="constant")
    return sig[:target_len]


def _compute_rms(sig: np.ndarray) -> float:
    if sig.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(sig ** 2) + 1e-12))


def _integrated_lufs(signal: np.ndarray, sample_rate: int) -> Optional[float]:
    if pyln is None or signal.size == 0:
        return None
    try:
        meter = pyln.Meter(sample_rate)
        return float(meter.integrated_loudness(signal.astype(np.float32)))
    except Exception:
        return None


def compute_effective_snr(fg: np.ndarray, bg: np.ndarray) -> float:
    fg_rms = _compute_rms(fg)
    bg_rms = _compute_rms(bg)
    if fg_rms <= 0.0 or bg_rms <= 0.0:
        return 0.0
    return 20.0 * np.log10(fg_rms / (bg_rms + 1e-12))


def _extract_scaper_audio(output) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if isinstance(output, np.ndarray):
        return output, None
    if isinstance(output, tuple):
        audio = None
        sr = None
        for item in output:
            if isinstance(item, np.ndarray):
                if audio is None or item.size > audio.size:
                    audio = item
            elif isinstance(item, (int, float)):
                rate = int(item)
                if 8000 <= rate <= 192000:
                    sr = rate
        return audio, sr
    return None, None


def _generate_scaper_audio(sc) -> Tuple[np.ndarray, Optional[int]]:
    try:
        output = sc.generate(
            None,
            None,
            allow_repeated_label=True,
            allow_repeated_source=True,
            reverb=None,
            disable_sox_warnings=True,
            no_audio=False,
        )
    except TypeError:
        output = sc.generate(
            None,
            None,
            allow_repeated_label=True,
            allow_repeated_source=True,
            reverb=None,
            disable_sox_warnings=True,
        )
    sig, sr = _extract_scaper_audio(output)
    if sig is None:
        raise RuntimeError("Scaper did not return audio for in-memory generation.")
    return sig, sr


class SoundscapeGenerator:
    def __init__(self, noise_dir: str, sample_rate: int, duration: float, use_scaper: bool = True):
        self.noise_dir = Path(noise_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_scaper = use_scaper
        self.noise_files = self._collect_noise_files()
        self._audio_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def _collect_noise_files(self):
        if not self.noise_dir.exists():
            return []
        return sorted([p for p in self.noise_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")])

    def _load_audio_cached(self, path: Path, target_len: int) -> Optional[np.ndarray]:
        key = (str(path), int(self.sample_rate), int(target_len))
        cached = self._audio_cache.get(key)
        if cached is not None:
            return cached.copy()
        try:
            sig, _ = librosa.load(
                str(path),
                sr=self.sample_rate,
                mono=True,
                duration=self.duration,
                res_type="kaiser_fast",
            )
        except Exception:
            return None
        sig = _ensure_mono(sig)
        sig = _pad_or_trim(sig, target_len)
        sig = sig.astype(np.float32)
        self._audio_cache[key] = sig
        return sig.copy()

    def _pick_noise_file(self, rng: np.random.Generator) -> Optional[Path]:
        if not self.noise_files:
            return None
        return self.noise_files[int(rng.integers(0, len(self.noise_files)))]

    def generate(
        self,
        fg_path: str,
        target_snr: float,
        seed: int,
        *,
        duration_mode: str = "full_or_partial",
        partial_prob: float = 0.35,
        partial_dur_range: Tuple[float, float] = (1.0, 2.5),
        full_dur_range: Tuple[float, float] = (2.8, 3.0),
        allow_random_event_time: bool = True,
        distractor_path: Optional[str] = None,
        distractor_prob: float = 0.25,
        distractor_dur_range: Tuple[float, float] = (0.5, 1.5),
        distractor_snr_offset_range: Tuple[float, float] = (-10.0, -3.0),
        bg2_path: Optional[str] = None,
        bg_mix_prob: float = 0.35,
        bg2_rel_db_range: Tuple[float, float] = (-10.0, -3.0),
        compute_eff_snr: bool = True,
    ) -> Tuple[np.ndarray, float]:
        if scaper is None or not self.use_scaper:
            raise RuntimeError("scaper is required for soundscape training. Please install scaper.")
        if not self.noise_files:
            raise RuntimeError("No background noise files found for soundscape generation.")

        rng = np.random.default_rng(seed)
        duration = float(self.duration)
        target_len = int(round(self.sample_rate * duration))

        bg1_path = self._pick_noise_file(rng)
        if bg1_path is None:
            raise RuntimeError("No background noise file available.")

        fg_file = Path(fg_path)
        fg_dir = fg_file.parent
        fg_root = fg_dir.parent if fg_dir.parent is not None else fg_dir
        bg_dir = bg1_path.parent
        bg_root = bg_dir.parent if bg_dir.parent is not None else bg_dir

        sc = scaper.Scaper(
            duration=duration,
            fg_path=str(fg_root),
            bg_path=str(bg_root),
            random_state=int(seed),
        )

        sc.add_background(
            label=("const", bg_dir.name),
            source_file=("const", str(bg1_path)),
            source_time=("const", 0.0),
        )

        if duration_mode == "full_or_partial" and rng.random() < partial_prob:
            chosen_dur = rng.uniform(partial_dur_range[0], partial_dur_range[1])
        else:
            chosen_dur = rng.uniform(full_dur_range[0], min(full_dur_range[1], duration))

        event_time_max = max(0.0, duration - float(chosen_dur))
        event_time = rng.uniform(0.0, event_time_max) if allow_random_event_time and event_time_max > 0 else 0.0

        sc.add_event(
            label=("const", fg_dir.name),
            source_file=("const", str(fg_file)),
            source_time=("const", 0.0),
            event_time=("const", float(event_time)),
            event_duration=("const", float(chosen_dur)),
            snr=("const", float(target_snr)),
            pitch_shift=("const", 0),
            time_stretch=("const", 1.0),
        )

        if distractor_path and rng.random() < distractor_prob:
            d_dur = rng.uniform(distractor_dur_range[0], distractor_dur_range[1])
            d_event_time_max = max(0.0, duration - float(d_dur))
            d_event_time = rng.uniform(0.0, d_event_time_max) if d_event_time_max > 0 else 0.0
            d_offset = rng.uniform(distractor_snr_offset_range[0], distractor_snr_offset_range[1])
            sc.add_event(
                label=("const", Path(distractor_path).parent.name),
                source_file=("const", str(distractor_path)),
                source_time=("const", 0.0),
                event_time=("const", float(d_event_time)),
                event_duration=("const", float(d_dur)),
                snr=("const", float(target_snr + d_offset)),
                pitch_shift=("const", 0),
                time_stretch=("const", 1.0),
            )

        sig, rate = _generate_scaper_audio(sc)
        sig = _ensure_mono(sig)
        if rate and int(rate) != int(self.sample_rate):
            sig = _resample_signal(sig, int(rate), int(self.sample_rate))
        sig = _pad_or_trim(sig, target_len).astype(np.float32)

        bg2_used = False
        bg2_gain = None
        bg2_sig = None
        if rng.random() < bg_mix_prob:
            if bg2_path is None:
                bg2_path = str(self._pick_noise_file(rng) or "")
            if bg2_path:
                bg2_sig = self._load_audio_cached(Path(bg2_path), target_len)
                if bg2_sig is not None and bg2_sig.shape[0] == target_len:
                    rel_db = rng.uniform(bg2_rel_db_range[0], bg2_rel_db_range[1])
                    sig_lufs = _integrated_lufs(sig, self.sample_rate)
                    bg2_lufs = _integrated_lufs(bg2_sig, self.sample_rate)
                    if sig_lufs is not None and bg2_lufs is not None:
                        target_bg2_lufs = sig_lufs + rel_db
                        gain_db = target_bg2_lufs - bg2_lufs
                        bg2_gain = 10 ** (gain_db / 20.0)
                    else:
                        sig_rms = _compute_rms(sig)
                        bg2_rms = _compute_rms(bg2_sig)
                        if bg2_rms > 1e-12:
                            target_bg2_rms = sig_rms * (10 ** (rel_db / 20.0))
                            bg2_gain = target_bg2_rms / bg2_rms
                    if bg2_gain is not None:
                        sig = sig + bg2_gain * bg2_sig
                        bg2_used = True

        eff_snr = float(target_snr)
        if compute_eff_snr:
            bg1_sig = self._load_audio_cached(Path(bg1_path), target_len)
            if bg1_sig is not None and bg1_sig.shape[0] == target_len:
                bg_total = bg1_sig.copy()
                if bg2_used and bg2_sig is not None and bg2_gain is not None:
                    bg_total = bg_total + bg2_gain * bg2_sig
                fg_est = sig - bg_total
                eff_snr = compute_effective_snr(fg_est, bg_total)

        return sig.astype(np.float32), eff_snr
