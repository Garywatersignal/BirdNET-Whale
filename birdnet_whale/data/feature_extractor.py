import os
import sys
import importlib
from typing import Optional

import numpy as np
import librosa

from birdnet_whale.config import ExperimentConfig


class EmbeddingExtractor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.embedding_dim = config.embedding_dim
        self._backend_ready = False
        self._has_birdnet = False
        self._bn_model = None
        self._bn_audio = None
        self._rng = np.random.default_rng(config.split_seed)

    def _setup_backend(self) -> None:
        if self._backend_ready:
            return
        self._backend_ready = True

        if self.config.embedding_backend != "birdnet_analyzer":
            return

        try:
            bn_path = os.environ.get("BIRDNET_ANALYZER_PATH", self.config.birdnet_analyzer_path)
            if bn_path:
                bn_path = os.path.abspath(bn_path)
                if bn_path not in sys.path:
                    sys.path.insert(0, bn_path)
            bn_model = importlib.import_module("birdnet_analyzer.model")
            bn_audio = importlib.import_module("birdnet_analyzer.audio")
            bn_cfg = importlib.import_module("birdnet_analyzer.config")

            bn_cfg.SAMPLE_RATE = self.config.sample_rate
            bn_cfg.SIG_LENGTH = self.config.audio_duration

            if not os.path.isabs(bn_cfg.MODEL_PATH) and self.config.birdnet_analyzer_path:
                candidate = os.path.join(self.config.birdnet_analyzer_path, "birdnet_analyzer", bn_cfg.MODEL_PATH)
                if os.path.exists(candidate):
                    bn_cfg.MODEL_PATH = candidate

            self._bn_model = bn_model
            self._bn_audio = bn_audio
            self._has_birdnet = True
        except Exception:
            self._has_birdnet = False

    @staticmethod
    def _pad_or_truncate(sig: np.ndarray, target_len: int) -> np.ndarray:
        if sig.shape[0] >= target_len:
            return sig[:target_len]
        pad_len = target_len - sig.shape[0]
        return np.pad(sig, (0, pad_len), mode="constant")

    def _load_audio(self, audio_path: str, duration: Optional[float] = None) -> np.ndarray:
        duration = duration or self.config.audio_duration
        if self._has_birdnet and self._bn_audio is not None:
            sig, _ = self._bn_audio.openAudioFile(
                audio_path,
                sample_rate=self.config.sample_rate,
                duration=duration,
                fmin=None,
                fmax=None,
            )
        else:
            sig, _ = librosa.load(
                audio_path,
                sr=self.config.sample_rate,
                mono=True,
                duration=duration,
                res_type="kaiser_fast",
            )
        target_len = int(self.config.sample_rate * duration)
        sig = self._pad_or_truncate(sig, target_len)
        return sig.astype(np.float32)

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        if not self.config.normalize_embeddings:
            return emb
        denom = np.linalg.norm(emb) + 1e-8
        return (emb / denom).astype(np.float32)

    def extract_embedding_from_audio(self, audio_path: str) -> np.ndarray:
        self._setup_backend()
        sig = self._load_audio(audio_path)
        return self.extract_embedding_from_signal(sig)

    def load_audio(self, audio_path: str, duration: Optional[float] = None) -> np.ndarray:
        self._setup_backend()
        return self._load_audio(audio_path, duration)

    def extract_embedding_from_signal(self, sig: np.ndarray) -> np.ndarray:
        self._setup_backend()
        sig = np.asarray(sig, dtype=np.float32)
        if self._has_birdnet and self._bn_model is not None:
            emb = self._bn_model.embeddings([sig])[0]
            emb = np.array(emb, dtype=np.float32)
            if emb.shape[0] != self.embedding_dim:
                emb = emb.reshape(-1).astype(np.float32)
            return self._normalize(emb)

        if self.config.require_real_embeddings:
            raise RuntimeError("BirdNET backend not available. Set embedding_backend='birdnet_analyzer' and verify path.")
        emb = self._rng.standard_normal(self.embedding_dim).astype(np.float32)
        return self._normalize(emb)
