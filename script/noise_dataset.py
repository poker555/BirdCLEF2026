"""
NoiseDataset：從 train_soundscapes 中抽取低能量（無鳥聲）片段作為噪音來源。

使用方式：
    noise_ds = NoiseDataset(soundscape_dir, chunk_length=160000, max_files=500)
    noise_wav = noise_ds.sample()   # 回傳 (160000,) numpy array
"""

import warnings
import numpy as np
from pathlib import Path


SAMPLE_RATE   = 32000
RMS_THRESHOLD = 0.01   # 低於此 RMS 視為低能量片段（無明顯鳥聲）


class NoiseDataset:
    def __init__(self, soundscape_dir: str, chunk_length: int = 160000,
                 max_files: int = 500, rng_seed: int = 42):
        """
        soundscape_dir : train_soundscapes 資料夾路徑
        chunk_length   : 每個噪音片段長度（samples），預設 5 秒 @ 32kHz
        max_files      : 最多掃描幾支音訊（避免啟動太慢）
        """
        self.chunk_length = chunk_length
        self.rng = np.random.default_rng(rng_seed)
        self.chunks = []
        self._build(Path(soundscape_dir), max_files)
        print(f"[NoiseDataset] 收集到 {len(self.chunks)} 個低能量噪音片段")

    def _build(self, soundscape_dir: Path, max_files: int):
        import librosa
        files = sorted(soundscape_dir.glob('*.ogg'))[:max_files]
        for fp in files:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    y, _ = librosa.load(str(fp), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue

            # 切成 chunk_length 片段，只保留低能量的
            n = len(y) // self.chunk_length
            for i in range(n):
                chunk = y[i * self.chunk_length: (i + 1) * self.chunk_length]
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < RMS_THRESHOLD:
                    self.chunks.append(chunk.astype(np.float32))

    def sample(self) -> np.ndarray:
        """隨機回傳一個噪音片段 (chunk_length,)"""
        if not self.chunks:
            return np.zeros(self.chunk_length, dtype=np.float32)
        idx = self.rng.integers(len(self.chunks))
        return self.chunks[idx].copy()

    def __len__(self):
        return len(self.chunks)
