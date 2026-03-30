"""
NoiseDataset：從 ESC-50 資料集下載並抽取指定類別的音訊作為噪音來源。

保留類別（純自然環境音，模擬溼地錄音場景）：
  10 - rain          雨聲
  16 - wind          風聲
  17 - pouring_water 流水聲
  19 - thunderstorm  雷雨聲

使用方式：
    noise_ds = NoiseDataset('ESC-50-master/audio')
    noise_wav = noise_ds.sample()   # 回傳 (160000,) numpy array
"""

import warnings
import zipfile
import urllib.request
import numpy as np
from pathlib import Path

SAMPLE_RATE = 32000

# 保留的 ESC-50 target 編號
# 純自然環境音（溼地錄音場景）
# 10-rain, 16-wind, 17-pouring_water, 19-thunderstorm
ALLOWED_TARGETS = {10, 16, 17, 19}

ESC50_URL      = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
ESC50_ZIP_NAME = 'ESC-50-master.zip'
ESC50_DIR_NAME = 'ESC-50-master'


def _download_esc50(dest_dir: Path) -> Path:
    """下載 ESC-50 zip 並解壓縮到 dest_dir，回傳 audio 資料夾路徑。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path  = dest_dir / ESC50_ZIP_NAME
    audio_dir = dest_dir / ESC50_DIR_NAME / 'audio'

    if audio_dir.exists():
        print(f"[NoiseDataset] ESC-50 已存在：{audio_dir}")
        return audio_dir

    print(f"[NoiseDataset] 下載 ESC-50 中（約 600MB）...")
    urllib.request.urlretrieve(ESC50_URL, zip_path)
    print(f"[NoiseDataset] 解壓縮中...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    print(f"[NoiseDataset] ESC-50 已解壓至：{audio_dir}")
    return audio_dir


class NoiseDataset:
    def __init__(self, esc50_dir: str, chunk_length: int = 160000,
                 auto_download: bool = True, rng_seed: int = 42):
        """
        esc50_dir    : ESC-50 audio 資料夾路徑（即 ESC-50-master/audio/）
        chunk_length : 每個噪音片段長度（samples），預設 5 秒 @ 32kHz
        auto_download: 若 esc50_dir 不存在，自動下載 ESC-50
        rng_seed     : 隨機種子
        """
        self.chunk_length = chunk_length
        self.rng          = np.random.default_rng(rng_seed)
        self.chunks       = []

        audio_dir = Path(esc50_dir)
        if not audio_dir.exists():
            if auto_download:
                audio_dir = _download_esc50(audio_dir.parent)
            else:
                raise FileNotFoundError(
                    f"找不到 ESC-50 資料夾：{audio_dir}，"
                    "請設定 auto_download=True 或手動下載。"
                )

        self._build(audio_dir)
        print(f"[NoiseDataset] 收集到 {len(self.chunks)} 個噪音片段"
              f"（來自 {len(ALLOWED_TARGETS)} 個類別）")

    def _build(self, audio_dir: Path):
        import librosa
        for fp in sorted(audio_dir.glob('*.wav')):
            try:
                target = int(fp.stem.split('-')[-1])
            except ValueError:
                continue
            if target not in ALLOWED_TARGETS:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    y, _ = librosa.load(str(fp), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue
            if len(y) >= self.chunk_length:
                self.chunks.append(y[:self.chunk_length].astype(np.float32))
            else:
                self.chunks.append(np.pad(y, (0, self.chunk_length - len(y))).astype(np.float32))

    def sample(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(self.chunk_length, dtype=np.float32)
        return self.chunks[self.rng.integers(len(self.chunks))].copy()

    def __len__(self):
        return len(self.chunks)
