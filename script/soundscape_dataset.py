"""
SoundscapeDataset：從 train_soundscapes_labels.csv 讀取 5 秒片段作為驗證集。
每筆回傳 (waveform_or_mel, soft_label, class_id)，格式與訓練 Dataset 一致。
"""

import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


SAMPLE_RATE  = 32000
CHUNK_LENGTH = 160000   # 5 秒 @ 32kHz


def _time_to_sec(t: str) -> int:
    """'HH:MM:SS' -> 秒數"""
    h, m, s = t.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


class SoundscapeWaveformDataset(Dataset):
    """給 PANNs 用：回傳原始波形"""

    def __init__(self, soundscape_dir: str, labels_csv: str,
                 taxonomy_csv: str, num_classes: int = 234):
        self.soundscape_dir = Path(soundscape_dir)
        self.num_classes    = num_classes

        tax_df   = pd.read_csv(taxonomy_csv)
        label_map = {str(r['primary_label']): int(r['label_id'])
                     for _, r in tax_df.iterrows()}
        class_map = {str(r['primary_label']): int(r['class_id'])
                     for _, r in tax_df.iterrows()}

        df = pd.read_csv(labels_csv)
        self.samples = []
        for _, row in df.iterrows():
            start_sec = _time_to_sec(str(row['start']))
            end_sec   = _time_to_sec(str(row['end']))
            species_list = [s.strip() for s in str(row['primary_label']).split(';') if s.strip()]

            soft_label = np.zeros(num_classes, dtype=np.float32)
            class_id   = 0
            for s in species_list:
                lid = label_map.get(s)
                if lid is not None:
                    soft_label[lid] = 1.0
                    class_id = class_map.get(s, 0)

            if soft_label.sum() == 0:
                continue  # 沒有對應 label 的片段跳過

            self.samples.append({
                'filename':  row['filename'],
                'start_sec': start_sec,
                'soft_label': soft_label,
                'class_id':  class_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import librosa
        s = self.samples[idx]
        path = self.soundscape_dir / s['filename']
        start = s['start_sec']

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                                    offset=start, duration=5.0)
        except Exception:
            y = np.zeros(CHUNK_LENGTH, dtype=np.float32)

        if len(y) < CHUNK_LENGTH:
            y = np.pad(y, (0, CHUNK_LENGTH - len(y)))
        else:
            y = y[:CHUNK_LENGTH]

        return (torch.tensor(y, dtype=torch.float32),
                torch.tensor(s['soft_label'], dtype=torch.float32),
                torch.tensor(s['class_id'], dtype=torch.long))


class SoundscapeMelDataset(Dataset):
    """給 CNN 用：回傳 Mel-Spectrogram"""

    N_MELS     = 128
    N_FFT      = 2048
    HOP_LENGTH = 512
    FMIN       = 150
    FMAX       = 16000
    CHUNK_FRAMES = 313  # 5 秒對應的 frame 數

    def __init__(self, soundscape_dir: str, labels_csv: str,
                 taxonomy_csv: str, num_classes: int = 234):
        # 共用 SoundscapeWaveformDataset 的解析邏輯
        self._wav_ds = SoundscapeWaveformDataset(
            soundscape_dir, labels_csv, taxonomy_csv, num_classes)

    def __len__(self):
        return len(self._wav_ds)

    def __getitem__(self, idx):
        import librosa
        wav, soft_label, class_id = self._wav_ds[idx]
        y = wav.numpy()

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=self.N_MELS,
            n_fft=self.N_FFT, hop_length=self.HOP_LENGTH,
            fmin=self.FMIN, fmax=self.FMAX
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

        if mel_db.shape[1] < self.CHUNK_FRAMES:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.CHUNK_FRAMES - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :self.CHUNK_FRAMES]

        mel_db = np.expand_dims(mel_db, axis=0)  # (1, n_mels, frames)
        return (torch.tensor(mel_db, dtype=torch.float32), soft_label, class_id)
