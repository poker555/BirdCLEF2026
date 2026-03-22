"""
SoundscapeDataset：從 train_soundscapes_labels.csv 讀取 5 秒片段。

split 參數控制分流：
  'train' - 稀少物種（train_audio ≤ rare_threshold 筆）全部進訓練集；
            一般物種依 random_state 做 80/20，取訓練的 80%
  'val'   - 一般物種的 20%
  'all'   - 全部（預設，向下相容）

soft label 設計：
  第一個物種（primary）= 1.0
  其餘共存物種         = 0.5（soundscape 多物種是真實共存，比 secondary_weight=0.3 高）
  完全沒有對應 label   → 跳過
"""

import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE       = 32000
CHUNK_LENGTH      = 160000   # 5 秒 @ 32kHz
COEXIST_WEIGHT    = 0.5      # soundscape 多物種共存權重


def _time_to_sec(t: str) -> int:
    """'HH:MM:SS' -> 秒數"""
    h, m, s = t.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def _build_samples(labels_csv: str, taxonomy_csv: str,
                   num_classes: int, sc_df: pd.DataFrame = None,
                   rare_threshold: int = 5, split: str = 'all',
                   random_state: int = 42):
    """
    解析 labels_csv，依 split 分流，回傳 samples list。
    sc_df: species_counts.csv（split != 'all' 時必須提供）
    """
    tax_df    = pd.read_csv(taxonomy_csv)
    label_map = {str(r['primary_label']): int(r['label_id']) for _, r in tax_df.iterrows()}
    class_map = {str(r['primary_label']): int(r['class_id']) for _, r in tax_df.iterrows()}

    df = pd.read_csv(labels_csv)

    # 建立 count_map（物種 → train_audio 筆數）
    count_map = {}
    if sc_df is not None:
        count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))

    all_samples = []
    for _, row in df.iterrows():
        start_sec    = _time_to_sec(str(row['start']))
        species_list = [s.strip() for s in str(row['primary_label']).split(';') if s.strip()]

        # soft label：第一個物種 1.0，其餘 0.5
        soft_label = np.zeros(num_classes, dtype=np.float32)
        class_id   = 0
        for i, s in enumerate(species_list):
            lid = label_map.get(s)
            if lid is not None:
                soft_label[lid] = 1.0 if i == 0 else COEXIST_WEIGHT
                if i == 0:
                    class_id = class_map.get(s, 0)

        if soft_label.sum() == 0:
            continue

        # 判斷這個片段的「代表物種」是否稀少（以第一個物種為準）
        primary_sp = species_list[0] if species_list else ''
        is_rare    = count_map.get(primary_sp, 0) <= rare_threshold

        all_samples.append({
            'filename':   row['filename'],
            'start_sec':  start_sec,
            'soft_label': soft_label,
            'class_id':   class_id,
            'is_rare':    is_rare,
        })

    if split == 'all':
        return all_samples

    # 分流：稀少物種全進 train，一般物種 80/20
    rare_samples   = [s for s in all_samples if s['is_rare']]
    normal_samples = [s for s in all_samples if not s['is_rare']]

    rng = np.random.default_rng(random_state)
    idx = np.arange(len(normal_samples))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    normal_train = [normal_samples[i] for i in idx[:cut]]
    normal_val   = [normal_samples[i] for i in idx[cut:]]

    if split == 'train':
        return rare_samples + normal_train
    else:  # 'val'
        return normal_val


class SoundscapeWaveformDataset(Dataset):
    """給 PANNs 用：回傳原始波形"""

    def __init__(self, soundscape_dir: str, labels_csv: str,
                 taxonomy_csv: str, num_classes: int = 234,
                 split: str = 'all', sc_df: pd.DataFrame = None,
                 rare_threshold: int = 5):
        """
        split: 'all' | 'train' | 'val'
        sc_df: species_counts DataFrame（split != 'all' 時需要）
        """
        self.soundscape_dir = Path(soundscape_dir)
        self.num_classes    = num_classes
        self.samples        = _build_samples(
            labels_csv, taxonomy_csv, num_classes,
            sc_df=sc_df, rare_threshold=rare_threshold,
            split=split
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import librosa
        s    = self.samples[idx]
        path = self.soundscape_dir / s['filename']

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                                    offset=s['start_sec'], duration=5.0)
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

    N_MELS       = 128
    N_FFT        = 2048
    HOP_LENGTH   = 512
    FMIN         = 150
    FMAX         = 16000
    CHUNK_FRAMES = 313

    def __init__(self, soundscape_dir: str, labels_csv: str,
                 taxonomy_csv: str, num_classes: int = 234,
                 split: str = 'all', sc_df: pd.DataFrame = None,
                 rare_threshold: int = 5):
        self._wav_ds = SoundscapeWaveformDataset(
            soundscape_dir, labels_csv, taxonomy_csv, num_classes,
            split=split, sc_df=sc_df, rare_threshold=rare_threshold
        )

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

        mel_db = np.expand_dims(mel_db, axis=0)
        return (torch.tensor(mel_db, dtype=torch.float32), soft_label, class_id)
