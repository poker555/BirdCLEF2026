import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import warnings


class PANNsDataset(Dataset):
    """
    直接讀取 .ogg 原始波形，根據音訊長度動態產生多個 5 秒片段。
    每支音訊可貢獻 max(1, duration_sec // 5) 個訓練樣本。
    """

    def __init__(self, df, audio_dir="train_audio", sample_rate=32000, chunk_sec=5):
        """
        :param df:          pandas DataFrame，需含 'filename' 與 'label_id' 欄位
        :param audio_dir:   音檔根目錄
        :param sample_rate: 取樣率，預設 32000
        :param chunk_sec:   每個片段長度（秒），預設 5
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.chunk_length = sample_rate * chunk_sec

        # 預先掃描每支音訊長度，建立 (filename, label_id, seg_idx, total_samples) 索引表
        self.samples = []

        for _, row in df.iterrows():
            file_path = self.audio_dir / row['filename']
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    duration = librosa.get_duration(path=str(file_path))
                total_samples = int(duration * sample_rate)
                n_segments = max(1, int(duration) // chunk_sec)
            except Exception:
                # 讀不到長度就當成 1 個片段
                total_samples = self.chunk_length
                n_segments = 1

            for seg_idx in range(n_segments):
                self.samples.append((row['filename'], int(row['label_id']), seg_idx, total_samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label_id, seg_idx, total_samples = self.samples[idx]
        file_path = self.audio_dir / filename

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, _ = librosa.load(str(file_path), sr=self.sample_rate, mono=True)

            if total_samples <= self.chunk_length:
                # 音訊比 5 秒短，補零
                pad_size = self.chunk_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_size), mode='constant')
            else:
                # 在這個 segment 的範圍內隨機取起點
                seg_start = seg_idx * self.chunk_length
                seg_end = min(seg_start + self.chunk_length, len(waveform)) - self.chunk_length
                start = seg_start if seg_start >= seg_end else np.random.randint(seg_start, seg_end + 1)
                waveform = waveform[start:start + self.chunk_length]

            # 最後確保長度正確
            if len(waveform) < self.chunk_length:
                waveform = np.pad(waveform, (0, self.chunk_length - len(waveform)), mode='constant')

        except Exception as e:
            print(f"警告：讀取 {filename} 時出錯 ({e})")
            waveform = np.zeros(self.chunk_length, dtype=np.float32)

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)
