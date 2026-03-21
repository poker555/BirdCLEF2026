import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PANNsDataset(Dataset):
    """
    從 HDF5 讀取原始波形，根據音訊長度動態產生多個 5 秒片段。
    每支音訊可貢獻 max(1, duration_sec // chunk_sec) 個訓練樣本。
    """

    def __init__(self, h5_path, keys, sample_rate=32000, chunk_sec=5):
        """
        :param h5_path:     train_waveforms.h5 路徑
        :param keys:        要使用的 dataset key 清單
        :param sample_rate: 取樣率，預設 32000
        :param chunk_sec:   每個片段長度（秒），預設 5
        """
        self.h5_path = h5_path
        self.sample_rate = sample_rate
        self.chunk_length = sample_rate * chunk_sec

        # 預先掃描每支音訊的總 sample 數，建立 (key, seg_idx, total_samples) 索引表
        self.samples = []

        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                total_samples = f[key].shape[0]  # waveform shape: (samples,)
                n_segments = max(1, total_samples // self.chunk_length)
                for seg_idx in range(n_segments):
                    self.samples.append((key, seg_idx, total_samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, seg_idx, total_samples = self.samples[idx]

        with h5py.File(self.h5_path, 'r') as f:
            ds = f[key]
            label_id = ds.attrs['label_id']

            if total_samples <= self.chunk_length:
                # 音訊比 5 秒短，讀全部再補零
                waveform = ds[:]
            else:
                # 在這個 segment 的範圍內隨機取起點
                seg_start = seg_idx * self.chunk_length
                seg_end = min(seg_start + self.chunk_length, total_samples) - self.chunk_length
                start = seg_start if seg_start >= seg_end else np.random.randint(seg_start, seg_end + 1)
                waveform = ds[start:start + self.chunk_length]

        # 確保長度正確
        if len(waveform) < self.chunk_length:
            waveform = np.pad(waveform, (0, self.chunk_length - len(waveform)), mode='constant')

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)
