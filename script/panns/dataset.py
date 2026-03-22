import threading

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# thread-local HDF5 file handle：避免每次 __getitem__ 重複開關檔案
_tl = threading.local()


def _get_h5(path: str) -> h5py.File:
    """取得（或重建）當前 thread 的 HDF5 file handle。"""
    f = getattr(_tl, 'file', None)
    if f is None or not f.id.valid:
        _tl.file = h5py.File(path, 'r')
    return _tl.file


class PANNsDataset(Dataset):
    """
    從 HDF5 讀取整支音訊的原始波形，根據音訊長度動態產生多個 5 秒片段。
    人聲時間段已在前處理階段靜音，隨機裁切不會取到人聲。
    """

    def __init__(self, h5_path, keys, chunk_length=160000):
        """
        :param h5_path:      train_waveforms.h5 路徑
        :param keys:         要使用的 dataset key 清單
        :param chunk_length: 每個片段的 sample 數，預設 160000 (5秒 @ 32kHz)
        """
        self.h5_path = h5_path
        self.chunk_length = chunk_length

        # 預先計算每支音訊能產生幾個片段
        self.samples = []
        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                total_samples = f[key].shape[0]  # shape: (samples,)
                n_segments = max(1, total_samples // chunk_length)
                for seg_idx in range(n_segments):
                    self.samples.append((key, seg_idx, total_samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, seg_idx, total_samples = self.samples[idx]

        f  = _get_h5(self.h5_path)
        ds = f[key]
        soft_label = ds.attrs['soft_label'].astype(np.float32)  # (234,)
        class_id   = int(ds.attrs['class_id'])

        if total_samples <= self.chunk_length:
            waveform = ds[:]
        else:
            # 修正 off-by-one：max_start 是合法的最大起始點
            max_start = total_samples - self.chunk_length
            seg_start = seg_idx * self.chunk_length
            seg_end   = min(seg_start + self.chunk_length, max_start)
            start     = int(np.random.randint(seg_start, seg_end + 1))
            waveform  = ds[start:start + self.chunk_length]

        if len(waveform) < self.chunk_length:
            waveform = np.pad(waveform, (0, self.chunk_length - len(waveform)))

        return (torch.tensor(waveform, dtype=torch.float32),
                torch.tensor(soft_label, dtype=torch.float32),
                torch.tensor(class_id, dtype=torch.long))
