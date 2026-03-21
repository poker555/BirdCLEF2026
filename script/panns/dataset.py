import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PANNsDataset(Dataset):
    """
    從 HDF5 讀取已切好的 5 秒原始波形片段。
    preprocess_waveform.py 已在前處理階段切片並過濾人聲，
    每個 key 格式為 'species/filename.ogg/chunk_N'。
    """

    def __init__(self, h5_path, keys, chunk_length=160000):
        """
        :param h5_path:      train_waveforms.h5 路徑
        :param keys:         要使用的 dataset key 清單
        :param chunk_length: 每個片段的 sample 數，預設 160000 (5秒 @ 32kHz)
        """
        self.h5_path = h5_path
        self.keys = keys
        self.chunk_length = chunk_length

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with h5py.File(self.h5_path, 'r') as f:
            ds = f[key]
            waveform = ds[:]                  # shape: (samples,)
            label_id = ds.attrs['label_id']

        # 補零對齊（前處理已切好，通常不需要）
        if len(waveform) < self.chunk_length:
            waveform = np.pad(waveform, (0, self.chunk_length - len(waveform)))

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)
