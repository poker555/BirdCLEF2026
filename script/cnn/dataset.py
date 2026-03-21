import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BirdDataset(Dataset):
    """
    從 HDF5 讀取已切好的 5 秒 Mel-Spectrogram 片段。
    preprocess.py 已在前處理階段切片並過濾人聲，
    每個 key 格式為 'species/filename.ogg/chunk_N'。
    """

    def __init__(self, h5_path, keys, chunk_length=313):
        """
        :param h5_path:      train_spectrograms.h5 路徑
        :param keys:         要使用的 dataset key 清單
        :param chunk_length: time frame 數，用於補零對齊（預設 313）
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
            mel_matrix = ds[:]                        # shape: (n_mels, time_frames)
            label_id   = ds.attrs['label_id']

        # 補零或裁切對齊 chunk_length（前處理已切好，通常不需要）
        current_len = mel_matrix.shape[1]
        if current_len < self.chunk_length:
            mel_matrix = np.pad(mel_matrix, ((0, 0), (0, self.chunk_length - current_len)))
        elif current_len > self.chunk_length:
            start = np.random.randint(0, current_len - self.chunk_length + 1)
            mel_matrix = mel_matrix[:, start:start + self.chunk_length]

        mel_matrix = np.expand_dims(mel_matrix, axis=0)   # (1, n_mels, chunk_length)
        return torch.tensor(mel_matrix, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)


if __name__ == '__main__':
    h5_path = 'processed_data/train_spectrograms.h5'
    with h5py.File(h5_path, 'r') as f:
        # key 格式：species/filename.ogg/chunk_N，需遞迴兩層
        keys = []
        for species in f.keys():
            for fname in f[species].keys():
                for chunk in f[species][fname].keys():
                    keys.append(f"{species}/{fname}/{chunk}")

    print(f"總共找到 {len(keys)} 個訓練片段")
    dataset = BirdDataset(h5_path, keys)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)
    imgs, labels = next(iter(loader))
    print(f"圖片 shape: {imgs.shape}, 標籤 shape: {labels.shape}")
