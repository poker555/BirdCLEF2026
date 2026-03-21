import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BirdDataset(Dataset):
    """
    從 HDF5 讀取整張 Mel-Spectrogram，根據音訊長度動態產生多個 5 秒片段。
    人聲時間段已在前處理階段靜音，隨機裁切不會取到人聲。
    """

    def __init__(self, h5_path, keys, chunk_length=313):
        self.h5_path = h5_path
        self.chunk_length = chunk_length

        # 預先計算每支音訊能產生幾個片段，建立 (key, seg_idx, total_frames) 索引表
        self.samples = []
        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                total_frames = f[key].shape[1]  # shape: (n_mels, time_frames)
                n_segments = max(1, total_frames // chunk_length)
                for seg_idx in range(n_segments):
                    self.samples.append((key, seg_idx, total_frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, seg_idx, total_frames = self.samples[idx]

        with h5py.File(self.h5_path, 'r') as f:
            ds = f[key]
            label_id = ds.attrs['label_id']

            if total_frames <= self.chunk_length:
                mel_matrix = ds[:]
            else:
                seg_start = seg_idx * self.chunk_length
                seg_end   = min(seg_start + self.chunk_length, total_frames) - self.chunk_length
                start = seg_start if seg_start >= seg_end else np.random.randint(seg_start, seg_end + 1)
                mel_matrix = ds[:, start:start + self.chunk_length]

        if mel_matrix.shape[1] < self.chunk_length:
            mel_matrix = np.pad(mel_matrix, ((0, 0), (0, self.chunk_length - mel_matrix.shape[1])))

        mel_matrix = np.expand_dims(mel_matrix, axis=0)  # (1, n_mels, chunk_length)
        return torch.tensor(mel_matrix, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)


if __name__ == '__main__':
    h5_path = 'processed_data/train_spectrograms.h5'
    keys = []
    with h5py.File(h5_path, 'r') as f:
        for group in f.keys():
            for dset in f[group].keys():
                keys.append(f"{group}/{dset}")

    print(f"總共找到 {len(keys)} 支音訊")
    dataset = BirdDataset(h5_path, keys)
    print(f"展開後共 {len(dataset)} 個訓練片段")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    imgs, labels = next(iter(loader))
    print(f"圖片 shape: {imgs.shape}, 標籤 shape: {labels.shape}")
