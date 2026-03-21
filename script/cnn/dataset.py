import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BirdDataset(Dataset):
    """
    從 HDF5 讀取 Mel-Spectrogram，根據音訊長度動態產生多個 5 秒片段。
    每支音訊可貢獻 max(1, total_frames // chunk_length) 個訓練樣本。
    """

    def __init__(self, h5_path, keys, chunk_length=313):
        """
        :param h5_path:      HDF5 檔案路徑
        :param keys:         要使用的 dataset key 清單
        :param chunk_length: 每個片段的 time frame 數，313 ≈ 5 秒 (sr=32000, hop=512)
        """
        self.h5_path = h5_path
        self.chunk_length = chunk_length

        # 預先計算每支音訊能產生幾個片段，建立 (key, segment_index) 的索引表
        self.samples = []  # list of (key, n_segments, total_frames)

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

        with h5py.File(self.h5_path, 'r') as h5_file:
            dataset = h5_file[key]
            label_id = dataset.attrs['label_id']

            if total_frames <= self.chunk_length:
                # 音訊比 5 秒短，直接讀全部再補零
                mel_matrix = dataset[:]
            else:
                # 計算這個 segment 的起始範圍，在該範圍內隨機取一個起點
                seg_start = seg_idx * self.chunk_length
                seg_end = min(seg_start + self.chunk_length, total_frames) - self.chunk_length
                # seg_end 可能等於 seg_start，此時 start 固定
                start = seg_start if seg_start >= seg_end else np.random.randint(seg_start, seg_end + 1)
                mel_matrix = dataset[:, start:start + self.chunk_length]

        # 不足 chunk_length 補零
        if mel_matrix.shape[1] < self.chunk_length:
            pad_amount = self.chunk_length - mel_matrix.shape[1]
            mel_matrix = np.pad(mel_matrix, ((0, 0), (0, pad_amount)), mode='constant')

        mel_matrix = np.expand_dims(mel_matrix, axis=0)  # (1, n_mels, chunk_length)
        return torch.tensor(mel_matrix, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)


if __name__ == '__main__':
    print('準備開始測試 Dataset')
    h5_path = 'processed_data/train_spectrograms.h5'
    keys = []
    with h5py.File(h5_path, 'r') as f:
        for group_name in f.keys():
            for dataset_name in f[group_name].keys():
                keys.append(f"{group_name}/{dataset_name}")

    print(f"總共找到 {len(keys)} 支音訊")

    dataset = BirdDataset(h5_path, keys, chunk_length=313)
    print(f"展開後共 {len(dataset)} 個訓練片段")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_images, batch_labels in dataloader:
        print(f"圖片張量形狀: {batch_images.shape}")
        print(f"標籤張量形狀: {batch_labels.shape}")
        break
