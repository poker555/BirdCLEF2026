import h5py
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader

class BirdDataset(Dataset):
    def __init__(self, h5_path, keys, chunk_length=313):
        self.h5_path = h5_path 
        self.keys = keys
        self.chunk_length = chunk_length
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self,idx):
        file_key = self.keys[idx]
        with h5py.File(self.h5_path,'r') as h5_file:
            dataset = h5_file[file_key]
            mel_matrix = dataset[:]
            label_id = dataset.attrs['label_id']

        current_length = mel_matrix.shape[1]

        if current_length < self.chunk_length:
            pad_amount = self.chunk_length - current_length
            mel_matrix = np.pad(mel_matrix,((0, 0), (0, pad_amount)), mode='constant')

        elif current_length > self.chunk_length:
            max_start = current_length - self.chunk_length
            start_idx = np.random.randint(0, max_start + 1)
            mel_matrix = mel_matrix[:, start_idx : start_idx + self.chunk_length]

        mel_matrix = np.expand_dims(mel_matrix, axis=0)
        return torch.tensor(mel_matrix,dtype=torch.float32),torch.tensor(label_id,dtype=torch.long)

if __name__ == '__main__':
    print('準備開始測試 Dataset')
    h5_path = 'processed_data/train_spectrograms.h5'
    keys = []
    with h5py.File(h5_path,'r') as f:
        for group_name in f.keys():
            for dataset_name in f[group_name].keys():
                keys.append(f"{group_name}/{dataset_name}")
    
    print(f"總共找到 {len(keys)} 筆可訓練資料")

    dataset = BirdDataset(h5_path, keys, chunk_length=313)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch_images, batch_labels in dataloader:
        print("\n 第一批資料檢查")
        print(f"圖片張量形狀:{batch_images.shape}")
        print(f"標籤張量形狀:{batch_labels.shape}")
        print(f"物種ID:{batch_labels.tolist()}")
        break