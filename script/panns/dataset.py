import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

class PANNsDataset(Dataset):
    def __init__(self, df, audio_dir="train_audio", sample_rate=32000, max_length=5):
        """
        :param df: pandas DataFrame 包含 'filename' 與 'label_id' 兩個欄位
        :param audio_dir: 音檔所在的資料夾路徑
        :param sample_rate: 音訊採樣率，預設 32000 (BirdCLEF 與 PANNs 常用標準)
        :param max_length: 擷取的音訊長度 (秒)，預設 5 秒
        """
        self.df = df
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.target_length = self.sample_rate * self.max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label_id = row['label_id']
        file_path = self.audio_dir / filename
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 讀取音檔為原始聲波 (Mono, 32000Hz)
                waveform, _ = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            
            # 訓練時動態裁切 5 秒 (或者是其他指定的長度)
            if len(waveform) > self.target_length:
                # 隨機擷取一段 target_length 的聲波
                start = np.random.randint(0, len(waveform) - self.target_length)
                waveform = waveform[start:start + self.target_length]
            elif len(waveform) < self.target_length:
                # 若不足，則在前後補 0 (Padding)
                pad_size = self.target_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_size), mode='constant')

            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
            
        except Exception as e:
            # 以防萬一某個檔讀取失敗，拋出全零 Tensor 繼續往下跑
            print(f"警告：讀取 {filename} 時出錯 ({e})")
            waveform_tensor = torch.zeros(self.target_length, dtype=torch.float32)

        return waveform_tensor, torch.tensor(label_id, dtype=torch.long)
