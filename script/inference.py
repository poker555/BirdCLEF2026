import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import warnings

import sys
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

sys.path.append(script_dir)

import timm
import torchaudio

# ==============================================================================
# 模型類別定義區塊 (直接內嵌於此，避免 Kaggle 環境找不到外部模組)
# ==============================================================================

# --- PANNsCNN10 相關架構 ---
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

class PANNsCNN10(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=128, classes_num=234):
        super(PANNsCNN10, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, 
            n_mels=mel_bins, f_min=50, f_max=14000
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.fc_class = nn.Linear(512, 5, bias=True)  # 輔助頭，推論時不使用
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        init_layer(self.fc_class)

    def forward(self, input):
        x = self.mel_spectrogram(input)
        x = self.amplitude_to_db(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1) 
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(2, 2))
        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc_audioset(x)


# --- CNN Backbone 相關架構 ---
class BirdModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=234):
        super(BirdModel,self).__init__()
        
        # ⚠️ Kaggle 提交環境沒有網路，必須強制取消預訓練網路下載 (pretrained=False)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=1
        )
        
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.fc_species = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
        self.fc_class   = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 5))  # 推論時不使用

    def forward(self, x):
        return self.fc_species(self.backbone(x))
# ==============================================================================

def predict_for_audio(model_panns, model_cnn, file_path, device, class_columns, batch_size=32):
    # Kaggle BirdCLEF 規定切割長度為 5 秒
    chunk_length = 5 
    sr = 32000
    target_length = sr * chunk_length
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 在推論階段單聲道讀取
            waveform, _ = librosa.load(str(file_path), sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    
    num_chunks = int(np.ceil(len(waveform) / target_length))
    predictions = []
    filename = file_path.stem
    
    wav_chunks = []
    mel_db_chunks = []
    row_ids = []
    
    # 避免洗版警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_chunks):
            start = i * target_length
            end = start + target_length
            chunk = waveform[start:end]
            
            # 若最後一段不足 5 秒，補零
            if len(chunk) < target_length:
                pad_size = target_length - len(chunk)
                chunk = np.pad(chunk, (0, pad_size), mode='constant')
                
            wav_chunks.append(chunk)
            
            # 給 CNN 模型用的頻譜特徵轉換
            mel_spec = librosa.feature.melspectrogram(
                y=chunk, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmin=150, fmax=16000
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_db_chunks.append(mel_db)
            
            row_ids.append(f"{filename}_{(i+1)*5}")
        
    wav_chunks = np.array(wav_chunks) # shape: (num_chunks, 160000)
    mel_db_chunks = np.array(mel_db_chunks) # shape: (num_chunks, 128, 313)
    
    tensor_wavs = torch.tensor(wav_chunks, dtype=torch.float32)
    tensor_mels = torch.tensor(mel_db_chunks, dtype=torch.float32).unsqueeze(1) # CNN 期望 (batch, 1, H, W)
    
    model_panns.eval()
    model_cnn.eval()
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(tensor_wavs), batch_size):
            batch_wav = tensor_wavs[i:i+batch_size].to(device)
            batch_mel = tensor_mels[i:i+batch_size].to(device)
            
            # 確保使用混合精度加速與防 NaN
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits_panns = model_panns(batch_wav)
                logits_cnn = model_cnn(batch_mel)
                
            probs_panns = F.softmax(logits_panns, dim=1)
            probs_cnn = F.softmax(logits_cnn, dim=1)
            
            # 模型融合 (Ensemble) - 這裡採用簡單平均法
            probs_ensemble = (probs_panns + probs_cnn) / 2.0
            all_probs.extend(probs_ensemble.cpu().float().numpy())
            
    for row_id, prob in zip(row_ids, all_probs):
        row_dict = {'row_id': row_id}
        for idx, col_name in enumerate(class_columns):
            row_dict[col_name] = prob[idx]
        predictions.append(row_dict)
        
    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"推論裝置: {device}")
    
    try:
        base_dir = Path(__file__).resolve().parent.parent # 指向 d:\BirdCLEF2026
    except NameError:
        # 在 Kaggle Notebook 上，目前的目錄會是 /kaggle/working/
        base_dir = Path(os.getcwd())
    kaggle_test_dir = Path('/kaggle/input/birdclef-2026/test_soundscapes')
    
    # 判斷運行環境與給定路徑
    if kaggle_test_dir.exists():
        test_dir = kaggle_test_dir
        sample_sub_path = '/kaggle/input/birdclef-2026/sample_submission.csv'
        # 在 Kaggle 上，您需要把模型權重當作 Data 掛載上來，路徑需依照您的 dataset 調整
        panns_model_path = '/kaggle/input/models/poker555/test-2-panns/pytorch/v1/1/best_panns_model.pth' 
        cnn_model_path = '/kaggle/input/models/poker555/birdclef2026-team-test1/pytorch/baseline/1/best_bird_model.pth' 
        is_kaggle = True
        print("偵測到 Kaggle 運行環境！")
    else:
        # 本地端測試路徑 (依據要求使用 train_audio)
        test_dir = base_dir / 'train_audio'
        sample_sub_path = base_dir / 'sample_submission.csv'
        panns_model_path = base_dir / 'best_panns_model.pth'
        cnn_model_path = base_dir / 'best_bird_model.pth'
        is_kaggle = False
        print("偵測到本地測試環境！")
        
    if not test_dir.exists():
        print(f"找不到測試資料夾 {test_dir}，此處終止執行。")
        return
        
    # 讀取 sample_submission.csv
    try:
        sample_sub = pd.read_csv(sample_sub_path)
        class_columns = sample_sub.columns[1:].tolist() # 排除第一欄 'row_id'
        print(f"成功讀取 {len(class_columns)} 個鳥類類別。")
    except Exception as e:
        print(f"讀取 {sample_sub_path} 失敗，錯誤：{e}")
        return
        
    # 建立兩個模型
    model_panns = PANNsCNN10(classes_num=len(class_columns)).to(device)
    model_cnn = BirdModel(num_classes=len(class_columns)).to(device)
    
    # 分別載入權重
    for m, path, name in zip([model_panns, model_cnn], [panns_model_path, cnn_model_path], ['PANNs', 'CNN']):
        try:
            m.load_state_dict(torch.load(path, map_location=device))
            print(f"成功載入 {name} 模型權重: {path}")
        except Exception as e:
            print(f"⚠️ 無法載入 {name} 權重 ({e})！將使用隨機初始化的權重進行測試。")
        
    all_predictions = []
    
    # 掃描所有音軌 (如果有子資料夾可以用 rglob)
    audio_files = list(test_dir.rglob('*.ogg'))
    
    # 條件邏輯：若在本地環境，只抽取前 100 筆測試
    if not is_kaggle:
        # 固定一下 seed 抽取隨機或直接取前 100 筆
        audio_files = audio_files[:100]
        print(f"本地測試模式：已擷取 {len(audio_files)} 個音檔準備分析...")
    else:
        print(f"Kaggle 推論模式：共有 {len(audio_files)} 個隱藏音檔準備全力分析...")
    
    if len(audio_files) == 0:
        print("資料夾中未找到音檔，產生預設的 submission.csv...")
        sample_sub.to_csv('submission.csv', index=False)
        return
    
    for idx, file_path in enumerate(audio_files, 1):
        print(f"正在預測 [{idx}/{len(audio_files)}]: {file_path.name}")
        preds = predict_for_audio(model_panns, model_cnn, file_path, device, class_columns)
        all_predictions.extend(preds)
        
    # 輸出最終合併的 submission.csv
    if len(all_predictions) > 0:
        submission_df = pd.DataFrame(all_predictions)
        submission_df = submission_df[['row_id'] + class_columns] # 確保欄位嚴格對齊
        
        # 由於使用 DataFrame 組合，將部分浮點數輸出格式化可避免過大
        submission_df.to_csv('submission.csv', index=False, float_format='%.6f')
        print(f"推論完成！共產出 {len(submission_df)} 筆預測，已成功匯出至 submission.csv。")
    else:
        sample_sub.to_csv('submission.csv', index=False)
        print("未產生預測，已回退輸出預設的 submission.csv")

if __name__ == '__main__':
    main()
