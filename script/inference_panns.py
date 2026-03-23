"""
BirdCLEF 2026 推論腳本（僅使用 PANNs 模型）
- 輸入：test_soundscapes/*.ogg
- 輸出：submission.csv
- 每支音訊切成 5 秒片段，以 sigmoid 輸出多標籤機率
"""

import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# PANNsCNN10 模型定義（內嵌，避免 Kaggle 找不到外部模組）
# ==============================================================================

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class PANNsCNN10(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                 mel_bins=128, classes_num=234, num_groups=5):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size,
            n_mels=mel_bins, f_min=50, f_max=16000
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.bn0         = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1,   64)
        self.conv_block2 = ConvBlock(64,  128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1         = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, classes_num)
        self.fc_class    = nn.Linear(512, num_groups)   # 輔助頭，推論時不使用
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        init_layer(self.fc_class)

    def forward(self, x):
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        # 與 panns/model.py 保持一致的 BN 維度處理
        x = x.unsqueeze(2)          # (B, F, 1, T)
        x = self.bn0(x)             # BN 作用在 F=mel_bins 維度
        x = x.squeeze(2)            # (B, F, T)
        x = x.transpose(1, 2)       # (B, T, F)
        x = x.unsqueeze(1)          # (B, 1, T, F)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc_audioset(x)


# ==============================================================================
# 推論核心
# ==============================================================================

SR            = 32000
CHUNK_SAMPLES = SR * 5   # 160000


def predict_file(model, file_path: Path, device, class_columns, batch_size=64):
    """將單支音訊切成 5 秒片段，回傳 list of {row_id, ...probs}"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            waveform, _ = librosa.load(str(file_path), sr=SR, mono=True)
    except Exception as e:
        print(f"[錯誤] 無法讀取 {file_path.name}: {e}")
        return []

    num_chunks = max(1, int(np.ceil(len(waveform) / CHUNK_SAMPLES)))
    chunks, row_ids = [], []

    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        chunk = waveform[start: start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
        row_ids.append(f"{file_path.stem}_{(i + 1) * 5}")

    tensor_wav = torch.tensor(np.array(chunks), dtype=torch.float32)
    all_probs  = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(tensor_wav), batch_size):
            batch = tensor_wav[i: i + batch_size].to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch)
            else:
                logits = model(batch)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)   # (num_chunks, num_classes)

    rows = []
    for row_id, prob in zip(row_ids, all_probs):
        row = {'row_id': row_id}
        row.update({col: float(prob[i]) for i, col in enumerate(class_columns)})
        rows.append(row)
    return rows


# ==============================================================================
# main
# ==============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"推論裝置: {device}")

    # ── 路徑設定 ──────────────────────────────────────────────────────
    kaggle_test_dir = Path('/kaggle/input/birdclef-2026/test_soundscapes')
    if kaggle_test_dir.exists():
        test_dir        = kaggle_test_dir
        sample_sub_path = Path('/kaggle/input/birdclef-2026/sample_submission.csv')
        model_path      = Path('/kaggle/input/models/poker555/birdclef2026-team-test1/pytorch/v3/1/best_panns_model.pth')
        is_kaggle       = True
        print("偵測到 Kaggle 環境")
    else:
        try:
            base_dir = Path(__file__).resolve().parent.parent
        except NameError:
            base_dir = Path(os.getcwd())
        test_dir        = base_dir / 'test_soundscapes'
        sample_sub_path = base_dir / 'sample_submission.csv'
        model_path      = base_dir / 'models' / 'best_panns_model.pth'
        is_kaggle       = False
        print("偵測到本地環境")

    # ── 讀取 sample_submission 取得欄位順序 ───────────────────────────
    try:
        sample_sub    = pd.read_csv(sample_sub_path)
        class_columns = sample_sub.columns[1:].tolist()
        print(f"共 {len(class_columns)} 個物種類別")
    except Exception as e:
        print(f"[錯誤] 無法讀取 {sample_sub_path}: {e}")
        return

    # ── 載入模型 ──────────────────────────────────────────────────────
    model = PANNsCNN10(classes_num=len(class_columns)).to(device)
    try:
        state = torch.load(str(model_path), map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"⚠️  缺少的 key（不影響推論）: {missing}")
        print(f"✅ 成功載入 PANNs 權重: {model_path}")
    except Exception as e:
        print(f"⚠️  無法載入權重 ({e})，使用隨機初始化繼續。")

    # ── 掃描音訊檔案 ──────────────────────────────────────────────────
    if not test_dir.exists():
        print(f"找不到測試資料夾 {test_dir}，輸出預設 submission.csv")
        sample_sub.to_csv('submission.csv', index=False)
        return

    audio_files = sorted(test_dir.rglob('*.ogg'))
    if not is_kaggle:
        audio_files = audio_files[:100]
        print(f"本地測試模式：取前 {len(audio_files)} 支音訊")
    else:
        print(f"Kaggle 模式：共 {len(audio_files)} 支音訊")

    if not audio_files:
        print("未找到音訊，輸出預設 submission.csv")
        sample_sub.to_csv('submission.csv', index=False)
        return

    # ── 逐檔推論 ──────────────────────────────────────────────────────
    all_predictions = []
    for idx, fp in enumerate(audio_files, 1):
        print(f"[{idx}/{len(audio_files)}] {fp.name}")
        all_predictions.extend(predict_file(model, fp, device, class_columns))

    # ── 輸出 submission.csv ───────────────────────────────────────────
    if all_predictions:
        df = pd.DataFrame(all_predictions)[['row_id'] + class_columns]
        df.to_csv('submission.csv', index=False, float_format='%.6f')
        print(f"\n推論完成，共 {len(df)} 筆，已輸出 submission.csv")
    else:
        sample_sub.to_csv('submission.csv', index=False)
        print("無預測結果，輸出預設 submission.csv")


if __name__ == '__main__':
    main()
