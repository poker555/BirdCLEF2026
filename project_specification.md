# 專案規格：BirdCLEF 2026 音訊分類系統

> 最後更新：2026-03-25
> 架構決策：專注單一 PANNs CNN10 模型，不實施雙模型 Ensemble

## 1. 專案目標
開發高準確率的鳥類音訊分類系統，目標超越銅牌門檻（ROC-AUC 0.913）。
評分方式：macro-averaged ROC-AUC，跳過無真實正樣本的類別。

## 2. 核心架構：PANNs CNN10

### 2.1 目前參數（已訓練，Kaggle 0.87）

| 參數 | 值 | 說明 |
| :--- | :--- | :--- |
| 取樣率 | 32kHz | |
| n_fft | 1024 | FFT 視窗大小 |
| hop_length | 320 | 約 10ms 步長 |
| n_mels | 128 | Mel 頻帶數 |
| f_min | 50 Hz | |
| f_max | 16000 Hz | |
| 時間窗口 | 5s（160000 samples）| 對齊 Kaggle 評分機制 |
| 類別數 | 234 | |

### 2.2 下一版目標參數（待重新前處理後訓練）

| 參數 | 目前值 | 目標值 | 理由 |
| :--- | :---: | :---: | :--- |
| n_fft | 1024 | 512 | 提升時間解析度（BirdNET 論文建議） |
| n_mels | 128 | 160 | 提升頻率解析度，捕捉更細緻鳥鳴特徵 |
| hop_length | 320 | 320 | 維持不變 |

> 兩項調整都需要重新執行 `preprocess_waveform.py` 重建 `train_waveforms.h5`。

## 3. 模型架構細節

```
Input: waveform (B, 160000)
  ↓ MelSpectrogram → AmplitudeToDB
  ↓ BN0 (mel_bins)
  ↓ ConvBlock1: 1→64,   pool (2,2)
  ↓ ConvBlock2: 64→128, pool (2,2)
  ↓ ConvBlock3: 128→256, pool (2,2)
  ↓ ConvBlock4: 256→512, pool (2,2)
  ↓ Global Mean + Max Pooling → (B, 512)
  ↓ Dropout(0.2) → FC(512→512) → Dropout(0.2)
  ├─ fc_audioset → (B, 234)   ← 主輸出（物種分類）
  └─ fc_class    → (B, 5)     ← 輔助輸出（大類，僅訓練時使用）
```

## 4. 訓練設定

| 項目 | 設定 |
| :--- | :--- |
| Loss（主） | BCEWithLogitsLoss |
| Loss（輔助） | CrossEntropyLoss × 0.2 |
| Optimizer | AdamW，lr=1e-3 |
| Scheduler | ReduceLROnPlateau（mode=max，factor=0.5，patience=2） |
| Batch size | 64 |
| Max epochs | 100（early stopping patience=5） |
| Mixed precision | bfloat16（CUDA only） |

## 5. 資料處理管線

### 5.1 前處理（preprocess_waveform.py）
- 原始 .ogg → 32kHz 單聲道波形
- 人聲靜音（voice_detection_report.csv）
- 稀少物種（≤5筆）離線增強 ×4：time stretch ×0.9/1.1、pitch shift ±1 半音
- 輸出：`processed_data/train_waveforms.h5`（lzf 壓縮）

### 5.2 訓練時增強（動態，不需重新前處理）
- **MixUp**：alpha=0.4，prob=0.7，最多混合 2 個樣本
- **噪音混入**：p=0.5，SNR 5~20dB，來源為 train_soundscapes 低能量片段
- **SpecAugment**（待實作）：頻率遮罩 + 時間遮罩

### 5.3 驗證集策略
- 稀少物種（≤5筆）：全進訓練集
- 一般物種：stratified 80/20 split
- Soundscape 資料：同樣依稀少/一般分流

### 5.4 WeightedRandomSampler
- weight = quality_weight(XC rating) / sqrt(species_count)
- 平衡類別不平衡，同時保留品質資訊

## 6. 推論管線（inference_panns.py）

- 每支音訊切成 5 秒片段
- TTA ×3 取平均：原始 / 音量 +6dB / 時間翻轉
- 輸出：sigmoid 機率 → submission.csv

## 7. 已捨棄的方向

| 項目 | 原因 |
| :--- | :--- |
| EfficientNet 模型 | PANNs 效果已足夠，維持單模型簡化流程 |
| 雙模型 Ensemble | 暫不實施，先以單模型衝分數 |
| Focal Loss | BirdNET 論文實驗顯示無效 |
| LME Pooling | 適用於長音訊聚合，與本專案 per-5s 格式不符 |
| ESC-50 噪音資料庫 | train_soundscapes 與競賽 domain 更匹配 |
