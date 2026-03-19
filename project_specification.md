# 專案規格：BirdCLEF 音訊分類系統 (雙模型融合)

## 1. 專案目標
開發一個具備高準確率的鳥類音訊分類系統，採用雙模型融合 (Dual-Model Ensemble) 策略，結合影像分類模型和音訊專用模型，以提升預測的穩健性與整體準確率。

## 2. 核心架構與參數設定
本專案將整合以下兩種深度學習模型，並根據模型特性設計不同的特徵萃取參數：

### Model 1: EfficientNet
將音訊轉換為高頻率解析度的梅爾頻譜圖 (Mel-Spectrogram)，將其視為二維影像進行特徵提取與分類。適合處理微小的頻率特徵。
- **取樣率 (Sample Rate)**：32kHz
- **FFT 大小 (n_fft)**：1024 (提供較高的頻率解析度)
- **重疊率 (Overlap)**：50%
- **特徵類型**：Mel Spectrogram
- **Mel 頻帶數 (mels)**：128
- **頻率範圍 (fmin, fmax)**：150Hz - 16000Hz (有效濾除環境低頻噪音)
- **時間窗口 (Chunk Size)**：5s (對齊 Kaggle 評分推論機制)

### Model 2: Pre-trained Audio Neural Network (PANNs)
直接針對音訊特徵設計，利用在大規模音訊資料集預訓練的強大權重。適合捕捉較高時間解析度的時序變化。
- **取樣率 (Sample Rate)**：32kHz
- **FFT 大小 (n_fft)**：512 (提供較級的時間解析度)
- **重疊率 (Overlap)**：50%
- **特徵類型**：Mel Spectrogram
- **Mel 頻帶數 (mels)**：128
- **頻率範圍 (fmin, fmax)**：150Hz - 16000Hz
- **時間窗口 (Chunk Size)**：5s

## 3. 融合策略 (Ensemble Strategy)
- 採用 **後期融合 (Late Fusion)**：
  - 將 EfficientNet 與 PANNs 兩個網路分別獨立訓練。
  - 在推論 (Inference) 階段，將兩者輸出的預測機率或是 Logits 進行**加權平均 (Weighted Average)**。
  - 例：`Final_Prediction = (Weight1 * Pred_EffNet) + (Weight2 * Pred_PANNs)` 

## 4. 資料處理管線 (Data Pipeline)
- **資料預處理與儲存 (Data Preprocessing & Storage)**：為了最大化 I/O 訓練效能，將原始音訊 (`.ogg`) 預先轉換為 Mel-Spectrogram 數值矩陣，並統一打包儲存為 **HDF5 格式 (`.h5`)**，供 PyTorch DataLoader 在訓練期間進行高吞吐量的隨機讀取。
- **資料擴增 (Data Augmentation)**：實作 MixUp 方法以提升模型泛化能力。
