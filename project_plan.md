# 專案規劃：BirdCLEF 雙模型開發藍圖

## 1. 專案目錄結構設計

```text
BirdCLEF2026/
├── train_audio/           # 訓練用音訊檔案 (按鳥類分類)
├── train_soundscapes/     # 訓練用環境音景檔案
├── test_soundscapes/      # 測試用環境音景檔案 (隱藏，僅在 Kaggle Notebook 中可見部分)
├── processed_data/        # 存放透過預處理腳本生成的 HDF5 特徵檔案
├── train.csv              # 訓練音訊的 metadata 與標籤
├── train_soundscapes_labels.csv # 聲音景觀的標籤
├── taxonomy.csv           # 鳥類分類學資訊
├── sample_submission.csv  # 提交格式範例
├── recording_location.txt # 錄音地點資訊
├── script/                # 自訂的 Python 腳本目錄 (放置您開發的模型、資料前處理、推論等程式碼)
│   ├── preprocess.py      # 音訊特徵預處理腳本 (轉換 Mel-Spectrogram 並儲存為 HDF5)
│   ├── dataset.py         # PyTorch Dataset 實作 (負責讀取 HDF5 檔案與 MixUp 擴增)
│   ├── models/            # 放置自定義模型
│   │   ├── efficientnet.py 
│   │   ├── panns.py        
│   │   └── ensemble.py     
│   ├── train.py           # 核心訓練腳本
│   └── inference.py       # 推論與提交腳本
├── project_specification.md # 專案規格書
└── project_plan.md          # 專案進度與規劃
```

## 2. 開發階段 (Development Phases)

### 階段一：資料與基礎設施準備 (Data & Infrastructure)
- [x] 建立基礎資料夾結構。
- [x] 分析 BirdCLEF 2026 競賽資料集，驗證物種音訊數量與類別不平衡的情況 (EDA)。
- [ ] 開發 `script/preprocess.py`，將原始音訊轉換為 Mel-Spectrogram 並建立為 `processed_data/` 下的 HDF5 資料庫。
- [ ] 實作 PyTorch `Dataset` 與 `DataLoader`，針對 HDF5 結構設計高效率讀取邏輯、固定長度裁切。
- [ ] 設定驗證集的分割策略 (例如：Stratified K-Fold 以確保各類別比例一致)。

### 階段二：EfficientNet 模型開發 (Model 1)
- [ ] 透過 `timm` 等套件載入預訓練的 EfficientNet 模型 (如 efficientnet_b0 或 b1)。
- [ ] 調整輸入層通道數以適應單通道頻譜圖。
- [ ] 實作模型訓練管線 (包含 Loss function, Optimizer, Learning Rate Scheduler)，並於此階段實作與加入 MixUp 資料擴增。
- [ ] 完成 EfficientNet Baseline 訓練並檢查成效。

### 階段三：PANNs 模型開發 (Model 2)
- [ ] 下載並整合 PANNs 預訓練模型 (例如 CNN14)。
- [ ] 微調 (Fine-tune) 模型最後的分類全連接層，以符合本競賽的鳥類類別數量。
- [ ] 確保於訓練管線中套用 MixUp 資料擴增。
- [ ] 完成 PANNs 獨立訓練並比較優劣。

### 階段四：模型融合與最終推論管線 (Ensemble & Inference)
- [ ] 開發 `ensemble.py`，匯入已訓練好的兩種模型權重。
- [ ] 實作驗證集上的搜尋演算法，找出兩種模型的最佳機率加權權重 (Weight search)。
- [ ] 撰寫最終的 `inference.py`，確保推論流程流暢並符合競賽的時間與格式限制。

## 3. 專案時程與進度規劃 (Project Schedule)

本專案預計為期 **4 週**，採用敏捷開發迭代，每週重點目標如下：

| 週次 (Week) | 重點任務 (Key Tasks) | 預期產出 (Deliverables) |
| :--- | :--- | :--- |
| **Week 1** | **環境建置與資料前處理**<br>- 探索性資料分析 (分析物種數量與類別不平衡)<br>- 建立專案目錄結構<br>- 實作 `preprocess.py` 將音訊轉換為 HDF5<br>- 實作 PyTorch Dataset 讀取 HDF5<br>- 設定交叉驗證策略 | - 資料不平衡分析檢查點<br>- 完整可運行的 HDF5 資料庫<br>- `dataset.py` 高效讀取管線初版 |
| **Week 2** | **Model 1 (EfficientNet) 開發與訓練**<br>- 引入 `timm` 的預訓練 EfficientNet<br>- 修改輸入層以適應單通道頻譜圖<br>- 於訓練階段加入 MixUp 擴增<br>- 完成單一模型的訓練管線 | - `models/efficientnet.py` 與 `train.py`<br>- 單一模型的本地驗證分數與權重檔 |
| **Week 3** | **Model 2 (PANNs) 開發與優化**<br>- 整合並微調 PANNs 預訓練模型<br>- 針對音訊序列調整網路架構<br>- 完成 PANNs 的獨立訓練 | - `models/panns.py`<br>- PANNs 模型的驗證分數與權重檔 |
| **Week 4** | **模型融合 (Ensemble) 與推論**<br>- 實作雙模型的後期機率融合<br>- 尋找最佳的模型融合權重<br>- 優化與完成推論腳本 | - `models/ensemble.py` 與 `inference.py`<br>- 最終版本程式碼庫與可產生 `submission.csv` 的推論流程 |
