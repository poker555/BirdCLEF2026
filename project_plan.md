# 專案規劃：BirdCLEF 2026 PANNs 開發藍圖

> 最後更新：2026-03-25
> 目前 Kaggle 成績：ROC-AUC **0.87**（銅牌門檻 0.913，金牌 0.937）

## 1. 專案目錄結構（實際）

```text
BirdCLEF2026/
├── train_audio/                    # 訓練音訊（206 物種，.ogg）
├── train_soundscapes/              # 訓練用環境音景
├── test_soundscapes/               # 測試音景（Kaggle 隱藏）
├── processed_data/
│   ├── train_waveforms.h5          # ✅ PANNs 用原始波形 HDF5
│   └── train_spectrograms.h5       # ✅ 舊版 Mel 頻譜 HDF5（已不使用）
├── models/
│   ├── best_panns_model.pth        # ✅ 目前最佳模型權重
│   ├── panns_train_result.json     # ✅ 訓練結果（含 kaggle_roc_auc）
│   └── per_class_analysis.csv      # ✅ 每物種 ROC-AUC / AP 分析
├── script/
│   ├── preprocess_waveform.py      # ✅ 波形 HDF5 前處理（含人聲靜音、稀少物種增強）
│   ├── panns/
│   │   ├── model.py                # ✅ PANNsCNN10（含輔助分類頭）
│   │   ├── dataset.py              # ✅ 波形 Dataset（thread-local HDF5）
│   │   └── train.py                # ❌ 舊版，已由 train_panns.py 取代
│   ├── train_panns.py              # ✅ 主訓練腳本（WeightedSampler、MixUp、噪音混入）
│   ├── inference_panns.py          # ✅ 推論腳本（TTA x3）
│   ├── noise_dataset.py            # ✅ 低能量噪音片段抽取
│   ├── soundscape_dataset.py       # ✅ Soundscape 波形/Mel Dataset
│   ├── analyze_per_class.py        # ✅ 每物種弱點分析
│   ├── run_pipeline.py             # ✅ 一鍵執行（純 PANNs 流程）
│   └── label_encoder.py            # ✅ 標籤編碼工具
├── experiment_log.csv              # ✅ 實驗紀錄
├── project_plan.md
└── project_specification.md
```

---

## 2. 開發階段進度

### 階段一：資料與基礎設施 ✅ 完成
- [x] 建立專案目錄結構
- [x] EDA：分析物種數量與類別不平衡（206 物種，嚴重長尾分佈）
- [x] `preprocess_waveform.py`：原始波形 → HDF5（含人聲靜音、稀少物種離線增強 ×4）
- [x] `panns/dataset.py`：thread-local HDF5 讀取、動態多片段切割
- [x] 驗證集策略：稀少物種（≤5筆）全進訓練集，一般物種 stratified 80/20

### 階段二：EfficientNet 模型 ❌ 已捨棄
- [~] ~~EfficientNet 模型開發~~（決定專注 PANNs，不實施）
- [~] ~~雙模型 Ensemble~~（暫不實施，單模型先衝分數）

### 階段三：PANNs 模型開發 ✅ 完成（持續優化中）
- [x] PANNsCNN10 架構（4 層 ConvBlock + 輔助分類頭）
- [x] 訓練管線：BCEWithLogitsLoss + CrossEntropyLoss（aux weight=0.2）
- [x] MixUp（alpha=0.4，prob=0.7）
- [x] WeightedRandomSampler（quality_weight / sqrt(species_count)）
- [x] Soundscape 資料整合（train/val 分流）
- [x] 噪音混入增強（p=0.5，SNR 5~20dB，來源：train_soundscapes 低能量片段）
- [x] 頻率範圍調整：f_min=50Hz，f_max=16000Hz
- [x] 訓練至 epoch 94（early stopping），驗證 mAP 0.647，Kaggle ROC-AUC 0.87

### 階段四：推論與分析 ✅ 完成（持續優化中）
- [x] `inference_panns.py`：TTA ×3（原始 / +6dB / 時間翻轉）
- [x] `analyze_per_class.py`：每物種 ROC-AUC / AP 分析
- [x] 實驗紀錄系統（experiment_log.csv）
- [x] git tag `v1.1-roc087`

---

## 3. 下一步計畫（優先順序）

### 🔜 短期（下一輪訓練）

| 優先 | 項目 | 說明 |
| :--: | :--- | :--- |
| 1 | **Mel 頻帶數 128 → 160** | 提升頻率解析度，對鳥鳴細節有幫助；n_fft=1024 時可安全使用 |
| 2 | **SpecAugment（頻率/時間遮罩）** | 防止過擬合，實作簡單，在訓練迴圈中動態套用，不需重新前處理 |
| 3 | **效能檢查** | 量測每個 epoch 的實際訓練時間、GPU 使用率、DataLoader 瓶頸，確保硬體資源被充分利用 |
| 4 | **PCEN（Per-Channel Energy Normalization）** | 取代 AmplitudeToDB，對背景噪音有更好的抑制效果，在野外錄音場景特別有效；替換 `amplitude_to_db` 層即可，不需重新前處理 |

### 📋 中期（評估後決定）

- 針對弱點物種加強：Rufous Cacholote、Plain Inezia、White-wedged Piculet（目前驗證 ROC-AUC < 0.89）
- 更多 TTA 策略（pitch shift、頻率遮罩）
- 學習率 warmup（cosine schedule）

### 🔮 長期（視成績決定）

- 升級至 CNN14 或更大模型
- 雙模型 Ensemble（若單模型遇到瓶頸）

---

## 4. 實驗紀錄摘要

| run_id | 模型 | val mAP | Kaggle ROC-AUC | 備註 |
| :--- | :--- | :---: | :---: | :--- |
| 2026-03-24 22:57 | PANNs CNN10 | 0.647 | **0.87** | baseline，epoch 94 |
