# BirdCLEF 2026

[繁體中文](#繁體中文) | [English](#english)

---

## 繁體中文

鳥類音訊辨識專案，基於 PANNs CNN10 架構，參加 [Kaggle BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) 競賽。

### 安裝套件

```bash
pip install -r requirements.txt
```

> PyTorch 建議依照你的 CUDA 版本從 [pytorch.org](https://pytorch.org/get-started/locally/) 安裝，例如：
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 準備資料

從 Kaggle 下載競賽資料並放到專案根目錄：

```
BirdCLEF2026/
├── train.csv
├── taxonomy.csv
├── train_audio/
└── test_soundscapes/
```

接著產生必要的輔助 CSV：

```bash
# 產生 taxonomy_encoded.csv（label_id / class_id 對照表）
python script/label_encoder.py

# 產生 species_counts.csv（各物種音訊數量統計）
python script/eda_duration.py
```

人聲過濾（可選，需要 Silero VAD）：

```bash
python script/vad/batch_detect_voice.py
```

### 執行訓練流程

```bash
# 完整流程（前處理 + 訓練）
python script/run_pipeline.py

# 跳過前處理（HDF5 已存在時）
python script/run_pipeline.py --skip-preprocess

# 只跑前處理
python script/run_pipeline.py --only preprocess

# 只跑訓練
python script/run_pipeline.py --only panns

# 快速測試（30 筆資料、2 epoch，驗證環境是否正常）
python script/run_pipeline.py --test
```

### 推論

```bash
python script/inference_panns.py
```

輸出 `submission.csv`，可直接上傳 Kaggle。

### 調整訓練參數

所有訓練參數、路徑與技術開關都集中在 `script/train_panns.py` 最前面的設定區：

```python
# ── 路徑（自動推算，通常不需修改）──────────────────────────
BASE_DIR    = Path(...)          # 專案根目錄
H5_PATH     = BASE_DIR / "processed_data/train_waveforms.h5"
ESC50_DIR   = BASE_DIR / "ESC-50-master/audio"  # 不存在時自動下載

# ── 基本訓練參數 ────────────────────────────────────────────
EPOCHS         = 100
LR             = 1e-3
BATCH_SIZE_GPU = 128

# ── 技術開關 ────────────────────────────────────────────────
USE_MIXUP            = True
USE_NOISE_AUG        = True   # ESC-50 噪音混入，首次訓練自動下載（約 600MB）
USE_SPEC_AUGMENT     = True
USE_LABEL_SMOOTHING  = True
USE_GRAD_CLIP        = True
USE_WEIGHTED_SAMPLER = True
LR_SCHEDULER         = 'cosine'  # 'cosine' 或 'plateau'
```

前處理的路徑設定在 `script/preprocess_waveform.py` 最前面，同樣集中管理。

### 專案結構

```
script/
├── preprocess_waveform.py     前處理：音訊 → HDF5 波形
├── train_panns.py             PANNs 訓練主腳本
├── inference_panns.py         推論腳本
├── run_pipeline.py            一鍵執行流程
├── analyze_per_class.py       訓練後 per-class 弱點分析
├── compare_models.py          比較兩個模型
├── label_encoder.py           產生 taxonomy_encoded.csv
├── noise_dataset.py           ESC-50 噪音資料集（自動下載）
├── soundscape_dataset.py      Soundscape 資料集
├── panns/
│   ├── model.py               PANNs CNN10 模型
│   └── dataset.py             HDF5 波形資料集
└── vad/
    └── batch_detect_voice.py  批次人聲偵測
```

訓練完成的模型儲存在 `models/`，以 NATO 字母命名（alpha、bravo、charlie...）。每次訓練結果記錄於 `experiment_log.csv`。

---

## English

Bird audio recognition project based on the PANNs CNN10 architecture, built for the [Kaggle BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) competition.

### Installation

```bash
pip install -r requirements.txt
```

> For PyTorch, install according to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### Data Preparation

Download the competition data from Kaggle and place it in the project root:

```
BirdCLEF2026/
├── train.csv
├── taxonomy.csv
├── train_audio/
└── test_soundscapes/
```

Then generate the required auxiliary CSV files:

```bash
# Generate taxonomy_encoded.csv (label_id / class_id mapping)
python script/label_encoder.py

# Generate species_counts.csv (per-species audio count)
python script/eda_duration.py
```

Voice filtering (optional, requires Silero VAD):

```bash
python script/vad/batch_detect_voice.py
```

### Training

```bash
# Full pipeline (preprocessing + training)
python script/run_pipeline.py

# Skip preprocessing (when HDF5 already exists)
python script/run_pipeline.py --skip-preprocess

# Preprocessing only
python script/run_pipeline.py --only preprocess

# Training only
python script/run_pipeline.py --only panns

# Quick test (30 samples, 2 epochs — verifies the environment works)
python script/run_pipeline.py --test
```

### Inference

```bash
python script/inference_panns.py
```

Outputs `submission.csv`, ready to upload to Kaggle.

### Configuring Training

All paths, training parameters and feature toggles are at the top of `script/train_panns.py`:

```python
# ── Paths (auto-resolved, usually no need to change) ────────
BASE_DIR    = Path(...)          # Project root
H5_PATH     = BASE_DIR / "processed_data/train_waveforms.h5"
ESC50_DIR   = BASE_DIR / "ESC-50-master/audio"  # Auto-downloaded if missing

# ── Training parameters ──────────────────────────────────────
EPOCHS         = 100
LR             = 1e-3
BATCH_SIZE_GPU = 128

# ── Feature toggles ──────────────────────────────────────────
USE_MIXUP            = True
USE_NOISE_AUG        = True   # ESC-50 noise aug, auto-downloaded on first run (~600MB)
USE_SPEC_AUGMENT     = True
USE_LABEL_SMOOTHING  = True
USE_GRAD_CLIP        = True
USE_WEIGHTED_SAMPLER = True
LR_SCHEDULER         = 'cosine'  # 'cosine' or 'plateau'
```

Preprocessing paths are similarly centralized at the top of `script/preprocess_waveform.py`.

### Project Structure

```
script/
├── preprocess_waveform.py     Preprocessing: audio → HDF5 waveforms
├── train_panns.py             PANNs training script
├── inference_panns.py         Inference script
├── run_pipeline.py            One-command pipeline runner
├── analyze_per_class.py       Per-class weakness analysis
├── compare_models.py          Compare two trained models
├── label_encoder.py           Generate taxonomy_encoded.csv
├── noise_dataset.py           ESC-50 noise dataset (auto-download)
├── soundscape_dataset.py      Soundscape dataset
├── panns/
│   ├── model.py               PANNs CNN10 model
│   └── dataset.py             HDF5 waveform dataset
└── vad/
    └── batch_detect_voice.py  Batch voice activity detection
```

Trained models are saved in `models/` using NATO phonetic alphabet names (alpha, bravo, charlie...). Each training run is logged in `experiment_log.csv`.
