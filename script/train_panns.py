import os
import sys
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from panns.dataset import PANNsDataset
from panns.model import PANNsCNN10
from soundscape_dataset import SoundscapeWaveformDataset
from noise_dataset import NoiseDataset

import h5py
from sklearn.model_selection import train_test_split

# ==============================================================================
# 路徑與參數設定 — 所有可調整的設定都在這裡
# ==============================================================================

# 專案根目錄（自動推算，不需手動修改）
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

# ── 輸入路徑 ───────────────────────────────────────────────────
H5_PATH        = BASE_DIR / "processed_data" / "train_waveforms.h5"
SC_CSV         = BASE_DIR / "species_counts.csv"
TRAIN_CSV      = BASE_DIR / "train.csv"
TAXONOMY_CSV   = BASE_DIR / "taxonomy_encoded.csv"
SOUNDSCAPE_DIR = BASE_DIR / "train_soundscapes"
LABELS_CSV     = BASE_DIR / "train_soundscapes_labels.csv"

# ── 輸出路徑 ───────────────────────────────────────────────────
MODELS_DIR      = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_panns_model.pth"
RESULT_JSON     = MODELS_DIR / "panns_train_result.json"

# ── ESC-50 噪音資料集路徑（不存在時自動下載）─────────────────
ESC50_DIR = BASE_DIR / "ESC-50-master" / "audio"

# ── 基本訓練參數 ───────────────────────────────────────────────
EPOCHS         = 200
PATIENCE       = 7       # Early Stopping patience（以 mAP 為準）
NUM_CLASSES    = 234
CHUNK_LENGTH   = 160000  # 5 秒 @ 32kHz
RARE_THRESHOLD = 5       # 稀少物種門檻，全進訓練集
LR             = 1e-3
WEIGHT_DECAY   = 1e-4

# ── 自動偵測 num_workers（CPU 核心數 - 1，最少 0）─────────────
NUM_WORKERS = max(0, multiprocessing.cpu_count() - 1)

# ── 自動偵測 batch_size（依 VRAM 大小）────────────────────────
# 手動覆蓋：將 BATCH_SIZE 設為非 None 的整數即可
BATCH_SIZE = None  # None = 自動偵測

def _auto_batch_size() -> int:
    if BATCH_SIZE is not None:
        return BATCH_SIZE
    if not torch.cuda.is_available():
        return 32
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram_gb >= 40:
        return 256
    elif vram_gb >= 24:
        return 128
    elif vram_gb >= 16:
        return 96
    elif vram_gb >= 12:
        return 64
    elif vram_gb >= 8:
        return 48
    else:
        return 32

# ── 損失函數 ───────────────────────────────────────────────────
AUX_LOSS_WEIGHT = 0.2

# ── Label Smoothing ────────────────────────────────────────────
USE_LABEL_SMOOTHING = False   # 關閉：BCE + soft label 已足夠，雙重軟化反而模糊
LABEL_SMOOTHING     = 0.05

# ── MixUp ─────────────────────────────────────────────────────
USE_MIXUP   = True
MIXUP_ALPHA = 0.4
MIXUP_PROB  = 0.7

# ── 噪音混入（ESC-50）─────────────────────────────────────────
USE_NOISE_AUG   = True
NOISE_PROB      = 0.5
NOISE_SNR_RANGE = (5, 20)

# ── SpecAugment（模型內部）────────────────────────────────────
USE_SPEC_AUGMENT  = True
SPEC_FREQ_MASK    = 10   # 頻率遮罩最大寬度（原本 20，調小保留更多頻率資訊）
SPEC_TIME_MASK    = 20   # 時間遮罩最大寬度（原本 40，調小保留更多時間資訊）

# ── Gradient Clipping ─────────────────────────────────────────
USE_GRAD_CLIP  = True
GRAD_CLIP_NORM = 5.0

# ── LR Scheduler：'cosine' 或 'plateau' ───────────────────────
LR_SCHEDULER     = 'cosine'
COSINE_ETA_MIN   = 1e-6
PLATEAU_FACTOR   = 0.5
PLATEAU_PATIENCE = 2

# ── WeightedRandomSampler ─────────────────────────────────────
USE_WEIGHTED_SAMPLER = True

# ── Soundscape 資料集 ─────────────────────────────────────────
USE_SOUNDSCAPE = True

# ── 資料切分 ───────────────────────────────────────────────────
VAL_SPLIT   = 0.2
RANDOM_SEED = 42

# ==============================================================================
# 以下為程式邏輯，一般不需修改
# ==============================================================================


def add_noise(waveforms: torch.Tensor, noise_ds: 'NoiseDataset',
              prob: float = 0.5, snr_db_range=(5, 20)) -> torch.Tensor:
    if noise_ds is None or len(noise_ds) == 0:
        return waveforms
    result = waveforms.clone()
    for i in range(result.shape[0]):
        if torch.rand(1).item() > prob:
            continue
        noise = torch.tensor(noise_ds.sample(), dtype=torch.float32).to(waveforms.device)
        if noise.shape[0] < result.shape[1]:
            noise = noise.repeat(result.shape[1] // noise.shape[0] + 1)
        noise   = noise[:result.shape[1]]
        snr_db  = torch.empty(1).uniform_(*snr_db_range).item()
        sig_rms = result[i].pow(2).mean().sqrt().clamp(min=1e-9)
        noi_rms = noise.pow(2).mean().sqrt().clamp(min=1e-9)
        scale   = sig_rms / (noi_rms * (10 ** (snr_db / 20)))
        result[i] = (result[i] + noise * scale).clamp(-1.0, 1.0)
    return result


def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def build_split(h5_path: Path, sc_df: pd.DataFrame, rare_threshold: int = RARE_THRESHOLD):
    rare_species = set(sc_df[sc_df['audio_count'] <= rare_threshold]['primary_label'].astype(str))
    AUG_SUFFIXES = ('_ts09', '_ts11', '_ps+1', '_ps-1')

    all_keys, orig_keys, orig_labels = [], [], []
    with h5py.File(str(h5_path), 'r') as f:
        for group in f.keys():
            for dset in f[group].keys():
                full_key = f"{group}/{dset}"
                all_keys.append(full_key)
                if not any(full_key.endswith(s) for s in AUG_SUFFIXES):
                    orig_keys.append(full_key)
                    orig_labels.append(group)

    rare_train, normal_orig, normal_labels = [], [], []
    for key, label in zip(orig_keys, orig_labels):
        if label in rare_species:
            rare_train.append(key)
        else:
            normal_orig.append(key)
            normal_labels.append(label)

    normal_train, normal_val = train_test_split(
        normal_orig, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=normal_labels
    )
    aug_keys = [k for k in all_keys if any(k.endswith(s) for s in AUG_SUFFIXES)]
    return rare_train + normal_train + aug_keys, normal_val


def build_sampler(h5_path: Path, keys_train: list, sc_df: pd.DataFrame,
                  train_df: pd.DataFrame = None):
    count_map  = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))
    rating_map = {}
    if train_df is not None:
        for _, row in train_df.iterrows():
            rating_map[str(row['filename'])] = float(row.get('rating', 0))

    AUG_SUFFIXES    = ('_ts09', '_ts11', '_ps+1', '_ps-1')
    segment_weights = []

    with h5py.File(str(h5_path), 'r') as f:
        for key in keys_train:
            species = key.split('/')[0]
            cnt = count_map.get(species, 1)
            w   = 1.0 / np.sqrt(cnt)

            base_fname = key
            for suf in AUG_SUFFIXES:
                if key.endswith(suf):
                    base_fname = key[:-len(suf)]
                    break
            rating     = rating_map.get(base_fname, 0)
            fname_stem = base_fname.split('/')[-1] if '/' in base_fname else base_fname
            if 'XC' in fname_stem and rating > 0:
                w *= 0.5 + (rating / 5.0) * 0.5

            try:
                n_seg = max(1, f[key].shape[0] // CHUNK_LENGTH)
            except Exception:
                n_seg = 1
            segment_weights.extend([w] * n_seg)

    return segment_weights


def main():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = _auto_batch_size()

    print(f"目前運算裝置: {device}")
    if device.type == 'cuda':
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f} GB")
    print(f"CPU 核心數: {multiprocessing.cpu_count()}  num_workers: {NUM_WORKERS}")
    print(f"batch_size: {batch_size}{'（手動）' if BATCH_SIZE is not None else '（自動偵測）'}")

    print("\n── 技術開關狀態 ──────────────────────────────")
    print(f"  MixUp:             {'ON' if USE_MIXUP else 'OFF'}  (alpha={MIXUP_ALPHA}, prob={MIXUP_PROB})")
    print(f"  Noise Augment:     {'ON' if USE_NOISE_AUG else 'OFF'}  (prob={NOISE_PROB}, SNR={NOISE_SNR_RANGE}dB)")
    print(f"  SpecAugment:       {'ON' if USE_SPEC_AUGMENT else 'OFF'}")
    print(f"  Label Smoothing:   {'ON' if USE_LABEL_SMOOTHING else 'OFF'}")
    print(f"  Grad Clipping:     {'ON' if USE_GRAD_CLIP else 'OFF'}  (max_norm={GRAD_CLIP_NORM})")
    print(f"  LR Scheduler:      {LR_SCHEDULER}")
    print(f"  Weighted Sampler:  {'ON' if USE_WEIGHTED_SAMPLER else 'OFF'}")
    print(f"  Soundscape Data:   {'ON' if USE_SOUNDSCAPE else 'OFF'}")
    print("──────────────────────────────────────────────\n")

    sc_df    = pd.read_csv(SC_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    print("建立訓練/驗證切分...")
    keys_train, keys_val = build_split(H5_PATH, sc_df)
    print(f"訓練集 {len(keys_train)} 筆（含增強），驗證集 {len(keys_val)} 筆")

    train_dataset = PANNsDataset(str(H5_PATH), keys_train, chunk_length=CHUNK_LENGTH)
    val_dataset   = PANNsDataset(str(H5_PATH), keys_val,   chunk_length=CHUNK_LENGTH)

    sc_train_size = 0
    if USE_SOUNDSCAPE and SOUNDSCAPE_DIR.exists() and LABELS_CSV.exists():
        sc_train_dataset = SoundscapeWaveformDataset(
            str(SOUNDSCAPE_DIR), str(LABELS_CSV), str(TAXONOMY_CSV),
            split='train', sc_df=sc_df, rare_threshold=RARE_THRESHOLD
        )
        sc_val_dataset = SoundscapeWaveformDataset(
            str(SOUNDSCAPE_DIR), str(LABELS_CSV), str(TAXONOMY_CSV),
            split='val', sc_df=sc_df, rare_threshold=RARE_THRESHOLD
        )
        sc_train_size = len(sc_train_dataset)
        train_dataset = ConcatDataset([train_dataset, sc_train_dataset])
        val_dataset   = ConcatDataset([val_dataset,   sc_val_dataset])
        print(f"soundscape 訓練集: {sc_train_size} 片段，驗證集: {len(sc_val_dataset)} 片段")

    print(f"訓練集總計 {len(train_dataset)} 個片段，驗證集 {len(val_dataset)} 個片段")

    if USE_WEIGHTED_SAMPLER:
        segment_weights = build_sampler(H5_PATH, keys_train, sc_df, train_df=train_df)
        if sc_train_size > 0:
            count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))
            tax_df    = pd.read_csv(TAXONOMY_CSV)
            id_to_sp  = {int(r['label_id']): str(r['primary_label']) for _, r in tax_df.iterrows()}
            for sample in sc_train_dataset.samples:
                primary_idx = int(np.argmax(sample['soft_label']))
                sp  = id_to_sp.get(primary_idx, '')
                cnt = count_map.get(sp, 1)
                w   = 2.0 if cnt == 0 else 1.0 / np.sqrt(max(cnt, 1))
                segment_weights.append(w)
        sampler      = WeightedRandomSampler(segment_weights, len(segment_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
                                  persistent_workers=(NUM_WORKERS > 0))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
                                  persistent_workers=(NUM_WORKERS > 0))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
                            persistent_workers=(NUM_WORKERS > 0))

    model = PANNsCNN10(classes_num=NUM_CLASSES,
                       spec_freq_mask=SPEC_FREQ_MASK,
                       spec_time_mask=SPEC_TIME_MASK).to(device)
    if not USE_SPEC_AUGMENT:
        model._spec_augment = lambda x, **kw: x

    smoothing         = LABEL_SMOOTHING if USE_LABEL_SMOOTHING else 0.0
    criterion_species = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_class   = nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer         = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if LR_SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=COSINE_ETA_MIN
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
        )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    noise_ds = None
    if USE_NOISE_AUG:
        noise_ds = NoiseDataset(str(ESC50_DIR), chunk_length=CHUNK_LENGTH, auto_download=True)
        if len(noise_ds) == 0:
            print("⚠️  未找到噪音片段，跳過噪音混入增強")
            noise_ds = None

    MODELS_DIR.mkdir(exist_ok=True)
    best_f1  = 0.0
    best_map = 0.0
    counter  = 0
    print("開始 PANNs 模型訓練")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (waveforms, soft_labels, class_labels) in enumerate(train_loader):
            waveforms    = waveforms.to(device)
            soft_labels  = soft_labels.to(device)
            class_labels = class_labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                if USE_MIXUP and torch.rand(1).item() < MIXUP_PROB:
                    mixed_wav, labels_a, labels_b, lam = mixup_data(waveforms, soft_labels, alpha=MIXUP_ALPHA)
                    if USE_NOISE_AUG:
                        mixed_wav = add_noise(mixed_wav, noise_ds, prob=NOISE_PROB, snr_db_range=NOISE_SNR_RANGE)
                    logits_species, logits_class = model(mixed_wav)
                    mixed_soft = lam * labels_a + (1 - lam) * labels_b
                    if USE_LABEL_SMOOTHING:
                        mixed_soft = mixed_soft * (1 - smoothing) + smoothing / mixed_soft.shape[1]
                    loss_species = criterion_species(logits_species, mixed_soft)
                else:
                    if USE_NOISE_AUG:
                        waveforms = add_noise(waveforms, noise_ds, prob=NOISE_PROB, snr_db_range=NOISE_SNR_RANGE)
                    logits_species, logits_class = model(waveforms)
                    soft_labels_s = soft_labels * (1 - smoothing) + smoothing / soft_labels.shape[1] \
                                    if USE_LABEL_SMOOTHING else soft_labels
                    loss_species = criterion_species(logits_species, soft_labels_s)

                loss_class = criterion_class(logits_class, class_labels)
                loss = loss_species + AUX_LOSS_WEIGHT * loss_class

            scaler.scale(loss).backward()
            if USE_GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch[{epoch+1}/{EPOCHS}] Batch[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (sp: {loss_species.item():.4f}, cl: {loss_class.item():.4f})")

        print(f"Epoch {epoch+1} 平均 Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        n_val        = len(val_dataset)
        all_probs_np = np.empty((n_val, NUM_CLASSES), dtype=np.float32)
        all_preds    = np.empty(n_val, dtype=np.int64)
        all_targets  = np.empty(n_val, dtype=np.int64)
        ptr = 0

        with torch.no_grad():
            for waveforms, soft_labels, _ in val_loader:
                waveforms   = waveforms.to(device)
                soft_labels = soft_labels.to(device)
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                        dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                    outputs = model(waveforms)
                probs = torch.sigmoid(outputs).cpu().float().numpy()
                b = probs.shape[0]
                all_probs_np[ptr:ptr + b] = probs
                all_preds[ptr:ptr + b]    = outputs.argmax(dim=1).cpu().float().numpy().astype(np.int64)
                all_targets[ptr:ptr + b]  = soft_labels.argmax(dim=1).cpu().numpy()
                ptr += b

        val_f1 = f1_score(all_targets[:ptr], all_preds[:ptr], average='macro', zero_division=0)

        targets_onehot  = np.zeros_like(all_probs_np[:ptr])
        for i, t in enumerate(all_targets[:ptr]):
            targets_onehot[i, t] = 1.0
        present_classes = np.where(targets_onehot.sum(axis=0) > 0)[0]
        val_map = average_precision_score(
            targets_onehot[:, present_classes],
            all_probs_np[:ptr, present_classes],
            average='macro'
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | F1: {val_f1:.4f} | mAP: {val_map:.4f} | LR: {current_lr:.6f}")

        if LR_SCHEDULER == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_map)

        if val_map > best_map:
            best_map = val_map
            best_f1  = val_f1
            counter  = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"==> 最佳模型已儲存：{BEST_MODEL_PATH.name}")
        else:
            counter += 1
            print(f"mAP 未提升，累積 {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print("==> Early Stopping")
                break

    NATO = [
        'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
        'hotel', 'india', 'juliet', 'kilo', 'lima', 'mike', 'november',
        'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango', 'uniform',
        'victor', 'whiskey', 'xray', 'yankee', 'zulu'
    ]
    existing         = {p.stem.replace('model_', '') for p in MODELS_DIR.glob('model_*.pth')}
    model_codename   = next((n for n in NATO if n not in existing), f'model_{len(existing)}')
    named_model_path = MODELS_DIR / f'model_{model_codename}.pth'
    import shutil
    shutil.copy(BEST_MODEL_PATH, named_model_path)
    print(f"==> 模型已命名並儲存：{named_model_path.name}")

    import json
    result = {
        'model_name':           model_codename,
        'model':                'panns',
        'best_f1':              round(best_f1, 6),
        'best_map':             round(best_map, 6),
        'kaggle_roc_auc':       '',
        'epochs_trained':       epoch + 1,
        'rare_threshold':       RARE_THRESHOLD,
        'use_mixup':            USE_MIXUP,
        'mixup_alpha':          MIXUP_ALPHA,
        'mixup_prob':           MIXUP_PROB,
        'use_noise_aug':        USE_NOISE_AUG,
        'noise_prob':           NOISE_PROB,
        'noise_snr_range':      str(NOISE_SNR_RANGE),
        'use_spec_augment':     USE_SPEC_AUGMENT,
        'spec_freq_mask':       SPEC_FREQ_MASK,
        'spec_time_mask':       SPEC_TIME_MASK,
        'use_label_smoothing':  USE_LABEL_SMOOTHING,
        'label_smoothing':      LABEL_SMOOTHING,
        'use_grad_clip':        USE_GRAD_CLIP,
        'grad_clip_norm':       GRAD_CLIP_NORM,
        'lr_scheduler':         LR_SCHEDULER,
        'use_weighted_sampler': USE_WEIGHTED_SAMPLER,
        'use_soundscape':       USE_SOUNDSCAPE,
        'aux_loss_weight':      AUX_LOSS_WEIGHT,
        'soft_label_weight':    0.3,
        'val_split':            VAL_SPLIT,
        'val_strategy':         'rare_all_train + stratified_80_20 + soundscape',
        'batch_size':           batch_size,
        'num_workers':          NUM_WORKERS,
    }
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"訓練結果已寫入 {RESULT_JSON}")
    print("PANNs 訓練完成")


if __name__ == '__main__':
    main()
