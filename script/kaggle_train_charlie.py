# ============================================================
# BirdCLEF 2026 - Kaggle Training Notebook (Charlie)
# 複製此檔案內容到 Kaggle Notebook 的單一 Code Cell 執行
#
# 與 Bravo 的差異（噪音混入改進版）：
#   1. RMS 門檻放寬：0.01 → 0.03（收集更多噪音片段）
#   2. 每支音訊最多取 5 個片段（原本 3 個）
#   3. SNR 範圍調高：5~20dB → 10~30dB（噪音更輕微）
#   4. 噪音混入移到 MixUp 之後（避免雙重模糊）
#   5. 噪音混入機率降低：0.5 → 0.3
#
# Kaggle Dataset 設定：
#   - 競賽資料：birdclef-2026
#   - HDF5：上傳 processed_data/train_waveforms.h5 為 Dataset
#   - CSV：train.csv / taxonomy_encoded.csv / species_counts.csv
# ============================================================

import os, sys, warnings, json, shutil, threading
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

# ── 路徑設定（Kaggle 環境）────────────────────────────────────
BASE          = Path('/kaggle/working')
DATA          = Path('/kaggle/input/birdclef-2026')
# HDF5 Dataset 名稱請依你上傳的 dataset 名稱調整
H5_DATASET    = Path('/kaggle/input/birdclef2026-waveforms')
H5_PATH       = H5_DATASET / 'train_waveforms.h5'
TRAIN_CSV     = DATA / 'train.csv'
TAXONOMY_CSV  = DATA / 'taxonomy_encoded.csv'   # 若競賽資料沒有，請上傳為額外 dataset
SPECIES_CSV   = DATA / 'species_counts.csv'
SOUNDSCAPE_DIR= DATA / 'train_soundscapes'
MODEL_OUT     = BASE / 'model_charlie.pth'
RESULT_OUT    = BASE / 'charlie_train_result.json'

NUM_CLASSES   = 234
CHUNK_LENGTH  = 160000   # 5s @ 32kHz
SAMPLE_RATE   = 32000

# ── NoiseDataset（改進版）────────────────────────────────────
class NoiseDataset:
    """
    改進點：
    - RMS 門檻 0.01 → 0.03（收集更多片段）
    - 每支最多取 5 個片段（原本 3 個）
    """
    RMS_THRESHOLD = 0.03
    MAX_PER_FILE  = 5

    def __init__(self, soundscape_dir, chunk_length=160000, max_files=500, rng_seed=42):
        import librosa
        self.chunk_length = chunk_length
        self.rng    = np.random.default_rng(rng_seed)
        self.chunks = []
        files = sorted(Path(soundscape_dir).glob('*.ogg'))[:max_files]
        for fp in files:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    y, _ = librosa.load(str(fp), sr=SAMPLE_RATE, mono=True, duration=60.0)
            except Exception:
                continue
            n = min(len(y) // chunk_length, self.MAX_PER_FILE)
            for i in range(n):
                chunk = y[i * chunk_length: (i + 1) * chunk_length]
                if np.sqrt(np.mean(chunk ** 2)) < self.RMS_THRESHOLD:
                    self.chunks.append(chunk.astype(np.float32))
        print(f"[NoiseDataset] 收集到 {len(self.chunks)} 個低能量噪音片段")

    def sample(self):
        if not self.chunks:
            return np.zeros(self.chunk_length, dtype=np.float32)
        return self.chunks[self.rng.integers(len(self.chunks))].copy()

    def __len__(self):
        return len(self.chunks)


# ── PANNsDataset（thread-local HDF5）────────────────────────
_tl = threading.local()

def _get_h5(path):
    f = getattr(_tl, 'file', None)
    if f is None or not f.id.valid:
        _tl.file = h5py.File(path, 'r')
    return _tl.file

class PANNsDataset(Dataset):
    def __init__(self, h5_path, keys, chunk_length=160000):
        self.h5_path      = h5_path
        self.chunk_length = chunk_length
        self.samples      = []
        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                n_seg = max(1, f[key].shape[0] // chunk_length)
                for seg_idx in range(n_seg):
                    self.samples.append((key, seg_idx, f[key].shape[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, seg_idx, total = self.samples[idx]
        f  = _get_h5(self.h5_path)
        ds = f[key]
        soft_label = ds.attrs['soft_label'].astype(np.float32)
        class_id   = int(ds.attrs['class_id'])
        if total <= self.chunk_length:
            wav = ds[:]
        else:
            max_start = total - self.chunk_length
            seg_start = seg_idx * self.chunk_length
            seg_end   = min(seg_start + self.chunk_length, max_start)
            start     = int(np.random.randint(seg_start, seg_end + 1))
            wav       = ds[start: start + self.chunk_length]
        if len(wav) < self.chunk_length:
            wav = np.pad(wav, (0, self.chunk_length - len(wav)))
        return (torch.tensor(wav, dtype=torch.float32),
                torch.tensor(soft_label, dtype=torch.float32),
                torch.tensor(class_id, dtype=torch.long))


# ── PANNsCNN10 模型────────────────────────────────────────────
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        init_layer(self.conv1); init_layer(self.conv2)

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
            n_mels=mel_bins, f_min=50, f_max=16000)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.bn0         = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1,   64)
        self.conv_block2 = ConvBlock(64,  128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1         = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.fc_class    = nn.Linear(512, num_groups, bias=True)
        init_layer(self.fc1); init_layer(self.fc_audioset); init_layer(self.fc_class)

    def forward(self, x):
        x = self.amplitude_to_db(self.mel_spectrogram(x))
        x = x.unsqueeze(2); x = self.bn0(x); x = x.squeeze(2)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv_block1(x); x = self.conv_block2(x)
        x = self.conv_block3(x); x = self.conv_block4(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        logits = self.fc_audioset(x)
        if self.training:
            return logits, self.fc_class(x)
        return logits


# ── 工具函式 ──────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def add_noise(waveforms, noise_ds, prob=0.3, snr_db_range=(10, 30)):
    """
    改進版：prob 0.5→0.3，SNR 5~20dB → 10~30dB（噪音更輕微）
    且在 MixUp 之後呼叫，避免雙重模糊。
    """
    if noise_ds is None or len(noise_ds) == 0:
        return waveforms
    result = waveforms.clone()
    for i in range(result.shape[0]):
        if torch.rand(1).item() > prob:
            continue
        noise = torch.tensor(noise_ds.sample(), dtype=torch.float32).to(waveforms.device)
        if noise.shape[0] < result.shape[1]:
            noise = noise.repeat(result.shape[1] // noise.shape[0] + 1)
        noise    = noise[:result.shape[1]]
        snr_db   = torch.empty(1).uniform_(*snr_db_range).item()
        sig_rms  = result[i].pow(2).mean().sqrt().clamp(min=1e-9)
        noi_rms  = noise.pow(2).mean().sqrt().clamp(min=1e-9)
        scale    = sig_rms / (noi_rms * (10 ** (snr_db / 20)))
        result[i] = (result[i] + noise * scale).clamp(-1.0, 1.0)
    return result


# ── 資料切分與 Sampler ────────────────────────────────────────
AUG_SUFFIXES = ('_ts09', '_ts11', '_ps+1', '_ps-1')

def build_split(h5_path, sc_df, rare_threshold=5):
    rare_species = set(sc_df[sc_df['audio_count'] <= rare_threshold]['primary_label'].astype(str))
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
        normal_orig, test_size=0.2, random_state=42, stratify=normal_labels)
    aug_keys   = [k for k in all_keys if any(k.endswith(s) for s in AUG_SUFFIXES)]
    return rare_train + normal_train + aug_keys, normal_val

def build_sampler(h5_path, keys_train, sc_df, train_df=None):
    count_map  = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))
    rating_map = {}
    if train_df is not None:
        for _, row in train_df.iterrows():
            rating_map[str(row['filename'])] = float(row.get('rating', 0))
    weights = []
    with h5py.File(str(h5_path), 'r') as f:
        for key in keys_train:
            species = key.split('/')[0]
            cnt = count_map.get(species, 1)
            w   = 1.0 / np.sqrt(cnt)
            base_fname = key
            for suf in AUG_SUFFIXES:
                if key.endswith(suf):
                    base_fname = key[:-len(suf)]; break
            rating = rating_map.get(base_fname, 0)
            fname_stem = base_fname.split('/')[-1]
            if 'XC' in fname_stem and rating > 0:
                w *= 0.5 + (rating / 5.0) * 0.5
            try:
                n_seg = max(1, f[key].shape[0] // CHUNK_LENGTH)
            except Exception:
                n_seg = 1
            weights.extend([w] * n_seg)
    return weights


# ── 主訓練流程 ────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"裝置：{device}")

    sc_df    = pd.read_csv(SPECIES_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    print("建立資料切分...")
    keys_train, keys_val = build_split(H5_PATH, sc_df)
    print(f"訓練 {len(keys_train)} 筆，驗證 {len(keys_val)} 筆")

    train_ds = PANNsDataset(str(H5_PATH), keys_train)
    val_ds   = PANNsDataset(str(H5_PATH), keys_val)

    # Soundscape 資料（若有 labels CSV）
    labels_csv = DATA / 'train_soundscapes_labels.csv'
    sc_train_size = 0
    if SOUNDSCAPE_DIR.exists() and labels_csv.exists():
        from soundscape_dataset import SoundscapeWaveformDataset  # 若有上傳腳本
        # 若沒有上傳腳本，此區塊可略過，soundscape 資料不加入
        try:
            sc_train_ds = SoundscapeWaveformDataset(
                str(SOUNDSCAPE_DIR), str(labels_csv), str(TAXONOMY_CSV),
                split='train', sc_df=sc_df)
            sc_val_ds = SoundscapeWaveformDataset(
                str(SOUNDSCAPE_DIR), str(labels_csv), str(TAXONOMY_CSV),
                split='val', sc_df=sc_df)
            sc_train_size = len(sc_train_ds)
            from torch.utils.data import ConcatDataset
            train_ds = ConcatDataset([train_ds, sc_train_ds])
            val_ds   = ConcatDataset([val_ds,   sc_val_ds])
            print(f"Soundscape 訓練 {sc_train_size} 筆，驗證 {len(sc_val_ds)} 筆")
        except Exception as e:
            print(f"Soundscape 略過：{e}")

    seg_weights = build_sampler(H5_PATH, keys_train, sc_df, train_df=train_df)
    sampler     = WeightedRandomSampler(seg_weights, len(seg_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 噪音資料集（改進版參數）
    noise_ds = None
    if SOUNDSCAPE_DIR.exists():
        noise_ds = NoiseDataset(str(SOUNDSCAPE_DIR), chunk_length=CHUNK_LENGTH, max_files=500)
        if len(noise_ds) == 0:
            print("⚠️  未找到噪音片段，跳過噪音混入")
            noise_ds = None

    model             = PANNsCNN10(classes_num=NUM_CLASSES).to(device)
    criterion_species = nn.BCEWithLogitsLoss()
    criterion_class   = nn.CrossEntropyLoss()
    optimizer         = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler         = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='max', factor=0.5, patience=2)
    scaler            = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    epochs, patience  = 100, 5
    best_f1 = best_map = 0.0
    counter = 0
    print("開始訓練 Charlie...")

    for epoch in range(epochs):
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
                # MixUp 先做
                if torch.rand(1).item() > 0.3:
                    mixed_wav, labels_a, labels_b, lam = mixup_data(waveforms, soft_labels)
                    # 噪音混入在 MixUp 之後（改進點）
                    mixed_wav    = add_noise(mixed_wav, noise_ds, prob=0.3, snr_db_range=(10, 30))
                    logits_sp, logits_cl = model(mixed_wav)
                    loss_species = criterion_species(logits_sp, lam * labels_a + (1 - lam) * labels_b)
                else:
                    waveforms    = add_noise(waveforms, noise_ds, prob=0.3, snr_db_range=(10, 30))
                    logits_sp, logits_cl = model(waveforms)
                    loss_species = criterion_species(logits_sp, soft_labels)
                loss = loss_species + 0.2 * criterion_class(logits_cl, class_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch[{epoch+1}/{epochs}] Batch[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} 平均 Loss: {total_loss/len(train_loader):.4f}")

        # ── 驗證 ──────────────────────────────────────────────
        model.eval()
        n_val = len(val_ds)
        all_probs   = np.empty((n_val, NUM_CLASSES), dtype=np.float32)
        all_preds   = np.empty(n_val, dtype=np.int64)
        all_targets = np.empty(n_val, dtype=np.int64)
        ptr = 0

        with torch.no_grad():
            for waveforms, soft_labels, _ in val_loader:
                waveforms = waveforms.to(device)
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                        dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                    outputs = model(waveforms)
                probs = torch.sigmoid(outputs).cpu().float().numpy()
                b = probs.shape[0]
                all_probs[ptr:ptr+b]   = probs
                all_preds[ptr:ptr+b]   = outputs.argmax(dim=1).cpu().numpy()
                all_targets[ptr:ptr+b] = soft_labels.argmax(dim=1).numpy()
                ptr += b

        val_f1 = f1_score(all_targets[:ptr], all_preds[:ptr],
                          average='macro', zero_division=0)
        targets_oh = np.zeros_like(all_probs[:ptr])
        for i, t in enumerate(all_targets[:ptr]):
            targets_oh[i, t] = 1.0
        present = np.where(targets_oh.sum(axis=0) > 0)[0]
        val_map = average_precision_score(
            targets_oh[:, present], all_probs[:ptr, present], average='macro'
        ) if len(present) > 0 else 0.0

        print(f"Epoch {epoch+1} | F1: {val_f1:.4f} | mAP: {val_map:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_map)

        if val_map > best_map:
            best_map, best_f1, counter = val_map, val_f1, 0
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"==> 最佳模型已儲存：{MODEL_OUT}")
        else:
            counter += 1
            print(f"mAP 未提升，累積 {counter}/{patience}")
            if counter >= patience:
                print("==> Early Stopping")
                break

    # ── 結果輸出 ──────────────────────────────────────────────
    result = {
        'model_name':        'charlie',
        'model':             'panns',
        'best_f1':           round(best_f1, 6),
        'best_map':          round(best_map, 6),
        'kaggle_roc_auc':    '',
        'epochs_trained':    epoch + 1,
        'noise_rms_threshold': 0.03,
        'noise_max_per_file':  5,
        'noise_prob':          0.3,
        'noise_snr_range':     '10-30dB',
        'noise_after_mixup':   True,
        'notes': 'charlie: noise augmentation improved (rms=0.03, snr=10-30dB, prob=0.3, after mixup)',
    }
    with open(RESULT_OUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n訓練完成！結果：{RESULT_OUT}")
    print(f"best_f1={best_f1:.4f}  best_map={best_map:.4f}  epochs={epoch+1}")


main()
