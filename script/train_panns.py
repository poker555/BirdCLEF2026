import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from panns.dataset import PANNsDataset
from panns.model import PANNsCNN10
from soundscape_dataset import SoundscapeWaveformDataset

import h5py
from sklearn.model_selection import train_test_split


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def build_split(h5_path: Path, sc_df: pd.DataFrame, rare_threshold: int = 5):
    """
    稀少物種全進訓練集，一般物種 stratified 80/20 split。
    增強版本（key 含後綴）全進訓練集。
    """
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
        normal_orig, test_size=0.2, random_state=42, stratify=normal_labels
    )
    aug_keys   = [k for k in all_keys if any(k.endswith(s) for s in AUG_SUFFIXES)]
    keys_train = rare_train + normal_train + aug_keys
    keys_val   = normal_val
    return keys_train, keys_val


def build_sampler(h5_path: Path, keys_train: list, sc_df: pd.DataFrame):
    count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))
    CHUNK = 160000
    segment_weights = []
    with h5py.File(str(h5_path), 'r') as f:
        for key in keys_train:
            species = key.split('/')[0]
            cnt = count_map.get(species, 1)
            w = 1.0 / cnt
            try:
                total_samples = f[key].shape[0]
                n_seg = max(1, total_samples // CHUNK)
            except Exception:
                n_seg = 1
            segment_weights.extend([w] * n_seg)
    return segment_weights


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"目前運算裝置: {device}")

    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    h5_path  = base_dir / "processed_data" / "train_waveforms.h5"
    sc_df    = pd.read_csv(base_dir / "species_counts.csv")

    print("建立訓練/驗證切分...")
    keys_train, keys_val = build_split(h5_path, sc_df)
    print(f"訓練集 {len(keys_train)} 筆（含增強），驗證集 {len(keys_val)} 筆")

    train_dataset = PANNsDataset(str(h5_path), keys_train, chunk_length=160000)
    val_dataset   = PANNsDataset(str(h5_path), keys_val,   chunk_length=160000)

    # 加入 soundscape 驗證集
    soundscape_dir = base_dir / 'train_soundscapes'
    labels_csv     = base_dir / 'train_soundscapes_labels.csv'
    taxonomy_csv   = base_dir / 'taxonomy_encoded.csv'
    if soundscape_dir.exists() and labels_csv.exists():
        sc_val_dataset = SoundscapeWaveformDataset(
            str(soundscape_dir), str(labels_csv), str(taxonomy_csv)
        )
        val_dataset = ConcatDataset([val_dataset, sc_val_dataset])
        print(f"加入 soundscape 驗證集，驗證集總計 {len(val_dataset)} 個片段")

    # WeightedRandomSampler
    segment_weights = build_sampler(h5_path, keys_train, sc_df)
    sampler = WeightedRandomSampler(
        weights=segment_weights,
        num_samples=len(segment_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False,
                              num_workers=8, pin_memory=True)

    model = PANNsCNN10(classes_num=234).to(device)
    criterion_species = nn.BCEWithLogitsLoss()
    criterion_class   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler    = torch.amp.GradScaler('cuda')

    epochs   = 100
    patience = 5
    best_f1  = 0.0
    counter  = 0
    print("開始 PANNs 模型訓練")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (waveforms, soft_labels, class_labels) in enumerate(train_loader):
            waveforms    = waveforms.to(device)
            soft_labels  = soft_labels.to(device)
            class_labels = class_labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if np.random.rand() > 0.3:
                    mixed_wav, labels_a, labels_b, lam = mixup_data(waveforms, soft_labels)
                    logits_species, logits_class = model(mixed_wav)
                    mixed_soft = lam * labels_a + (1 - lam) * labels_b
                    loss_species = criterion_species(logits_species, mixed_soft)
                else:
                    logits_species, logits_class = model(waveforms)
                    loss_species = criterion_species(logits_species, soft_labels)
                loss_class = criterion_class(logits_class, class_labels)
                loss = loss_species + 0.2 * loss_class

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch[{epoch+1}/{epochs}] Batch[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (sp: {loss_species.item():.4f}, cl: {loss_class.item():.4f})")

        print(f"Epoch {epoch+1} 平均 Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for waveforms, soft_labels, _ in val_loader:
                waveforms   = waveforms.to(device)
                soft_labels = soft_labels.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(waveforms)
                _, predicted   = torch.max(outputs, 1)
                _, true_labels = torch.max(soft_labels, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(true_labels.cpu().numpy())

        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} 驗證 F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), base_dir / 'models' / 'best_panns_model.pth')
            print("==> 最佳模型已儲存")
        else:
            counter += 1
            print(f"F1 未提升，累積 {counter}/{patience}")
            if counter >= patience:
                print("==> Early Stopping")
                break

    print("PANNs 訓練完成")


if __name__ == '__main__':
    main()
