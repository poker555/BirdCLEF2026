import os
import sys
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


def add_noise(waveforms: torch.Tensor, noise_ds: 'NoiseDataset',
              prob: float = 0.5, snr_db_range=(5, 20)) -> torch.Tensor:
    """
    以 prob 機率對 batch 中每條波形混入一段隨機噪音。
    SNR 在 snr_db_range 之間均勻取樣，確保鳥聲仍清晰可辨。
    """
    if noise_ds is None or len(noise_ds) == 0:
        return waveforms
    result = waveforms.clone()
    for i in range(result.shape[0]):
        if torch.rand(1).item() > prob:
            continue
        noise = torch.tensor(noise_ds.sample(), dtype=torch.float32).to(waveforms.device)
        # 對齊長度
        if noise.shape[0] < result.shape[1]:
            noise = noise.repeat(result.shape[1] // noise.shape[0] + 1)
        noise = noise[:result.shape[1]]
        # 依目標 SNR 縮放噪音
        snr_db  = torch.empty(1).uniform_(*snr_db_range).item()
        sig_rms = result[i].pow(2).mean().sqrt().clamp(min=1e-9)
        noi_rms = noise.pow(2).mean().sqrt().clamp(min=1e-9)
        scale   = sig_rms / (noi_rms * (10 ** (snr_db / 20)))
        result[i] = (result[i] + noise * scale).clamp(-1.0, 1.0)
    return result


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


def build_sampler(h5_path: Path, keys_train: list, sc_df: pd.DataFrame,
                  train_df: pd.DataFrame = None):
    """
    weight = quality_weight(filename, rating) / sqrt(species_count)

    - sqrt 平滑：避免 1/count 過於激進（499:1 → 22:1）
    - 品質加權：只對 XC 音訊且 rating>0 有效，iNat 音訊維持 1.0
      rating=5 → 1.0，rating=1 → 0.6，rating=0 或 iNat → 1.0
    """
    count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))

    # 建立 filename -> rating 查詢表（只有 train_df 提供時才用）
    rating_map = {}
    if train_df is not None:
        for _, row in train_df.iterrows():
            rating_map[str(row['filename'])] = float(row.get('rating', 0))

    CHUNK = 160000
    AUG_SUFFIXES = ('_ts09', '_ts11', '_ps+1', '_ps-1')
    segment_weights = []

    with h5py.File(str(h5_path), 'r') as f:
        for key in keys_train:
            species = key.split('/')[0]
            cnt = count_map.get(species, 1)

            # sqrt 平滑，避免極端比例
            w = 1.0 / np.sqrt(cnt)

            # 品質加權：還原原始 filename（去掉增強後綴）
            base_fname = key
            for suf in AUG_SUFFIXES:
                if key.endswith(suf):
                    base_fname = key[:-len(suf)]
                    break
            rating = rating_map.get(base_fname, 0)
            fname_stem = base_fname.split('/')[-1] if '/' in base_fname else base_fname
            if 'XC' in fname_stem and rating > 0:
                # rating=5 → 1.0，rating=1 → 0.6，線性映射
                quality_w = 0.5 + (rating / 5.0) * 0.5
                w *= quality_w

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
    train_df = pd.read_csv(base_dir / "train.csv")

    print("建立訓練/驗證切分...")
    keys_train, keys_val = build_split(h5_path, sc_df)
    print(f"訓練集 {len(keys_train)} 筆（含增強），驗證集 {len(keys_val)} 筆")

    train_dataset = PANNsDataset(str(h5_path), keys_train, chunk_length=160000)
    val_dataset   = PANNsDataset(str(h5_path), keys_val,   chunk_length=160000)

    # soundscape 分流：train 部分加入訓練集，val 部分加入驗證集
    soundscape_dir = base_dir / 'train_soundscapes'
    labels_csv     = base_dir / 'train_soundscapes_labels.csv'
    taxonomy_csv   = base_dir / 'taxonomy_encoded.csv'
    sc_train_size  = 0
    if soundscape_dir.exists() and labels_csv.exists():
        sc_train_dataset = SoundscapeWaveformDataset(
            str(soundscape_dir), str(labels_csv), str(taxonomy_csv),
            split='train', sc_df=sc_df
        )
        sc_val_dataset = SoundscapeWaveformDataset(
            str(soundscape_dir), str(labels_csv), str(taxonomy_csv),
            split='val', sc_df=sc_df
        )
        sc_train_size = len(sc_train_dataset)
        train_dataset = ConcatDataset([train_dataset, sc_train_dataset])
        val_dataset   = ConcatDataset([val_dataset,   sc_val_dataset])
        print(f"soundscape 訓練集: {sc_train_size} 片段，驗證集: {len(sc_val_dataset)} 片段")

    print(f"訓練集總計 {len(train_dataset)} 個片段，驗證集 {len(val_dataset)} 個片段")

    # WeightedRandomSampler：HDF5 片段 + soundscape 片段
    segment_weights = build_sampler(h5_path, keys_train, sc_df, train_df=train_df)
    if sc_train_size > 0:
        count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))
        tax_df    = pd.read_csv(taxonomy_csv)
        # label_id -> primary_label 反查表
        id_to_sp  = {int(r['label_id']): str(r['primary_label']) for _, r in tax_df.iterrows()}
        for sample in sc_train_dataset.samples:
            primary_idx = int(np.argmax(sample['soft_label']))
            sp  = id_to_sp.get(primary_idx, '')
            cnt = count_map.get(sp, 1)
            # 稀少物種（0筆）給 weight=2.0，其餘 1/sqrt(count)
            w   = 2.0 if cnt == 0 else 1.0 / np.sqrt(max(cnt, 1))
            segment_weights.append(w)

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
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 噪音混入：從 train_soundscapes 抽取低能量片段
    noise_ds = None
    if soundscape_dir.exists():
        noise_ds = NoiseDataset(str(soundscape_dir), chunk_length=160000, max_files=500)
        if len(noise_ds) == 0:
            print("⚠️  未找到低能量噪音片段，跳過噪音混入增強")
            noise_ds = None

    epochs   = 100
    patience = 5
    best_f1  = 0.0
    best_map = 0.0
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

            # 噪音混入增強（在 MixUp 之前，p=0.5，SNR 5~20 dB）
            waveforms = add_noise(waveforms, noise_ds, prob=0.5, snr_db_range=(5, 20))

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                if torch.rand(1).item() > 0.3:
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
        n_val = len(val_dataset)
        all_probs_np   = np.empty((n_val, 234), dtype=np.float32)
        all_preds      = np.empty(n_val, dtype=np.int64)
        all_targets    = np.empty(n_val, dtype=np.int64)
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

        # macro F1（argmax 預測）
        val_f1 = f1_score(all_targets[:ptr], all_preds[:ptr], average='macro', zero_division=0)

        # mAP：只計算驗證集中有出現的類別
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
        scheduler.step(val_map)  # 以 mAP 驅動學習率調整

        if val_map > best_map:
            best_map = val_map
            best_f1  = val_f1
            counter  = 0
            torch.save(model.state_dict(), base_dir / 'models' / 'best_panns_model.pth')
            print(f"==> 最佳模型已儲存（暫存為 best_panns_model.pth）")
        else:
            counter += 1
            print(f"mAP 未提升，累積 {counter}/{patience}")
            if counter >= patience:
                print("==> Early Stopping")
                break

    # 決定本次模型的北約名稱（依 models/ 下已有的命名檔案自動遞增）
    NATO = [
        'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
        'hotel', 'india', 'juliet', 'kilo', 'lima', 'mike', 'november',
        'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango', 'uniform',
        'victor', 'whiskey', 'xray', 'yankee', 'zulu'
    ]
    existing = {p.stem.replace('model_', '') for p in (base_dir / 'models').glob('model_*.pth')}
    model_codename = next((n for n in NATO if n not in existing), f'model_{len(existing)}')
    named_model_path = base_dir / 'models' / f'model_{model_codename}.pth'
    import shutil
    shutil.copy(base_dir / 'models' / 'best_panns_model.pth', named_model_path)
    print(f"==> 模型已命名並儲存：{named_model_path.name}")

    # 訓練結果寫入 JSON，供 run_pipeline.py 讀取後追加到 experiment_log.csv
    import json
    result = {
        'model_name':         model_codename,
        'model':              'panns',
        'best_f1':            round(best_f1, 6),
        'best_map':           round(best_map, 6),
        'kaggle_roc_auc':     '',
        'epochs_trained':     epoch + 1,
        'rare_threshold':     5,
        'augmentation':       'time_stretch_0.9/1.1, pitch_shift_+/-1',
        'sampler':            'WeightedRandomSampler(quality_weight/sqrt(species_count))',
        'mixup_alpha':        0.4,
        'mixup_prob':         0.7,
        'soft_label_weight':  0.3,
        'val_strategy':       'rare_all_train + stratified_80_20 + soundscape',
        'aux_loss_weight':    0.2,
        'notes':              'n_fft=512, n_mels=128, noise_augmentation: soundscape low-energy chunks, p=0.5, SNR 5-20dB',
    }
    out_path = base_dir / 'models' / 'panns_train_result.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"訓練結果已寫入 {out_path}")
    print("PANNs 訓練完成")


if __name__ == '__main__':
    main()
