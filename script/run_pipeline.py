"""
一鍵執行完整訓練流程（PANNs）：
    1. 前處理 Waveform HDF5（PANNs 用）
    2. 訓練 PANNs 模型

使用方式（從專案根目錄執行）：
    python script/run_pipeline.py

常用參數：
    --skip-preprocess   跳過前處理（HDF5 已存在時使用）
    --only preprocess   只跑前處理
    --only panns        只跑 PANNs 訓練
    --test              快速測試模式（少量資料、2 epoch，驗證流程可跑通）
"""

import argparse
import subprocess
import sys
import os
import textwrap
from pathlib import Path


def run(script_path: str, desc: str, extra_env: dict = None):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env
    )
    if result.returncode != 0:
        print(f"\n[錯誤] {desc} 失敗，中止流程。")
        sys.exit(result.returncode)
    print(f"\n[完成] {desc}")


def run_test_mode(base: Path):
    """
    快速測試模式：用少量資料跑完整流程，驗證每個環節都能通過。
    輸出到 processed_data/test_waveforms.h5，不覆蓋正式資料。
    """
    print("\n" + "="*60)
    print("  TEST MODE：快速驗證流程可跑通")
    print("  - 前處理只取前 30 筆音訊")
    print("  - 訓練只跑 2 epoch，batch_size=8")
    print("  - 輸出至 processed_data/test_waveforms.h5（不覆蓋正式資料）")
    print("="*60)

    test_dir = base / 'processed_data'
    test_dir.mkdir(exist_ok=True)

    test_preprocess_wav = base / '_test_preprocess_wav.py'
    test_train_panns    = base / '_test_train_panns.py'

    # 測試前處理：waveform（只取前 30 筆，輸出到 test_waveforms.h5）
    test_preprocess_wav.write_text(textwrap.dedent("""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        if __name__ == '__main__':
            import preprocess_waveform as pw
            import pandas as pd
            import h5py
            from multiprocessing import Pool
            from tqdm import tqdm

            pw.OUTPUT_H5 = pw.Path("processed_data/test_waveforms.h5")
            pw.OUTPUT_H5.parent.mkdir(exist_ok=True)

            df     = pd.read_csv("train.csv").head(30)
            tax_df = pd.read_csv("taxonomy_encoded.csv")
            sc_df  = pd.read_csv("species_counts.csv")
            df = df.merge(tax_df[['primary_label','label_id','class_id']], on='primary_label', how='left')

            label_map    = pw.build_label_map(tax_df)
            rare_species = set(sc_df[sc_df['audio_count'] <= pw.RARE_THRESHOLD]['primary_label'].astype(str))
            voice_map    = pw.load_voice_map(pw.VOICE_CSV)

            tasks, task_map = [], {}
            for _, row in df.iterrows():
                fname   = row['filename']
                species = str(row['primary_label'])
                key     = '/'.join(fname.replace('\\\\', '/').split('/')[-2:])
                soft_label = pw.build_soft_label(int(row['label_id']), str(row['secondary_labels']), label_map, pw.NUM_CLASSES)
                class_id   = int(row['class_id'])
                voice_segs = voice_map.get(key, [])
                task_map[fname] = (soft_label, class_id)
                tasks.append({'filename': fname, 'voice_segments': voice_segs, 'aug': None})
                if species in rare_species:
                    for aug in pw.AUG_CONFIGS:
                        h5_key = fname + aug['suffix']
                        task_map[h5_key] = (soft_label, class_id)
                        tasks.append({'filename': fname, 'voice_segments': voice_segs, 'aug': aug})

            print(f"[TEST] 共 {len(tasks)} 筆任務（含增強）")
            success = failed = 0
            with h5py.File(str(pw.OUTPUT_H5), 'w', rdcc_nbytes=0) as h5_file:
                h5_file.attrs['sample_rate'] = pw.SAMPLE_RATE
                with Pool(processes=2, maxtasksperchild=20) as pool:
                    with tqdm(total=len(tasks)) as pbar:
                        for batch_start in range(0, len(tasks), 8):
                            batch = tasks[batch_start:batch_start+8]
                            results = pool.map(pw.process_single_audio, batch)
                            for result in results:
                                if result['status'] == 'success':
                                    sl, cid = task_map[result['h5_key']]
                                    ds = h5_file.create_dataset(name=result['h5_key'], data=result['waveform'], compression='gzip', compression_opts=4)
                                    ds.attrs['soft_label']  = sl
                                    ds.attrs['class_id']    = cid
                                    ds.attrs['sample_rate'] = pw.SAMPLE_RATE
                                    success += 1
                                else:
                                    print(f"[錯誤] {result['h5_key']} - {result['error_msg']}")
                                    failed += 1
                                del result
                            h5_file.flush()
                            del results
                            pbar.update(len(batch))
            print(f"[TEST] Waveform 前處理完成：成功 {success}，失敗 {failed}")
    """), encoding='utf-8')

    # 測試訓練：PANNs（2 epoch，batch_size=8，使用 test_waveforms.h5）
    # 與正式訓練邏輯一致：MixUp、噪音混入、autocast、aux loss、WeightedSampler
    test_train_panns.write_text(textwrap.dedent("""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        # 清除舊的模組快取，確保載入最新版本
        for _mod in list(sys.modules.keys()):
            if any(_mod.startswith(p) for p in ['noise_dataset', 'train_panns', 'panns', 'soundscape_dataset']):
                del sys.modules[_mod]

        if __name__ == '__main__':
            import torch, numpy as np, pandas as pd
            import torch.nn as nn, torch.optim as optim
            from pathlib import Path
            from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
            from sklearn.metrics import f1_score, average_precision_score
            from panns.dataset import PANNsDataset
            from panns.model import PANNsCNN10
            from train_panns import build_split, build_sampler, mixup_data, add_noise
            from noise_dataset import NoiseDataset
            from soundscape_dataset import SoundscapeWaveformDataset

            device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
            h5_path  = base_dir / 'processed_data/test_waveforms.h5'
            sc_df    = pd.read_csv(base_dir / 'species_counts.csv')
            train_df = pd.read_csv(base_dir / 'train.csv')

            keys_train, keys_val = build_split(h5_path, sc_df)
            print(f"[TEST PANNs] 訓練 {len(keys_train)} 筆，驗證 {len(keys_val)} 筆")

            train_ds = PANNsDataset(str(h5_path), keys_train, chunk_length=160000)
            val_ds   = PANNsDataset(str(h5_path), keys_val,   chunk_length=160000) if keys_val else train_ds

            # Soundscape 資料集（有則加入，無則跳過）
            soundscape_dir = base_dir / 'train_soundscapes'
            labels_csv     = base_dir / 'train_soundscapes_labels.csv'
            taxonomy_csv   = base_dir / 'taxonomy_encoded.csv'
            sc_train_size  = 0
            if soundscape_dir.exists() and labels_csv.exists():
                sc_train_ds = SoundscapeWaveformDataset(
                    str(soundscape_dir), str(labels_csv), str(taxonomy_csv),
                    split='train', sc_df=sc_df
                )
                sc_val_ds = SoundscapeWaveformDataset(
                    str(soundscape_dir), str(labels_csv), str(taxonomy_csv),
                    split='val', sc_df=sc_df
                )
                sc_train_size = len(sc_train_ds)
                train_ds = ConcatDataset([train_ds, sc_train_ds])
                val_ds   = ConcatDataset([val_ds,   sc_val_ds])
                print(f"[TEST PANNs] soundscape 訓練 {sc_train_size} 筆，驗證 {len(sc_val_ds)} 筆")

            # WeightedRandomSampler（含品質加權，與正式訓練一致）
            seg_weights = build_sampler(h5_path, keys_train, sc_df, train_df=train_df)
            if sc_train_size > 0:
                count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))
                tax_df    = pd.read_csv(taxonomy_csv)
                id_to_sp  = {int(r['label_id']): str(r['primary_label']) for _, r in tax_df.iterrows()}
                for sample in sc_train_ds.samples:
                    primary_idx = int(np.argmax(sample['soft_label']))
                    sp  = id_to_sp.get(primary_idx, '')
                    cnt = count_map.get(sp, 1)
                    seg_weights.append(2.0 if cnt == 0 else 1.0 / np.sqrt(max(cnt, 1)))
            sampler      = WeightedRandomSampler(seg_weights, len(seg_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,   num_workers=0)

            # 噪音混入
            noise_ds = None
            esc50_dir = base_dir / 'ESC-50-master' / 'audio'
            try:
                noise_ds = NoiseDataset(str(esc50_dir), chunk_length=160000, auto_download=False)
                if len(noise_ds) == 0:
                    print("[TEST PANNs] ⚠️  未找到噪音片段，跳過噪音混入")
                    noise_ds = None
            except FileNotFoundError:
                print("[TEST PANNs] ⚠️  ESC-50 不存在，跳過噪音混入")

            model              = PANNsCNN10(classes_num=234).to(device)
            criterion_species  = nn.BCEWithLogitsLoss()
            criterion_class    = nn.CrossEntropyLoss()
            optimizer          = optim.AdamW(model.parameters(), lr=1e-3)
            scaler             = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

            # ── 訓練迴圈（與正式訓練完全一致）──
            for epoch in range(2):
                model.train()
                for waveforms, soft_labels, class_labels in train_loader:
                    waveforms    = waveforms.to(device)
                    soft_labels  = soft_labels.to(device)
                    class_labels = class_labels.to(device)
                    optimizer.zero_grad()

                    waveforms = add_noise(waveforms, noise_ds, prob=0.5, snr_db_range=(5, 20))

                    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                            dtype=torch.bfloat16,
                                            enabled=(device.type == 'cuda')):
                        if torch.rand(1).item() > 0.3:
                            mixed_wav, labels_a, labels_b, lam = mixup_data(waveforms, soft_labels)
                            logits_species, logits_class = model(mixed_wav)
                            mixed_soft   = lam * labels_a + (1 - lam) * labels_b
                            loss_species = criterion_species(logits_species, mixed_soft)
                        else:
                            logits_species, logits_class = model(waveforms)
                            loss_species = criterion_species(logits_species, soft_labels)
                        loss_class = criterion_class(logits_class, class_labels)
                        loss       = loss_species + 0.2 * loss_class

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                print(f"[TEST PANNs] Epoch {epoch+1}/2 完成，loss={loss.item():.4f} "
                      f"(sp={loss_species.item():.4f}, cl={loss_class.item():.4f})")

            # ── 驗證迴圈（與正式訓練完全一致）──
            model.eval()
            all_probs, all_preds, all_targets = [], [], []
            with torch.no_grad():
                for waveforms, soft_labels, _ in val_loader:
                    waveforms = waveforms.to(device)
                    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                            dtype=torch.bfloat16,
                                            enabled=(device.type == 'cuda')):
                        outputs = model(waveforms)
                    probs = torch.sigmoid(outputs).cpu().float().numpy()
                    all_probs.append(probs)
                    all_preds.extend(outputs.argmax(dim=1).cpu().float().numpy().astype(int))
                    all_targets.extend(soft_labels.argmax(dim=1).numpy())

            all_probs   = np.vstack(all_probs)
            all_targets = np.array(all_targets)
            all_preds   = np.array(all_preds)
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

            targets_onehot  = np.zeros_like(all_probs)
            for i, t in enumerate(all_targets):
                targets_onehot[i, t] = 1.0
            present = np.where(targets_onehot.sum(axis=0) > 0)[0]
            val_map = average_precision_score(
                targets_onehot[:, present], all_probs[:, present], average='macro'
            ) if len(present) > 0 else 0.0

            print(f"[TEST PANNs] 驗證 F1={f1:.4f}  mAP={val_map:.4f}  ✅ PANNs 流程正常")
    """), encoding='utf-8')

    steps = [
        (str(test_preprocess_wav), '測試 1/2：Waveform 前處理（30 筆 + 增強）'),
        (str(test_train_panns),    '測試 2/2：PANNs 訓練（2 epoch）'),
    ]

    all_passed = True
    for script, desc in steps:
        print(f"\n{'─'*60}")
        print(f"  {desc}")
        print(f"{'─'*60}")
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(base.parent)
        )
        if result.returncode != 0:
            print(f"\n❌ {desc} 失敗！")
            all_passed = False
            break
        print(f"✅ {desc} 通過")

    for f in [test_preprocess_wav, test_train_panns]:
        f.unlink(missing_ok=True)

    print("\n" + "="*60)
    if all_passed:
        print("  ✅ 所有測試通過！可以執行完整訓練流程。")
        print("  測試用 HDF5 保留於 processed_data/test_waveforms.h5 供檢查。")
        print("  執行正式流程：python script/run_pipeline.py")
    else:
        print("  ❌ 測試未全部通過，請檢查上方錯誤訊息。")
    print("="*60)


def log_experiment(base: Path):
    """
    讀取 panns_train_result.json，追加一筆紀錄到 experiment_log.csv。
    """
    import json
    import csv
    from datetime import datetime

    log_path  = base.parent / 'experiment_log.csv'
    run_id    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json_path = base.parent / 'models' / 'panns_train_result.json'

    fieldnames = [
        'run_id', 'model_name', 'model', 'best_f1', 'best_map', 'kaggle_roc_auc', 'epochs_trained',
        'rare_threshold', 'use_mixup', 'mixup_alpha', 'mixup_prob',
        'use_noise_aug', 'noise_prob', 'noise_snr_range',
        'use_spec_augment', 'use_label_smoothing', 'label_smoothing',
        'use_grad_clip', 'grad_clip_norm', 'lr_scheduler',
        'use_weighted_sampler', 'use_soundscape',
        'soft_label_weight', 'aux_loss_weight', 'val_split', 'val_strategy',
    ]

    if not json_path.exists():
        print(f"[LOG] 找不到 {json_path}，跳過紀錄。")
        return

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    data['run_id'] = run_id
    row = {k: data.get(k, '') for k in fieldnames}

    write_header = not log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n[LOG] 實驗紀錄已追加至 {log_path}（run_id: {run_id}）")
    print(f"      [{row['model_name']}] panns | best_f1={row['best_f1']} | best_map={row['best_map']} | epochs={row['epochs_trained']}")


def main():
    parser = argparse.ArgumentParser(description='BirdCLEF 2026 PANNs 訓練流程')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳過前處理步驟（HDF5 已存在時使用）')
    parser.add_argument('--only', choices=['preprocess', 'panns'],
                        help='只執行指定步驟')
    parser.add_argument('--test', action='store_true',
                        help='快速測試模式（30 筆資料、2 epoch，驗證流程可跑通）')
    args = parser.parse_args()

    base = Path(os.path.dirname(os.path.abspath(__file__)))

    if args.test:
        run_test_mode(base)
        return

    steps = {
        'preprocess': (str(base / 'preprocess_waveform.py'), '步驟 1/2：前處理 Waveform HDF5（PANNs 用）'),
        'panns':      (str(base / 'train_panns.py'),          '步驟 2/2：訓練 PANNs 模型'),
    }

    if args.only == 'preprocess':
        run(*steps['preprocess'])
    elif args.only == 'panns':
        run(*steps['panns'])
    else:
        if not args.skip_preprocess:
            run(*steps['preprocess'])
        else:
            print("已跳過前處理步驟。")
        run(*steps['panns'])
        log_experiment(base)

    print("\n" + "="*60)
    print("  全部流程完成！")
    print("="*60)


if __name__ == '__main__':
    main()
