"""
一鍵執行完整訓練流程：
    1. 前處理 Mel-Spectrogram HDF5（CNN 用）
    2. 前處理 Waveform HDF5（PANNs 用）
    3. 訓練 CNN 模型
    4. 訓練 PANNs 模型

使用方式（從專案根目錄執行）：
    python script/run_pipeline.py

常用參數：
    --skip-preprocess   跳過前處理（HDF5 已存在時使用）
    --only preprocess   只跑前處理
    --only cnn          只跑 CNN 訓練
    --only panns        只跑 PANNs 訓練
    --test              快速測試模式（少量資料、2 epoch，驗證流程可跑通）
"""

import argparse
import subprocess
import sys
import os
import shutil
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
    輸出到 processed_data/test_spectrograms.h5 / test_waveforms.h5，不覆蓋正式資料。
    """
    print("\n" + "="*60)
    print("  TEST MODE：快速驗證流程可跑通")
    print("  - 前處理只取前 30 筆音訊")
    print("  - 訓練只跑 2 epoch，batch_size=8")
    print("  - 輸出至 processed_data/test_*.h5（不覆蓋正式資料）")
    print("="*60)

    test_dir = base / 'processed_data'
    test_dir.mkdir(exist_ok=True)

    # ── 步驟 1：產生測試用前處理腳本（inline，不修改原始腳本）──────────
    test_preprocess_mel = base / '_test_preprocess_mel.py'
    test_preprocess_wav = base / '_test_preprocess_wav.py'
    test_train_cnn      = base / '_test_train_cnn.py'
    test_train_panns    = base / '_test_train_panns.py'

    # 測試前處理：mel spectrogram（只取前 30 筆，輸出到 test_spectrograms.h5）
    test_preprocess_mel.write_text(textwrap.dedent("""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        if __name__ == '__main__':
            import preprocess as pp
            import pandas as pd
            import h5py
            from multiprocessing import Pool
            from tqdm import tqdm

            pp.OUTPUT_H5 = pp.Path("processed_data/test_spectrograms.h5")
            pp.OUTPUT_H5.parent.mkdir(exist_ok=True)

            df     = pd.read_csv("train.csv").head(30)
            tax_df = pd.read_csv("taxonomy_encoded.csv")
            sc_df  = pd.read_csv("species_counts.csv")
            df = df.merge(tax_df[['primary_label','label_id','class_id']], on='primary_label', how='left')

            label_map    = pp.build_label_map(tax_df)
            rare_species = set(sc_df[sc_df['audio_count'] <= pp.RARE_THRESHOLD]['primary_label'].astype(str))
            voice_map    = pp.load_voice_map(pp.VOICE_CSV)

            tasks, task_map = [], {}
            for _, row in df.iterrows():
                fname   = row['filename']
                species = str(row['primary_label'])
                key     = '/'.join(fname.replace('\\\\', '/').split('/')[-2:])
                soft_label = pp.build_soft_label(int(row['label_id']), str(row['secondary_labels']), label_map, pp.NUM_CLASSES)
                class_id   = int(row['class_id'])
                voice_segs = voice_map.get(key, [])
                task_map[fname] = (soft_label, class_id)
                tasks.append({'filename': fname, 'voice_segments': voice_segs, 'aug': None})
                if species in rare_species:
                    for aug in pp.AUG_CONFIGS:
                        h5_key = fname + aug['suffix']
                        task_map[h5_key] = (soft_label, class_id)
                        tasks.append({'filename': fname, 'voice_segments': voice_segs, 'aug': aug})

            print(f"[TEST] 共 {len(tasks)} 筆任務（含增強）")
            success = failed = 0
            with h5py.File(str(pp.OUTPUT_H5), 'w', rdcc_nbytes=0) as h5_file:
                with Pool(processes=2, maxtasksperchild=20) as pool:
                    with tqdm(total=len(tasks)) as pbar:
                        for batch_start in range(0, len(tasks), 8):
                            batch = tasks[batch_start:batch_start+8]
                            results = pool.map(pp.process_single_audio, batch)
                            for result in results:
                                if result['status'] == 'success':
                                    sl, cid = task_map[result['h5_key']]
                                    ds = h5_file.create_dataset(name=result['h5_key'], data=result['mel_db'], compression='gzip')
                                    ds.attrs['soft_label'] = sl
                                    ds.attrs['class_id']   = cid
                                    success += 1
                                else:
                                    print(f"[錯誤] {result['h5_key']} - {result['error_msg']}")
                                    failed += 1
                                del result
                            h5_file.flush()
                            del results
                            pbar.update(len(batch))
            print(f"[TEST] Mel 前處理完成：成功 {success}，失敗 {failed}")
    """), encoding='utf-8')

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

    # 測試訓練：CNN（2 epoch，batch_size=8，使用 test_spectrograms.h5）
    test_train_cnn.write_text(textwrap.dedent("""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        if __name__ == '__main__':
            import h5py, torch, numpy as np, pandas as pd
            import torch.nn as nn, torch.optim as optim
            from pathlib import Path
            from torch.utils.data import DataLoader, WeightedRandomSampler
            from sklearn.metrics import f1_score
            from cnn.dataset import BirdDataset
            from cnn.model import BirdModel
            from train import build_split, build_sampler

            device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
            h5_path  = base_dir / 'processed_data/test_spectrograms.h5'
            sc_df    = pd.read_csv(base_dir / 'species_counts.csv')

            keys_train, keys_val = build_split(h5_path, sc_df)
            print(f"[TEST CNN] 訓練 {len(keys_train)} 筆，驗證 {len(keys_val)} 筆")

            train_ds = BirdDataset(h5_path, keys_train)
            val_ds   = BirdDataset(h5_path, keys_val) if keys_val else train_ds

            key_weights = build_sampler(h5_path, keys_train, sc_df)
            seg_weights = []
            with h5py.File(str(h5_path), 'r') as f:
                for key, w in zip(keys_train, key_weights):
                    try:    n_seg = max(1, f[key].shape[1] // 313)
                    except: n_seg = 1
                    seg_weights.extend([w] * n_seg)

            sampler      = WeightedRandomSampler(seg_weights, len(seg_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,   num_workers=0)

            model     = BirdModel(num_classes=234).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            for epoch in range(2):
                model.train()
                for images, soft_labels, class_labels in train_loader:
                    images, soft_labels = images.to(device), soft_labels.to(device)
                    optimizer.zero_grad()
                    logits, _ = model(images)
                    loss = criterion(logits, soft_labels)
                    loss.backward()
                    optimizer.step()
                print(f"[TEST CNN] Epoch {epoch+1}/2 完成，loss={loss.item():.4f}")

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for images, soft_labels, _ in val_loader:
                    out = model(images.to(device))
                    preds.extend(torch.max(out, 1)[1].cpu().numpy())
                    targets.extend(torch.max(soft_labels, 1)[1].numpy())
            f1 = f1_score(targets, preds, average='macro', zero_division=0)
            print(f"[TEST CNN] 驗證 F1={f1:.4f}  ✅ CNN 流程正常")
    """), encoding='utf-8')

    # 測試訓練：PANNs（2 epoch，batch_size=8，使用 test_waveforms.h5）
    test_train_panns.write_text(textwrap.dedent("""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        if __name__ == '__main__':
            import h5py, torch, numpy as np, pandas as pd
            import torch.nn as nn, torch.optim as optim
            from pathlib import Path
            from torch.utils.data import DataLoader, WeightedRandomSampler
            from sklearn.metrics import f1_score
            from panns.dataset import PANNsDataset
            from panns.model import PANNsCNN10
            from train_panns import build_split, build_sampler

            device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
            h5_path  = base_dir / 'processed_data/test_waveforms.h5'
            sc_df    = pd.read_csv(base_dir / 'species_counts.csv')

            keys_train, keys_val = build_split(h5_path, sc_df)
            print(f"[TEST PANNs] 訓練 {len(keys_train)} 筆，驗證 {len(keys_val)} 筆")

            train_ds = PANNsDataset(str(h5_path), keys_train, chunk_length=160000)
            val_ds   = PANNsDataset(str(h5_path), keys_val,   chunk_length=160000) if keys_val else train_ds

            seg_weights = build_sampler(h5_path, keys_train, sc_df)
            sampler      = WeightedRandomSampler(seg_weights, len(seg_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,   num_workers=0)

            model     = PANNsCNN10(classes_num=234).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            for epoch in range(2):
                model.train()
                for waveforms, soft_labels, class_labels in train_loader:
                    waveforms, soft_labels = waveforms.to(device), soft_labels.to(device)
                    optimizer.zero_grad()
                    logits, _ = model(waveforms)
                    loss = criterion(logits, soft_labels)
                    loss.backward()
                    optimizer.step()
                print(f"[TEST PANNs] Epoch {epoch+1}/2 完成，loss={loss.item():.4f}")

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for waveforms, soft_labels, _ in val_loader:
                    out = model(waveforms.to(device))
                    preds.extend(torch.max(out, 1)[1].cpu().numpy())
                    targets.extend(torch.max(soft_labels, 1)[1].numpy())
            f1 = f1_score(targets, preds, average='macro', zero_division=0)
            print(f"[TEST PANNs] 驗證 F1={f1:.4f}  ✅ PANNs 流程正常")
    """), encoding='utf-8')

    # ── 執行四個測試步驟 ──────────────────────────────────────────────
    steps = [
        (str(test_preprocess_mel),  '測試 1/4：Mel 前處理（30 筆 + 增強）'),
        (str(test_preprocess_wav),  '測試 2/4：Waveform 前處理（30 筆 + 增強）'),
        (str(test_train_cnn),       '測試 3/4：CNN 訓練（2 epoch）'),
        (str(test_train_panns),     '測試 4/4：PANNs 訓練（2 epoch）'),
    ]

    all_passed = True
    for script, desc in steps:
        print(f"\n{'─'*60}")
        print(f"  {desc}")
        print(f"{'─'*60}")
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(base.parent)  # 專案根目錄
        )
        if result.returncode != 0:
            print(f"\n❌ {desc} 失敗！")
            all_passed = False
            break
        print(f"✅ {desc} 通過")

    # ── 清理暫存測試腳本 ─────────────────────────────────────────────
    for f in [test_preprocess_mel, test_preprocess_wav, test_train_cnn, test_train_panns]:
        f.unlink(missing_ok=True)

    print("\n" + "="*60)
    if all_passed:
        print("  ✅ 所有測試通過！可以執行完整訓練流程。")
        print("  測試用 HDF5 保留於 processed_data/test_*.h5 供檢查。")
        print("  執行正式流程：python script/run_pipeline.py")
    else:
        print("  ❌ 測試未全部通過，請檢查上方錯誤訊息。")
    print("="*60)


def run(script_path: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(Path(__file__).resolve().parent.parent)
    )
    if result.returncode != 0:
        print(f"\n[錯誤] {desc} 失敗，中止流程。")
        sys.exit(result.returncode)
    print(f"\n[完成] {desc}")


def log_experiment(base: Path):
    """
    讀取訓練腳本輸出的 JSON 結果，追加一筆紀錄到 experiment_log.csv。
    每次執行 pipeline 產生一個 run_id（時間戳），CNN 和 PANNs 各一列。
    """
    import json
    import csv
    from datetime import datetime

    log_path = base.parent / 'experiment_log.csv'
    run_id   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    fieldnames = [
        'run_id', 'model', 'best_f1', 'best_map', 'epochs_trained',
        'rare_threshold', 'augmentation', 'sampler',
        'mixup_alpha', 'mixup_prob', 'soft_label_weight',
        'val_strategy', 'aux_loss_weight', 'notes',
    ]

    rows = []
    for model_name in ['cnn', 'panns']:
        json_path = base.parent / 'models' / f'{model_name}_train_result.json'
        if not json_path.exists():
            print(f"[LOG] 找不到 {json_path}，跳過此模型紀錄。")
            continue
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        data['run_id'] = run_id
        rows.append({k: data.get(k, '') for k in fieldnames})

    if not rows:
        print("[LOG] 沒有可寫入的訓練結果。")
        return

    write_header = not log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n[LOG] 實驗紀錄已追加至 {log_path}（run_id: {run_id}）")
    for r in rows:
        print(f"      {r['model']:6s} | best_f1={r['best_f1']} | epochs={r['epochs_trained']}")


def main():
    parser = argparse.ArgumentParser(description='BirdCLEF 2026 完整訓練流程')
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
        'preprocess_wav': (str(base / 'preprocess_waveform.py'), '步驟 1/2：前處理 Waveform HDF5（PANNs 用）'),
        'panns':          (str(base / 'train_panns.py'),          '步驟 2/2：訓練 PANNs 模型'),
    }

    if args.only == 'preprocess':
        run(*steps['preprocess_wav'])
    elif args.only == 'panns':
        run(*steps['panns'])
    else:
        if not args.skip_preprocess:
            run(*steps['preprocess_wav'])
        else:
            print("已跳過前處理步驟。")
        run(*steps['panns'])
        log_experiment(base)

    print("\n" + "="*60)
    print("  全部流程完成！")
    print("="*60)


if __name__ == '__main__':
    main()
