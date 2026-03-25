"""
per-class 弱點分析腳本
- 用現有模型對驗證集跑推論
- 輸出每個物種的 ROC-AUC 和 AP，排序找出弱點
- 結果存到 models/per_class_analysis.csv
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from panns.dataset import PANNsDataset
from panns.model import PANNsCNN10
from train_panns import build_split


def run_analysis(model_path: Path, out_csv: Path = None):
    """對指定模型跑驗證集分析，回傳 DataFrame。"""
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    h5_path  = base_dir / 'processed_data' / 'train_waveforms.h5'
    sc_df    = pd.read_csv(base_dir / 'species_counts.csv')
    tax_df   = pd.read_csv(base_dir / 'taxonomy_encoded.csv')

    print("建立驗證集切分...")
    _, keys_val = build_split(h5_path, sc_df)
    print(f"驗證集 {len(keys_val)} 筆")

    val_dataset = PANNsDataset(str(h5_path), keys_val, chunk_length=160000)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = PANNsCNN10(classes_num=234).to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"模型載入：{model_path}")

    n_val    = len(val_dataset)
    all_probs   = np.empty((n_val, 234), dtype=np.float32)
    all_targets = np.empty((n_val, 234), dtype=np.float32)
    ptr = 0

    print("推論中...")
    with torch.no_grad():
        for waveforms, soft_labels, _ in val_loader:
            waveforms = waveforms.to(device)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu().float().numpy()
            b = probs.shape[0]
            all_probs[ptr:ptr+b]   = probs
            all_targets[ptr:ptr+b] = soft_labels.numpy()
            ptr += b

    all_probs   = all_probs[:ptr]
    all_targets = all_targets[:ptr]

    # label_id -> 物種資訊對照表
    id_to_info = {
        int(r['label_id']): {
            'primary_label': r['primary_label'],
            'common_name':   r['common_name'],
            'class_name':    r['class_name'],
        }
        for _, r in tax_df.iterrows()
    }
    count_map = dict(zip(sc_df['primary_label'].astype(str), sc_df['audio_count']))

    rows = []
    for i in range(234):
        y_true = (all_targets[:, i] >= 0.5).astype(int)
        y_prob = all_probs[:, i]
        n_pos  = y_true.sum()
        info   = id_to_info.get(i, {})
        sp     = str(info.get('primary_label', ''))

        if n_pos == 0:
            auc = ap = float('nan')
        elif n_pos == len(y_true):
            auc = ap = float('nan')
        else:
            auc = roc_auc_score(y_true, y_prob)
            ap  = average_precision_score(y_true, y_prob)

        rows.append({
            'label_id':      i,
            'primary_label': sp,
            'common_name':   info.get('common_name', ''),
            'class_name':    info.get('class_name', ''),
            'audio_count':   count_map.get(sp, 0),
            'val_samples':   int(n_pos),
            'roc_auc':       round(auc, 4) if not np.isnan(auc) else '',
            'avg_precision': round(ap,  4) if not np.isnan(ap)  else '',
        })

    df = pd.DataFrame(rows)
    df_valid = df[df['roc_auc'] != ''].copy()
    df_valid['roc_auc'] = df_valid['roc_auc'].astype(float)
    df_sorted = df_valid.sort_values('roc_auc')

    if out_csv is None:
        out_csv = base_dir / 'models' / 'per_class_analysis.csv'
    df_sorted.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n結果已儲存：{out_csv}")

    # ── 摘要輸出 ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  分析物種數：{len(df_valid)}（驗證集有樣本）")
    print(f"  平均 ROC-AUC：{df_valid['roc_auc'].mean():.4f}")
    print(f"{'='*65}")

    print("\n【最弱 20 個物種（ROC-AUC 最低）】")
    print(f"{'label_id':>8} {'primary_label':<15} {'common_name':<35} {'class':<10} {'count':>6} {'val_n':>6} {'AUC':>6}")
    print('-'*95)
    for _, r in df_sorted.head(20).iterrows():
        print(f"{int(r['label_id']):>8} {str(r['primary_label']):<15} {str(r['common_name']):<35} "
              f"{str(r['class_name']):<10} {int(r['audio_count']):>6} {int(r['val_samples']):>6} "
              f"{float(r['roc_auc']):>6.4f}")

    print("\n【最強 10 個物種（ROC-AUC 最高）】")
    print(f"{'label_id':>8} {'primary_label':<15} {'common_name':<35} {'class':<10} {'count':>6} {'AUC':>6}")
    print('-'*85)
    for _, r in df_sorted.tail(10).iterrows():
        print(f"{int(r['label_id']):>8} {str(r['primary_label']):<15} {str(r['common_name']):<35} "
              f"{str(r['class_name']):<10} {int(r['audio_count']):>6} {float(r['roc_auc']):>6.4f}")

    # ── 按類別分組統計 ────────────────────────────────────────────────
    print("\n【各生物類別平均 AUC】")
    for cls, grp in df_valid.groupby('class_name'):
        print(f"  {cls:<12}：{grp['roc_auc'].mean():.4f}（{len(grp)} 種）")

    # ── 稀少物種（≤10筆）的 AUC 分布 ─────────────────────────────────
    rare = df_valid[df_valid['audio_count'].astype(float) <= 10]
    if len(rare):
        print(f"\n【稀少物種（訓練資料 ≤10 筆）平均 AUC：{rare['roc_auc'].mean():.4f}，共 {len(rare)} 種】")

    return df_valid


def main():
    import argparse
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

    parser = argparse.ArgumentParser(description='Per-class 弱點分析')
    parser.add_argument('--model', type=str,
                        default=str(base_dir / 'models' / 'model_alpha.pth'),
                        help='模型權重路徑（預設：model_alpha.pth）')
    parser.add_argument('--out', type=str, default=None,
                        help='輸出 CSV 路徑（預設：models/per_class_analysis.csv）')
    args = parser.parse_args()

    out_csv = Path(args.out) if args.out else None
    run_analysis(Path(args.model), out_csv)


if __name__ == '__main__':
    main()
