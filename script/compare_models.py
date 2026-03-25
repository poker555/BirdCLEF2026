"""
比較兩個模型在驗證集上的 per-class 表現。

使用方式：
    python script/compare_models.py --a models/model_alpha.pth --b models/model_bravo.pth
    python script/compare_models.py  # 預設比較 alpha vs bravo
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyze_per_class import run_analysis


def main():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

    parser = argparse.ArgumentParser(description='比較兩個模型的 per-class 表現')
    parser.add_argument('--a', type=str,
                        default=str(base_dir / 'models' / 'model_alpha.pth'),
                        help='模型 A 路徑（預設：model_alpha.pth）')
    parser.add_argument('--b', type=str,
                        default=str(base_dir / 'models' / 'model_bravo.pth'),
                        help='模型 B 路徑（預設：model_bravo.pth）')
    args = parser.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    name_a = path_a.stem  # e.g. model_alpha
    name_b = path_b.stem

    print(f"\n{'='*65}")
    print(f"  模型 A：{name_a}")
    print(f"{'='*65}")
    out_a = base_dir / 'models' / f'per_class_{name_a}.csv'
    df_a = run_analysis(path_a, out_csv=out_a)

    print(f"\n{'='*65}")
    print(f"  模型 B：{name_b}")
    print(f"{'='*65}")
    out_b = base_dir / 'models' / f'per_class_{name_b}.csv'
    df_b = run_analysis(path_b, out_csv=out_b)

    # ── 合併比較 ──────────────────────────────────────────────────────
    df_a = df_a.set_index('label_id')[['primary_label', 'common_name', 'class_name',
                                        'audio_count', 'val_samples', 'roc_auc', 'avg_precision']]
    df_b = df_b.set_index('label_id')[['roc_auc', 'avg_precision']]

    merged = df_a.join(df_b, rsuffix='_b').rename(columns={
        'roc_auc':       f'auc_{name_a}',
        'avg_precision': f'ap_{name_a}',
        'roc_auc_b':     f'auc_{name_b}',
        'avg_precision_b': f'ap_{name_b}',
    })
    merged['auc_diff'] = merged[f'auc_{name_b}'] - merged[f'auc_{name_a}']
    merged = merged.dropna(subset=[f'auc_{name_a}', f'auc_{name_b}'])

    out_compare = base_dir / 'models' / f'compare_{name_a}_vs_{name_b}.csv'
    merged.sort_values('auc_diff').to_csv(out_compare, encoding='utf-8-sig')
    print(f"\n比較結果已儲存：{out_compare.name}")

    # ── 整體摘要 ──────────────────────────────────────────────────────
    mean_a = merged[f'auc_{name_a}'].mean()
    mean_b = merged[f'auc_{name_b}'].mean()
    improved = (merged['auc_diff'] > 0).sum()
    degraded = (merged['auc_diff'] < 0).sum()

    print(f"\n{'='*65}")
    print(f"  整體比較（{len(merged)} 個物種）")
    print(f"{'='*65}")
    print(f"  {name_a:<20} 平均 AUC：{mean_a:.4f}")
    print(f"  {name_b:<20} 平均 AUC：{mean_b:.4f}")
    print(f"  差異（B - A）：{mean_b - mean_a:+.4f}")
    print(f"  B 優於 A：{improved} 種 | A 優於 B：{degraded} 種")

    # ── 進步最多的 10 種 ──────────────────────────────────────────────
    print(f"\n【{name_b} 進步最多的 10 種】")
    print(f"  {'common_name':<35} {'class':<10} {name_a:>8} {name_b:>8} {'diff':>7}")
    print('  ' + '-'*72)
    for _, r in merged.sort_values('auc_diff', ascending=False).head(10).iterrows():
        print(f"  {str(r['common_name']):<35} {str(r['class_name']):<10} "
              f"{r[f'auc_{name_a}']:>8.4f} {r[f'auc_{name_b}']:>8.4f} "
              f"{r['auc_diff']:>+7.4f}")

    # ── 退步最多的 10 種 ──────────────────────────────────────────────
    print(f"\n【{name_b} 退步最多的 10 種】")
    print(f"  {'common_name':<35} {'class':<10} {name_a:>8} {name_b:>8} {'diff':>7}")
    print('  ' + '-'*72)
    for _, r in merged.sort_values('auc_diff').head(10).iterrows():
        print(f"  {str(r['common_name']):<35} {str(r['class_name']):<10} "
              f"{r[f'auc_{name_a}']:>8.4f} {r[f'auc_{name_b}']:>8.4f} "
              f"{r['auc_diff']:>+7.4f}")

    # ── 各類別平均 AUC 比較 ───────────────────────────────────────────
    print(f"\n【各生物類別平均 AUC 比較】")
    print(f"  {'class':<12} {name_a:>8} {name_b:>8} {'diff':>7}")
    print('  ' + '-'*40)
    for cls, grp in merged.groupby('class_name'):
        a = grp[f'auc_{name_a}'].mean()
        b = grp[f'auc_{name_b}'].mean()
        print(f"  {str(cls):<12} {a:>8.4f} {b:>8.4f} {b-a:>+7.4f}")


if __name__ == '__main__':
    main()
