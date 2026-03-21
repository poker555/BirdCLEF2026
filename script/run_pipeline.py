"""
一鍵執行完整訓練流程：
    1. 前處理 Mel-Spectrogram HDF5（CNN 用）
    2. 前處理 Waveform HDF5（PANNs 用）
    3. 訓練 CNN 模型
    4. 訓練 PANNs 模型

使用方式（從專案根目錄執行）：
    python script/run_pipeline.py

可透過 --skip-preprocess 跳過前處理（HDF5 已存在時使用）：
    python script/run_pipeline.py --skip-preprocess

可透過 --only 指定只跑某個步驟：
    python script/run_pipeline.py --only cnn
    python script/run_pipeline.py --only panns
    python script/run_pipeline.py --only preprocess
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run(script_path: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(Path(script_path).parent.parent)  # 專案根目錄
    )
    if result.returncode != 0:
        print(f"\n[錯誤] {desc} 失敗，中止流程。")
        sys.exit(result.returncode)
    print(f"\n[完成] {desc}")


def main():
    parser = argparse.ArgumentParser(description='BirdCLEF 2026 完整訓練流程')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳過前處理步驟（HDF5 已存在時使用）')
    parser.add_argument('--only', choices=['preprocess', 'cnn', 'panns'],
                        help='只執行指定步驟')
    args = parser.parse_args()

    base = Path(os.path.dirname(os.path.abspath(__file__)))

    steps = {
        'preprocess_mel':  (str(base / 'preprocess.py'),          '步驟 1/4：前處理 Mel-Spectrogram HDF5（CNN 用）'),
        'preprocess_wav':  (str(base / 'preprocess_waveform.py'), '步驟 2/4：前處理 Waveform HDF5（PANNs 用）'),
        'cnn':             (str(base / 'train.py'),                '步驟 3/4：訓練 CNN 模型'),
        'panns':           (str(base / 'train_panns.py'),          '步驟 4/4：訓練 PANNs 模型'),
    }

    if args.only == 'preprocess':
        run(*steps['preprocess_mel'])
        run(*steps['preprocess_wav'])
    elif args.only == 'cnn':
        run(*steps['cnn'])
    elif args.only == 'panns':
        run(*steps['panns'])
    else:
        if not args.skip_preprocess:
            run(*steps['preprocess_mel'])
            run(*steps['preprocess_wav'])
        else:
            print("已跳過前處理步驟。")
        run(*steps['cnn'])
        run(*steps['panns'])

    print("\n" + "="*60)
    print("  全部流程完成！")
    print("="*60)


if __name__ == '__main__':
    main()
