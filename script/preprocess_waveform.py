"""
PANNs 專用波形前處理腳本
將原始 .ogg 音訊重新取樣至 32kHz 並儲存原始波形至 HDF5
輸出：processed_data/train_waveforms.h5
使用方式：
    python script/preprocess_waveform.py
"""

import warnings
import numpy as np
import pandas as pd
import h5py
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ── 設定 ──────────────────────────────────────────────
SAMPLE_RATE = 32000
OUTPUT_H5   = Path("processed_data/train_waveforms.h5")
TRAIN_CSV   = Path("train.csv")
TAXONOMY_CSV = Path("taxonomy_encoded.csv")
AUDIO_DIR   = Path("train_audio")
# ──────────────────────────────────────────────────────


def process_single_audio(task: dict) -> dict:
    """載入單一音訊並回傳原始波形，在 worker process 中執行"""
    import librosa  # 在 worker 內 import 避免 multiprocessing 問題

    warnings.filterwarnings('ignore', category=UserWarning)

    filename = task['filename']
    label_id = task['label_id']
    file_path = AUDIO_DIR / filename

    try:
        waveform, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        return {
            'status': 'success',
            'filename': filename,
            'label_id': label_id,
            'waveform': waveform.astype(np.float32)
        }
    except Exception as e:
        return {
            'status': 'failed',
            'filename': filename,
            'error_msg': str(e)
        }


def main():
    OUTPUT_H5.parent.mkdir(exist_ok=True)

    print("讀取資料表...")
    df = pd.read_csv(TRAIN_CSV)
    tax_df = pd.read_csv(TAXONOMY_CSV)
    df = df.merge(tax_df[['primary_label', 'label_id']], on='primary_label', how='left')

    tasks = df[['filename', 'label_id']].to_dict('records')
    print(f"共 {len(tasks)} 筆音訊，開始多核心處理並寫入 {OUTPUT_H5}...")

    success_count = 0
    fail_count = 0

    with h5py.File(OUTPUT_H5, 'w') as h5_file:
        # 儲存取樣率供後續讀取時參考
        h5_file.attrs['sample_rate'] = SAMPLE_RATE

        with Pool() as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_audio, tasks, chunksize=8),
                total=len(tasks)
            ):
                if result['status'] == 'success':
                    ds = h5_file.create_dataset(
                        name=result['filename'],
                        data=result['waveform'],
                        compression='gzip',
                        compression_opts=4  # 波形資料量大，用適中壓縮率平衡速度與空間
                    )
                    ds.attrs['label_id'] = result['label_id']
                    ds.attrs['sample_rate'] = SAMPLE_RATE
                    success_count += 1
                else:
                    print(f"\n[錯誤] {result['filename']} - {result['error_msg']}")
                    fail_count += 1

    print(f"\n完成！成功 {success_count} 筆，失敗 {fail_count} 筆。")
    print(f"HDF5 已儲存至：{OUTPUT_H5.resolve()}")


if __name__ == '__main__':
    main()
