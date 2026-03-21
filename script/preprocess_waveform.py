"""
PANNs 專用波形前處理腳本（含人聲靜音 + soft label）
輸出：processed_data/train_waveforms.h5
使用方式：
    python script/preprocess_waveform.py
"""

import ast
import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

SAMPLE_RATE      = 32000
NUM_CLASSES      = 234
SECONDARY_WEIGHT = 0.3
VOICE_CSV        = Path("voice_detection_report.csv")
OUTPUT_H5        = Path("processed_data/train_waveforms.h5")
AUDIO_DIR        = Path("train_audio")


def build_label_map(tax_df: pd.DataFrame) -> dict:
    label_map = {}
    for _, row in tax_df.iterrows():
        label_map[str(row['primary_label'])] = int(row['label_id'])
        label_map[str(row['inat_taxon_id'])] = int(row['label_id'])
    return label_map


def build_soft_label(primary_id: int, secondary_labels_str: str,
                     label_map: dict, num_classes: int) -> np.ndarray:
    label = np.zeros(num_classes, dtype=np.float32)
    label[primary_id] = 1.0
    try:
        for s in ast.literal_eval(secondary_labels_str):
            sid = label_map.get(str(s))
            if sid is not None and sid != primary_id:
                label[sid] = SECONDARY_WEIGHT
    except Exception:
        pass
    return label


def load_voice_map(csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"[警告] 找不到 {csv_path}，將不進行人聲過濾。")
        return {}
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df[df['has_voice'] == True]
    voice_map = {}
    for _, row in df.iterrows():
        key = '/'.join(str(row['filename']).replace('\\', '/').split('/')[-2:])
        segments = []
        if pd.notna(row['segments_detail']) and str(row['segments_detail']).strip():
            for seg in str(row['segments_detail']).split('|'):
                parts = seg.strip().split('-')
                if len(parts) == 2:
                    try:
                        segments.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        voice_map[key] = segments
    return voice_map


def silence_voice_segments(waveform: np.ndarray, voice_segments: list, sr: int) -> np.ndarray:
    y = waveform.copy()
    for start_sec, end_sec in voice_segments:
        y[int(start_sec * sr):min(int(end_sec * sr), len(y))] = 0.0
    return y


def process_single_audio(task: dict) -> dict:
    import librosa
    warnings.filterwarnings('ignore', category=UserWarning)

    filename   = task['filename']
    voice_segs = task.get('voice_segments', [])
    file_path  = AUDIO_DIR / filename

    try:
        y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        if voice_segs:
            y = silence_voice_segments(y, voice_segs, SAMPLE_RATE)
        return {
            'status': 'success',
            'filename': filename,
            'waveform': y.astype(np.float32)
        }
    except Exception as e:
        return {'status': 'failed', 'filename': filename, 'error_msg': str(e)}


def main():
    OUTPUT_H5.parent.mkdir(exist_ok=True)

    print("讀取資料表...")
    df = pd.read_csv("train.csv")
    tax_df = pd.read_csv("taxonomy_encoded.csv")
    df = df.merge(tax_df[['primary_label', 'label_id', 'class_id']], on='primary_label', how='left')

    label_map = build_label_map(tax_df)

    print("載入人聲過濾清單...")
    voice_map = load_voice_map(VOICE_CSV)
    print(f"共 {len(voice_map)} 支音訊含有人聲，將對其進行靜音處理。")

    tasks = []
    task_map = {}
    for _, row in df.iterrows():
        key = '/'.join(str(row['filename']).replace('\\', '/').split('/')[-2:])
        soft_label = build_soft_label(
            int(row['label_id']), str(row['secondary_labels']), label_map, NUM_CLASSES
        )
        fname = row['filename']
        task_map[fname] = (soft_label, int(row['class_id']))
        tasks.append({
            'filename': fname,
            'voice_segments': voice_map.get(key, [])
        })

    # worker 數量：保留足夠核心給系統，避免 OOM
    NUM_WORKERS = 4

    print(f"共 {len(tasks)} 筆音訊，開始多核心處理並寫入 {OUTPUT_H5}（workers={NUM_WORKERS}）...")
    success = failed = 0

    with h5py.File(OUTPUT_H5, 'w') as h5_file:
        h5_file.attrs['sample_rate'] = SAMPLE_RATE
        # maxtasksperchild=50：每個 worker 處理 50 筆後自動重啟，釋放 librosa 殘留記憶體
        # chunksize=1：主 process 逐筆取結果，避免 queue 堆積大量波形資料
        with Pool(processes=NUM_WORKERS, maxtasksperchild=50) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_audio, tasks, chunksize=1),
                total=len(tasks)
            ):
                if result['status'] == 'success':
                    soft_label, class_id = task_map[result['filename']]
                    ds = h5_file.create_dataset(
                        name=result['filename'],
                        data=result['waveform'],
                        compression='gzip', compression_opts=4
                    )
                    ds.attrs['soft_label']  = soft_label
                    ds.attrs['class_id']    = class_id
                    ds.attrs['sample_rate'] = SAMPLE_RATE
                    success += 1
                else:
                    print(f"\n[錯誤] {result['filename']} - {result['error_msg']}")
                    failed += 1
                # 主 process 每筆處理完後主動釋放 result 物件
                del result

    print(f"\n完成！成功 {success} 筆，失敗 {failed} 筆。")
    print(f"HDF5 已儲存至：{OUTPUT_H5.resolve()}")


if __name__ == '__main__':
    main()
