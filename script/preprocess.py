import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ── 設定 ──────────────────────────────────────────────
SAMPLE_RATE = 32000
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
FMIN        = 150
FMAX        = 16000
VOICE_CSV   = Path("voice_detection_report.csv")
OUTPUT_H5   = Path("processed_data/train_spectrograms.h5")
AUDIO_DIR   = Path("train_audio")
# ──────────────────────────────────────────────────────


def load_voice_map(csv_path: Path) -> dict:
    """載入人聲時間段，回傳 { 'species/file.ogg': [(start_sec, end_sec), ...] }"""
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
    """將波形中含人聲的時間段補零（靜音）"""
    y = waveform.copy()
    for start_sec, end_sec in voice_segments:
        start_sample = int(start_sec * sr)
        end_sample   = min(int(end_sec * sr), len(y))
        y[start_sample:end_sample] = 0.0
    return y


def process_single_audio(task: dict) -> dict:
    import librosa
    warnings.filterwarnings('ignore', category=UserWarning)

    filename   = task['filename']
    label_id   = task['label_id']
    class_id   = task['class_id']
    voice_segs = task.get('voice_segments', [])
    file_path  = AUDIO_DIR / filename

    try:
        y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)

        # 將人聲時間段靜音
        if voice_segs:
            y = silence_voice_segments(y, voice_segs, SAMPLE_RATE)

        # 整支音訊轉 Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            fmin=FMIN, fmax=FMAX
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

        return {
            'status': 'success',
            'filename': filename,
            'label_id': label_id,
            'class_id': class_id,
            'mel_db': mel_db
        }

    except Exception as e:
        return {'status': 'failed', 'filename': filename, 'error_msg': str(e)}


def main():
    OUTPUT_H5.parent.mkdir(exist_ok=True)

    print("讀取資料表...")
    df = pd.read_csv("train.csv")
    tax_df = pd.read_csv("taxonomy_encoded.csv")
    df = df.merge(tax_df[['primary_label', 'label_id', 'class_id']], on='primary_label', how='left')

    print("載入人聲過濾清單...")
    voice_map = load_voice_map(VOICE_CSV)
    print(f"共 {len(voice_map)} 支音訊含有人聲，將對其進行靜音處理。")

    tasks = []
    for _, row in df.iterrows():
        key = '/'.join(str(row['filename']).replace('\\', '/').split('/')[-2:])
        tasks.append({
            'filename': row['filename'],
            'label_id': row['label_id'],
            'class_id': int(row['class_id']),
            'voice_segments': voice_map.get(key, [])
        })

    print(f"共 {len(tasks)} 筆音訊，開始多核心處理並寫入 {OUTPUT_H5}...")
    success = failed = 0

    with h5py.File(OUTPUT_H5, 'w') as h5_file:
        with Pool() as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_audio, tasks, chunksize=8),
                total=len(tasks)
            ):
                if result['status'] == 'success':
                    ds = h5_file.create_dataset(
                        name=result['filename'],
                        data=result['mel_db'],
                        compression='gzip'
                    )
                    ds.attrs['label_id'] = result['label_id']
                    ds.attrs['class_id'] = result['class_id']
                    success += 1
                else:
                    print(f"\n[錯誤] {result['filename']} - {result['error_msg']}")
                    failed += 1

    print(f"\n完成！成功 {success} 筆，失敗 {failed} 筆。")
    print(f"HDF5 已儲存至：{OUTPUT_H5.resolve()}")


if __name__ == '__main__':
    main()
