import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ── 設定 ──────────────────────────────────────────────
SAMPLE_RATE      = 32000
N_MELS           = 128
N_FFT            = 2048
HOP_LENGTH       = 512
FMIN             = 150
FMAX             = 16000
CHUNK_SEC        = 5                          # 每個片段長度（秒）
FRAMES_PER_CHUNK = 313                        # 5 秒對應的 time frame 數
VOICE_CSV        = Path("voice_detection_report.csv")
OUTPUT_H5        = Path("processed_data/train_spectrograms.h5")
AUDIO_DIR        = Path("train_audio")
# ──────────────────────────────────────────────────────


def load_voice_map(csv_path: Path) -> dict:
    """載入人聲時間段，回傳 { 'species/file.ogg': [(start, end), ...] }"""
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


def has_voice_overlap(voice_segments: list, start_sec: float, end_sec: float,
                      tolerance: float = 0.5) -> bool:
    """判斷時間窗口是否與人聲片段重疊超過 tolerance 秒"""
    for v_start, v_end in voice_segments:
        overlap = min(end_sec, v_end) - max(start_sec, v_start)
        if overlap > tolerance:
            return True
    return False


def process_single_audio(task: dict) -> dict:
    import librosa
    warnings.filterwarnings('ignore', category=UserWarning)

    filename  = task['filename']
    label_id  = task['label_id']
    voice_segs = task.get('voice_segments', [])
    file_path = AUDIO_DIR / filename

    try:
        y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        total_sec = len(y) / SAMPLE_RATE

        # 將整支音訊切成 5 秒片段，過濾含人聲的片段
        n_chunks = max(1, int(total_sec) // CHUNK_SEC)
        clean_chunks = []

        for i in range(n_chunks):
            start_sec = i * CHUNK_SEC
            end_sec   = start_sec + CHUNK_SEC

            if has_voice_overlap(voice_segs, start_sec, end_sec):
                continue  # 跳過含人聲的片段

            start_sample = int(start_sec * SAMPLE_RATE)
            chunk = y[start_sample:start_sample + SAMPLE_RATE * CHUNK_SEC]

            # 補零（最後一段可能不足）
            if len(chunk) < SAMPLE_RATE * CHUNK_SEC:
                chunk = np.pad(chunk, (0, SAMPLE_RATE * CHUNK_SEC - len(chunk)))

            mel_spec = librosa.feature.melspectrogram(
                y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH,
                fmin=FMIN, fmax=FMAX
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            clean_chunks.append((i, mel_db))

        if not clean_chunks:
            return {'status': 'skipped', 'filename': filename, 'reason': '所有片段均含人聲'}

        return {
            'status': 'success',
            'filename': filename,
            'label_id': label_id,
            'chunks': clean_chunks   # list of (chunk_idx, mel_db)
        }

    except Exception as e:
        return {'status': 'failed', 'filename': filename, 'error_msg': str(e)}


def main():
    OUTPUT_H5.parent.mkdir(exist_ok=True)

    print("讀取資料表...")
    df = pd.read_csv("train.csv")
    tax_df = pd.read_csv("taxonomy_encoded.csv")
    df = df.merge(tax_df[['primary_label', 'label_id']], on='primary_label', how='left')

    print("載入人聲過濾清單...")
    voice_map = load_voice_map(VOICE_CSV)
    print(f"共 {len(voice_map)} 支音訊含有人聲，將過濾其中的人聲片段。")

    # 將人聲資訊注入 task
    tasks = []
    for _, row in df.iterrows():
        key = '/'.join(str(row['filename']).replace('\\', '/').split('/')[-2:])
        tasks.append({
            'filename': row['filename'],
            'label_id': row['label_id'],
            'voice_segments': voice_map.get(key, [])
        })

    print(f"共 {len(tasks)} 筆音訊，開始多核心處理並寫入 {OUTPUT_H5}...")

    success = skipped = failed = 0

    with h5py.File(OUTPUT_H5, 'w') as h5_file:
        with Pool() as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_audio, tasks, chunksize=8),
                total=len(tasks)
            ):
                if result['status'] == 'success':
                    for chunk_idx, mel_db in result['chunks']:
                        # key 格式：species/filename.ogg/chunk_0, chunk_1, ...
                        key = f"{result['filename']}/chunk_{chunk_idx}"
                        ds = h5_file.create_dataset(
                            name=key, data=mel_db, compression='gzip'
                        )
                        ds.attrs['label_id'] = result['label_id']
                    success += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    print(f"\n[錯誤] {result['filename']} - {result['error_msg']}")
                    failed += 1

    print(f"\n完成！成功 {success} 筆，跳過(全人聲) {skipped} 筆，失敗 {failed} 筆。")
    print(f"HDF5 已儲存至：{OUTPUT_H5.resolve()}")


if __name__ == '__main__':
    main()
