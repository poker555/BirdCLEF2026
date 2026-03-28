import ast
import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ── 設定 ──────────────────────────────────────────────
SAMPLE_RATE       = 32000
N_MELS            = 128
N_FFT             = 2048
HOP_LENGTH        = 512
FMIN              = 150
FMAX              = 16000
NUM_CLASSES       = 234
SECONDARY_WEIGHT  = 0.3
RARE_THRESHOLD    = 5     # ≤ 此筆數視為稀少物種，進行離線增強
VOICE_CSV         = Path("voice_detection_report.csv")
OUTPUT_H5         = Path("processed_data/train_spectrograms.h5")
AUDIO_DIR         = Path("train_audio")
# ──────────────────────────────────────────────────────

# 增強參數：保守設定避免破壞生理特徵
AUG_CONFIGS = [
    {'type': 'ts', 'rate': 0.9,  'suffix': '_ts09'},   # time stretch 慢 10%
    {'type': 'ts', 'rate': 1.1,  'suffix': '_ts11'},   # time stretch 快 10%
    {'type': 'ps', 'steps': 1,   'suffix': '_ps+1'},   # pitch shift +1 半音
    {'type': 'ps', 'steps': -1,  'suffix': '_ps-1'},   # pitch shift -1 半音
]


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


def augment_waveform(y: np.ndarray, aug: dict, sr: int) -> np.ndarray:
    """
    對波形做單一增強，trim/pad 回原始長度避免長度漂移。
    time_stretch 用 res_type='kaiser_fast' 降低運算量同時保留頻率精度。
    pitch_shift 用 n_fft=2048 保持與訓練頻譜一致的解析度。
    """
    import librosa
    orig_len = len(y)
    if aug['type'] == 'ts':
        y_aug = librosa.effects.time_stretch(y, rate=aug['rate'])
    else:
        y_aug = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=aug['steps'], n_fft=2048
        )
    # trim 或 pad 回原始長度
    if len(y_aug) > orig_len:
        y_aug = y_aug[:orig_len]
    elif len(y_aug) < orig_len:
        y_aug = np.pad(y_aug, (0, orig_len - len(y_aug)))
    return y_aug.astype(np.float32)


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    import librosa
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    return librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)


def process_single_audio(task: dict) -> dict:
    import librosa
    warnings.filterwarnings('ignore', category=UserWarning)

    filename   = task['filename']
    voice_segs = task.get('voice_segments', [])
    aug        = task.get('aug', None)   # None 表示原始音訊
    file_path  = AUDIO_DIR / filename

    try:
        y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        if voice_segs:
            y = silence_voice_segments(y, voice_segs, SAMPLE_RATE)
        if aug is not None:
            y = augment_waveform(y, aug, SAMPLE_RATE)
        mel_db = wav_to_mel(y)
        # aug 的 HDF5 key 加後綴區分
        h5_key = filename + (aug['suffix'] if aug else '')
        return {'status': 'success', 'h5_key': h5_key, 'mel_db': mel_db}
    except Exception as e:
        h5_key = filename + (aug['suffix'] if aug else '')
        return {'status': 'failed', 'h5_key': h5_key, 'error_msg': str(e)}


def main():
    OUTPUT_H5.parent.mkdir(exist_ok=True)

    print("讀取資料表...")
    df = pd.read_csv("train.csv")
    tax_df = pd.read_csv("taxonomy_encoded.csv")
    sc_df  = pd.read_csv("species_counts.csv")
    df = df.merge(tax_df[['primary_label', 'label_id', 'class_id']], on='primary_label', how='left')

    label_map = build_label_map(tax_df)
    rare_species = set(sc_df[sc_df['audio_count'] <= RARE_THRESHOLD]['primary_label'].astype(str))
    print(f"稀少物種（≤{RARE_THRESHOLD}筆）共 {len(rare_species)} 種，將進行離線增強。")

    voice_map = load_voice_map(VOICE_CSV)
    print(f"共 {len(voice_map)} 支音訊含有人聲，將對其進行靜音處理。")

    tasks    = []
    task_map = {}  # h5_key -> (soft_label, class_id)

    for _, row in df.iterrows():
        fname  = row['filename']
        species = str(row['primary_label'])
        key    = '/'.join(fname.replace('\\', '/').split('/')[-2:])
        soft_label = build_soft_label(
            int(row['label_id']), str(row['secondary_labels']), label_map, NUM_CLASSES
        )
        class_id = int(row['class_id'])
        voice_segs = voice_map.get(key, [])

        # 原始音訊
        task_map[fname] = (soft_label, class_id)
        tasks.append({'filename': fname, 'voice_segments': voice_segs, 'aug': None})

        # 稀少物種額外產生 4 個增強版本
        if species in rare_species:
            for aug in AUG_CONFIGS:
                h5_key = fname + aug['suffix']
                task_map[h5_key] = (soft_label, class_id)
                tasks.append({'filename': fname, 'voice_segments': voice_segs, 'aug': aug})

    print(f"共 {len(tasks)} 筆任務（含增強），開始多核心處理...")
    NUM_WORKERS = 4
    BATCH_SIZE  = 16
    success = failed = 0

    with h5py.File(OUTPUT_H5, 'w', rdcc_nbytes=0) as h5_file:
        with Pool(processes=NUM_WORKERS, maxtasksperchild=50) as pool:
            with tqdm(total=len(tasks)) as pbar:
                for batch_start in range(0, len(tasks), BATCH_SIZE):
                    batch = tasks[batch_start: batch_start + BATCH_SIZE]
                    results = pool.map(process_single_audio, batch)
                    for result in results:
                        if result['status'] == 'success':
                            soft_label, class_id = task_map[result['h5_key']]
                            ds = h5_file.create_dataset(
                                name=result['h5_key'],
                                data=result['mel_db'],
                                compression='gzip'
                            )
                            ds.attrs['soft_label'] = soft_label
                            ds.attrs['class_id']   = class_id
                            success += 1
                        else:
                            print(f"\n[錯誤] {result['h5_key']} - {result['error_msg']}")
                            failed += 1
                        del result
                    h5_file.flush()
                    del results
                    pbar.update(len(batch))

    print(f"\n完成！成功 {success} 筆，失敗 {failed} 筆。")
    print(f"HDF5 已儲存至：{OUTPUT_H5.resolve()}")


if __name__ == '__main__':
    main()
