import librosa
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import librosa
import os
from pathlib import Path
from multiprocessing import Pool

def audio_to_melspectrogram(file_path,sr=32000,n_mels=128):
    y, _ = librosa.load(file_path,sr=sr,mono=True)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        fmin=150,fmax=16000
    )
    mdel_db = librosa.power_to_db(mel_spec,ref=np.max)
    return mdel_db

def process_single_audio(task):
    warnings.filterwarnings('ignore',category=UserWarning)
    filename = task['filename']
    label_id = task['label_id']
    file_path = f"train_audio/{filename}"

    try:
        mel_db = audio_to_melspectrogram(file_path)
        return{
            "status": "success",
            "filename": filename,
            "label_id": label_id,
            "mel_db": mel_db
        }
    except Exception as e:
        return{
            "status": "failed",
            "filename": filename,
            "error_msg": str(e)
        }

def main():
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    print("讀取資料表")
    df = pd.read_csv("train.csv")
    tax_df = pd.read_csv("taxonomy_encoded.csv")
    df = df.merge(tax_df[['primary_label','label_id']],on='primary_label',how='left')
    h5_path = output_dir / "train_spectrograms.h5"
    audio_dir = Path("train_audio")
    print(f"即將開始前處理並寫入{h5_path}")

    with h5py.File(h5_path,'w') as h5_file:

        # for idx,row in df.head(5).iterrows():
        #     filename = row['filename']
        #     label_id = row['label_id']
        #     file_path = audio_dir / filename
            
        #     try:

        #         mel_db = audio_to_melspectrogram(str(file_path))

        #         dataset = h5_file.create_dataset(
        #             name = filename,
        #             data = mel_db,
        #             compression = 'gzip'
        #         )

        #         dataset.attrs['label_id'] = label_id
        #     except Exception as e:
        #         print(f"警告:處理{filename}時發生錯誤，原因:{e}")
            
        # print("測試資料完成")

        tasks = df[['filename','label_id']].to_dict('records')

        print(f"啟動多核心處理{len(tasks)}筆資料")

        with Pool() as pool:
            for result in tqdm(pool.imap_unordered(process_single_audio,tasks),total=len(tasks)):
                if result['status'] == 'success':
                    dataset = h5_file.create_dataset(
                        name = result['filename'],
                        data = result['mel_db'],
                        compression = 'gzip'
                    )
                    dataset.attrs['label_id'] = result['label_id']
                    #print(f"成功寫入:{result['filename']}")
                else:
                    print(f"發生錯誤:{result['filename']}- 錯誤訊息:{result['error_msg']}")

        print("測試資料完成")
            

if __name__ == '__main__':
    main()
