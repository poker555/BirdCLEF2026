
import pandas as pd 
import soundfile as sf
import os
from multiprocessing import Pool
from pathlib import Path

def get_duration(file_path):
    try:
        if os.path.exists(file_path):
            info = sf.info(file_path)
            return info.duration
        return 0.0
    except Exception:
        return 0.0

        
def main():
    print("計算數量中")
    df = pd.read_csv('train.csv')
    # file_counts = df['primary_label'].value_counts()
    # print("數量最多的前五名物種: \n", file_counts.head())
    # counts_df = file_counts.reset_index()
    # counts_df.columns = ['primary_label','audio_count']
    # counts_df.to_csv('species_counts.csv',index = False)
    # print('第一步完成')

    audio_dir = Path('train_audio')
    file_paths = [str(audio_dir/name) for name in df['filename']]
    print(f"準備測量{len(file_paths)}個音檔的長度，啟動平行處理")
    with Pool() as pool:
        durations = pool.map(get_duration,file_paths)
    df['duration_sec'] = durations
    summary_df = df.groupby('primary_label').agg(
        audio_count = ('filename','count'),
        total_duration_sec = ('duration_sec','sum')
    ).reset_index()

    summary_df = summary_df.sort_values(by='total_duration_sec',ascending=False)
    summary_df.to_csv('species_counts.csv',index=False)
    print("\n前五名資料最豐富的物種")
    print(summary_df.head())
    print("\n分析成功")

    

if __name__ == '__main__':
        main()

