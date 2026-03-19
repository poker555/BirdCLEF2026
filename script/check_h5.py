import h5py
import sys

def check_h5(file_path):
    print(f"檔案路徑: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"最上層的群組 (Groups) 總數: {len(f.keys())}")
            for group_name in list(f.keys())[:3]: # 只印前幾個避免洗版
                print(f"📁 Group: {group_name}")
                group = f[group_name]
                for dataset_name in list(group.keys())[:3]:
                    dataset = group[dataset_name]
                    print(f"  📄 Dataset: {dataset_name}")
                    print(f"      👉 維度大小 (Shape): {dataset.shape}")
                    print(f"      👉 資料型別 (Dtype): {dataset.dtype}")
                    print(f"      👉 裡面貼的標籤 (Attributes): {dict(dataset.attrs)}")
            print("...")
            
    except Exception as e:
        print(f"讀取失敗: {e}")

if __name__ == '__main__':
    check_h5('processed_data/train_spectrograms.h5')
