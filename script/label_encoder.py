import pandas as pd

def generate_encoded_taxonomy(input_path='taxonomy.csv', output_path='taxonomy_encoded.csv'):
    # 1. 讀取原始的 taxonomy
    df = pd.read_csv(input_path)
    
    # 2. 為了保證跨平台一致，手動按字母排序
    # 先建立一個包含全部 primary_label 的清單並排序
    sorted_labels = sorted(df['primary_label'].unique())
    
    # 3. 建立 一個對應用的簡單字典 { "banana": 0, "roahaw": 1 ... }
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    
    # 4. 利用 pandas 的 map 功能，新增一個 'label_id' 欄位
    df['label_id'] = df['primary_label'].map(label_to_id)
    
    # 5. 存檔（建議另存新檔以免覆蓋原始資料）
    df.to_csv(output_path, index=False)
    print(f"✅ 成功為 {len(sorted_labels)} 個類別建立 ID，檔案已儲存至 {output_path}")

if __name__ == "__main__":
    generate_encoded_taxonomy()
