import pandas as pd

# 大類排序固定，確保跨平台一致
CLASS_ORDER = ['Amphibia', 'Aves', 'Insecta', 'Mammalia', 'Reptilia']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = 5

def generate_encoded_taxonomy(input_path='taxonomy.csv', output_path='taxonomy_encoded.csv'):
    df = pd.read_csv(input_path)

    # label_id：物種 ID（按 primary_label 字母排序）
    sorted_labels = sorted(df['primary_label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    df['label_id'] = df['primary_label'].map(label_to_id)

    # class_id：大類 ID
    df['class_id'] = df['class_name'].map(CLASS_TO_ID)

    df.to_csv(output_path, index=False)
    print(f"成功為 {len(sorted_labels)} 個物種建立 label_id，{len(CLASS_TO_ID)} 個大類建立 class_id")
    print(f"大類對應：{CLASS_TO_ID}")
    print(f"檔案已儲存至 {output_path}")

if __name__ == "__main__":
    generate_encoded_taxonomy()
