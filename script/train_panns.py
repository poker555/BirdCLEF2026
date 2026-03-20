import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pathlib import Path

# 將上層目錄加入以便從 script 資料夾直接執行
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from panns.dataset import PANNsDataset
from panns.model import PANNsCNN10

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"目前運算裝置是: {device}")
    
    # 這裡的路徑假設您在 d:\BirdCLEF2026 目錄底下執行 python script/train_panns.py
    # 確保抓取到正確的 CSV 路徑
    base_dir = Path("d:/BirdCLEF2026")
    train_csv = base_dir / "train.csv"
    labels_csv = base_dir / "taxonomy_encoded.csv"
    audio_dir = base_dir / "train_audio"
    
    df = pd.read_csv(train_csv)
    tax_df = pd.read_csv(labels_csv)
    df = df.merge(tax_df[['primary_label', 'label_id']], on='primary_label', how='left')
    
    # 切分訓練集與驗證集 (依照 label_id 分布)
    # 注意如果有類別數量少於2的可能無法 stratify，這裡加上防呆機制或者您可以移除 stratify
    try:
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])
    except ValueError:
        print("警告: 資料分布極端無法 stratify，改為隨機切分。")
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    print(f"分配完畢: 訓練集 {len(df_train)} 筆, 驗證集 {len(df_val)} 筆")

    # 建立 Dataset 與 DataLoader
    train_dataset = PANNsDataset(df_train, audio_dir=audio_dir, sample_rate=32000, max_length=5)
    val_dataset = PANNsDataset(df_val, audio_dir=audio_dir, sample_rate=32000, max_length=5)

    # 啟動多核心並行載入資料 (num_workers=8)，並裝滿 VRAM 以吃滿 GPU
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # 建立 PANNs 模型 (我們的鳥類一共有 234 類)
    model = PANNsCNN10(classes_num=234).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 加入動態學習率 (ReduceLROnPlateau)：如果 F1 Score 連續 2 次沒突破，就把學習率砍半 (factor=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 30
    print("開始 PANNs 模型訓練")

    # Early Stopping 設定：當 F1 連續 5 次沒突破，提早停止訓練
    patience = 5
    best_f1 = 0.0
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms, labels = waveforms.to(device), labels.to(device)

            optimizer.zero_grad()

            use_mixup = np.random.rand() > 0.5

            # 重新加入混合精度訓練！並指定使用 torch.bfloat16 來完美防禦對數轉換產生的 NaN 崩潰
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if use_mixup:
                    mixed_waveforms, targets_a, targets_b, lam = mixup_data(waveforms, labels)
                    outputs = model(mixed_waveforms)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(waveforms)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch[{epoch+1}/{epochs}] | Batch[{batch_idx}/{len(train_loader)}] | 誤差 Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 結束 | 平均 Loss: {avg_loss:.4f}\n")

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(waveforms)
                
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # 計算 Macro F1 Score
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # 取得當前的學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} 驗證 F1 Score: {val_f1:.4f} | 當前學習率: {current_lr:.6f}")

        # 呼叫動態學習率排程器 (依據 val_f1 的表現來決定是否要降學習率)
        scheduler.step(val_f1)

        # Early Stopping 邏輯
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            # 儲存到 d:\BirdCLEF2026\best_panns_model.pth
            torch.save(model.state_dict(), base_dir / 'best_panns_model.pth')
            print("==> 🌟 突破紀錄！最佳模型已儲存")
        else:
            counter += 1
            print(f"F1 分數沒有提升，累積 {counter} 次 (距 Early Stopping 還有 {patience - counter} 次)")
            if counter >= patience:
                print('\n==> ⛔ 達到 Early Stopping 條件，提早結束訓練！')
                break

    print("完成訓練流程！")

if __name__ == '__main__':
    main()
