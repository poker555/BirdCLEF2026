import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim import optimizer
from attr import s
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py

from cnn.dataset import BirdDataset
from cnn.model import BirdModel

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
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    h5_path = base_dir / 'processed_data/train_spectrograms.h5'
    keys = []
    with h5py.File(str(h5_path), 'r') as f:
        for group in f.keys():
            for dset in f[group].keys():
                keys.append(f"{group}/{dset}")
    
    keys_train, keys_val = train_test_split(keys, test_size=0.2, random_state=42)
    print(f"分配完畢:訓練集{len(keys_train)}筆,驗證集{len(keys_val)}筆")


    

    train_dataset = BirdDataset(h5_path, keys_train)
    val_dataset = BirdDataset(h5_path, keys_val)

    # 啟動多核心並行載入資料 (num_workers=8)，並裝滿 VRAM 以吃滿 GPU
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = BirdModel(num_classes=234).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 動態學習率 (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 100
    print("開始 CNN 模型訓練")

    patience = 5
    best_f1 = 0.0
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                use_mixup = np.random.rand() >0.5

                if use_mixup:
                    mixded_images, targets_a, targets_b, lam = mixup_data(images, labels)
                    outputs = model(mixded_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)


            

            # outputs = model(images)

            #loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            # loss.backward()

            # optimizer.step()


            if batch_idx % 10 == 0:
                print(f"Epoch[{epoch+1}/{epoch}] | Batch[{batch_idx}/{len(train_loader)}] | 誤差 Loss: {loss.item(): .4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch{epoch+1}結束 | 平均 Loss: {avg_loss: .4f}\n")

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images,labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                
                _,predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # 取得當前的學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} 驗證 F1 Score: {val_f1:.4f} | 當前學習率: {current_lr:.6f}")

        # 呼叫動態學習率排程器 (依據 val_f1 的表現來決定是否要降學習率)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), base_dir / 'models' / 'best_bird_model.pth')
            print("==> 🌟 突破紀錄！最佳模型已儲存")
        else:
            counter += 1
            print(f"F1 分數沒有提升，累積 {counter} 次 (距 Early Stopping 還有 {patience - counter} 次)")
            if counter >= patience:
                print('\n==> ⛔ 達到 Early Stopping 條件，提早結束訓練！')
                break

    
    print("完成一次訓練流程")

if __name__ == '__main__':
    main()