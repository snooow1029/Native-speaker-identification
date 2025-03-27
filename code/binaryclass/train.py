from dataset import L2ArcticDataset, VCTKDataset, SimpleData
from model import LinearModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm  
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
import random

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.set_num_threads(4)  
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

def collate_fn(batch, max_length):
    features, labels = zip(*batch)
    
    padded_features = []
    for feature in features:
        # 截斷長度過長的特徵
        if feature.size(0) > max_length:
            feature = feature[:max_length]
        # 填充長度不足的特徵
        elif feature.size(0) < max_length:
            padding_size = max_length - feature.size(0)
            # 填充時間維度（即第一維度），特徵維度保持不變
            feature = F.pad(feature, (0, 0, 0, padding_size), value=0.0)  # 填充時間維度
        
        padded_features.append(feature)
    
    # 使用 pad_sequence 來處理 batch 的組合
    features = torch.stack(padded_features, dim=0)
    
    # 處理 label
    labels = torch.stack(labels)  # label 不需要填充
    return features, labels
    
def evaluate(model, data_loader, device):
    model.eval()  # 設置為評估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

batch_size = 256
max_length = 300
save_epoch = 1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l2arctic_dataset = L2ArcticDataset("/work/u1284878/hw2/feature/l2arctic/train")
    vctk_dataset = VCTKDataset("/work/u1284878/hw2/feature/VCTK")
    TED_dataset = SimpleData("/work/u1284878/hw2/ted_audio_wav/ted_npy")

    combined_dataset = ConcatDataset([l2arctic_dataset, vctk_dataset, TED_dataset])

    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)  # 80% 作為訓練集 修改小數點，以選擇測試集比例
    test_size = total_size - train_size  # 剩餘 20% 作為測試集
    train_data, test_data = random_split(combined_dataset, [train_size, test_size])
    
    # 將 Train 和 test 分兩個 Dataloader 
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,  
        pin_memory=True, 
        collate_fn=lambda batch: collate_fn(batch, max_length)
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,  
        pin_memory=True, 
        collate_fn=lambda batch: collate_fn(batch, max_length)
    )

    #在這裡選擇 downstream model
    #model = CNNModel(input_channels=256, num_classes=2).to(device)
    model = LinearModel(input_dim=1024, output_class_num=2).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 訓練過程
    num_epochs = 20
    accumulation_steps = 4  # 設置累積的步數，即每四個 batch 更新一次

    checkpoint_dir = "/work/u1284878/hw2/code/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    curEpoch = 0
    for epoch in range(num_epochs):
        model.train()  # 設置模型為訓練模式
        curEpoch += 1 # 用來計數跑過的Epoch，每存檔一次就歸零
        
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()  # 清空梯度，初始化為零

        # 使用 tqdm 包裝 train_loader 來顯示進度條
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (inputs, labels) in pbar:
                # 將數據移動到 GPU
                inputs, labels = inputs.to(device), labels.to(device)
                #print(labels)
                # 前向傳播
                outputs = model(inputs)

                # 計算損失
                loss = criterion(outputs, labels)
                
                # 反向傳播
                loss.backward()

                # 梯度累積
                if (i + 1) % accumulation_steps == 0:  # 每累積4個 batch 更新一次
                    optimizer.step()
                    optimizer.zero_grad()  # 重置梯度

                # 統計
                running_loss += loss.item()

                # 計算準確率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新進度條
                pbar.set_postfix(loss=running_loss/(i+1), accuracy=100 * correct/total)

        # 輸出訓練狀況
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        if curEpoch == save_epoch:
            save_checkpoint(model, optimizer, epoch+1, running_loss/len(train_loader), 100 * correct / total, checkpoint_dir, filename=f"checkpoint_epoch_add_TED{epoch+1}.pth")
            curEpoch = 0
        # 在每個 epoch 結束後評估測試集
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()