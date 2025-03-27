from binaryclass.dataset import L2ArcticDataset, VCTKDataset, SimpleData
from binaryclass.model import LinearModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm  # 導入 tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from integrated_gradient import get_attribution_score


batch_size = 32
max_length = 300
save_epoch = 1

# OOD = Out of distribution, which is those Speaker NOT used in training.

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

def collate_fn_noLabel(batch, max_length):
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
    
    return features, labels

def custom_collate_fn(batch):
    return collate_fn(batch, max_length)
def custom_collate_fn_noLabel(batch):
    return collate_fn_noLabel(batch, max_length)
def evaluate(model, data_loader, device, max_samples=5000):
    model.eval()  # 設置為評估模式
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            batch_size = labels.size(0)
            
            if total + batch_size > max_samples:
                batch_size = max_samples - total  # 只取所需數量
                correct += (predicted[:batch_size] == labels[:batch_size]).sum().item()
                total += batch_size
                break
            
            correct += (predicted == labels).sum().item()
            total += batch_size
            
            if total >= max_samples:
                break

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def evaluate_noLabel(model, data_loader, device):
    model.eval()  # 設置為評估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, dataName in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += inputs.size(0)
            labels = torch.ones(inputs.size(0)).to(device)
            #print(f'predicted:{predicted}')
            #print(f'labels:{labels}')
            correct_num = (predicted == labels).sum().item()
            correct += correct_num
            if correct_num != batch_size:
                indices = torch.where(~(predicted == labels))
                # for i in indices[0]:
                #     print(dataName[i.item()])

    accuracy = 100 * correct / total
    return accuracy

def main():
    # 創建 L2ArcticDataset 和 VCTKDataset 的實例
    import torch

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 檢查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 這裡必須要手動分割，將欲排除之speaker資料夾路徑填入

    l2arctic_dataset = L2ArcticDataset("/work/u1284878/hw2/feature/l2arctic/test")
    vctk_dataset = VCTKDataset("/work/u1284878/hw2/feature/VCTK")
    #simple_data = SimpleData("/work/u1284878/hw2/feature/TTS")

    
    l2_loader = DataLoader(
        l2arctic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # 减少worker数量，默认值可能较高
        pin_memory=True,  # 对GPU训练有帮助
        collate_fn=custom_collate_fn
    )
    
    vctk_loader = DataLoader(
        vctk_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    '''
    simple_loader = DataLoader(
        simple_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn_noLabel
    )
    '''
    checkpoint = torch.load("/work/u1284878/hw2/code/checkpoints/checkpoint_epoch_finetune_on_TED_2.pth")  
    model = LinearModel(input_dim=1024, output_class_num=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating L2----------------------------")
    print(evaluate(model, l2_loader, device))
    print("Evaluating VCYK----------------------------")
    print(evaluate(model, vctk_loader, device))
    #print(evaluate_noLabel(model, simple_loader, device))


if __name__ == "__main__":
    main()