from dataset import L2ArcticDataset, VCTKDataset, SimpleData
from model import LinearModel
from train import collate_fn, evaluate, save_checkpoint
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
from captum.attr import IntegratedGradients
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
            #print(predicted)
            correct_num = (predicted == labels).sum().item()
            correct += correct_num

    accuracy = 100 * correct / total
    return accuracy

def attribution_score(features_tensor, model, native_labels_tensor):
    prev_mode = model.training
    # Ensure the model is in evaluation mode
    model.eval()

    # Initialize the IntegratedGradients object
    ig = IntegratedGradients(model)
    
    # Create an empty tensor to hold the attribution scores for each sample in the batch
    attributions_batch = []

    # Process each sample in the batch
    for i in range(features_tensor.size(0)):  # Iterating over batch dimension
        # Get the features and the native label for the current sample
        input_sample = features_tensor[i].unsqueeze(0)  # Shape [1, ...] for single sample
        target_class_index = native_labels_tensor[i].item()  # Get the label for the current sample
        
        # Calculate the attribution using Integrated Gradients for the current sample
        attributions, approximation_error = ig.attribute(input_sample, target=target_class_index,
                                                         return_convergence_delta=True)
        
        # Append the attributions to the batch list
        attributions_batch.append(attributions.squeeze(0))  # Remove the extra batch dimension

    # Stack the attributions into a single tensor with shape [batch_size, ...]
    attributions_batch = torch.stack(attributions_batch)
        # 恢復到原來的模式
    if prev_mode:  
        model.train()

    return attributions_batch

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f"Loaded checkpoint from epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")
    return model, optimizer, epoch
        
def weighted_mse_loss(pred, target, weight_for_zero=1.0, weight_for_one=10.0):
    
    weights = torch.where(target == 1, weight_for_one, weight_for_zero)

    return (weights * (pred - target/1000) ** 2).mean()*1000

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
def custom_collate_fn_noLabel(batch):
    return collate_fn_noLabel(batch, max_length=300)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    max_length = 300
    save_epoch = 2

    l2arctic_dataset = L2ArcticDataset("/work/u1284878/hw2/feature/l2arctic/train")
    vctk_dataset = VCTKDataset("/work/u1284878/hw2/feature/VCTK_remove_silence")
    vctk_dataset = torch.utils.data.Subset(vctk_dataset, range(min(2000, len(vctk_dataset))))

    combined_dataset = ConcatDataset([l2arctic_dataset, vctk_dataset])

    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)  # 80% 作為訓練集 修改小數點，以選擇測試集比例
    test_size = total_size - train_size  # 剩餘 20% 作為測試集
    train_data, test_data = random_split(combined_dataset, [train_size, test_size])
    simple_data = SimpleData("/work/u1284878/hw2/feature/ted_trim")
    # 將 Train 和 test 分兩個 Dataloader 
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  
        pin_memory=True, 
        collate_fn=lambda batch: collate_fn(batch, max_length)
    )

    test_loader =  DataLoader(
        simple_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn_noLabel
    )
    model = LinearModel(input_dim=1024, output_class_num=2).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 訓練過程
    num_epochs = 10
    accumulation_steps = 1 
    checkpoint_dir = "/work/u1284878/hw2/code/checkpoints" #save directory
    checkpoint_path = "/work/u1284878/hw2/code/checkpoints/checkpoint_epoch_add_TED3.pth" # log saved checkpoint .pth

    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    curEpoch = 0
    for epoch in range(num_epochs):
        model.train()  # 設置模型為訓練模式
        curEpoch += 1 # 用來計數跑過的Epoch，每存檔一次就歸零
        
        running_loss = 0.0
        running_attribution_loss = 0.0  # 初始化 running_attribution_loss
        correct = 0
        total = 0
        optimizer.zero_grad()  # 清空梯度，初始化為零

        # 使用 tqdm 包裝 train_loader 來顯示進度條
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            # In your training loop
            for i, (inputs, native_labels, anno_labels) in pbar:
                inputs, native_labels, anno_labels = inputs.to(device), native_labels.to(device), anno_labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Main classification loss (cross-entropy)
                loss = criterion(outputs, native_labels)

                # Compute attribution scores dynamically based on the native label
                attributions = attribution_score(inputs, model, native_labels)
                attributions_avg = attributions.mean(dim=2)  # 在特徵維度上做平均
                
                # Compute attribution loss (e.g., MSE between attributions and ground truth annotations)
                #print(attributions_avg[0] , anno_labels[0])
                attribution_loss = weighted_mse_loss(attributions_avg, anno_labels, weight_for_zero=1.0, weight_for_one=100.0)
                # Combine classification loss and attribution loss
                total_loss = loss + attribution_loss

                # Backward pass
                total_loss.backward()

                # Gradient accumulation
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Statistics
                running_loss += loss.item()
                running_attribution_loss += attribution_loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += native_labels.size(0)
                correct += (predicted == native_labels).sum().item()

                # Update progress bar
                pbar.set_postfix(loss=running_loss/(i+1),
                                 attribution_loss=running_attribution_loss/(i+1),
                                 accuracy=100 * correct/total)
                                # 清理未使用的 GPU 記憶體
                torch.cuda.empty_cache()

        # 輸出訓練狀況
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        if curEpoch == save_epoch:
            save_checkpoint(model, optimizer, epoch+1, running_loss/len(train_loader), 100 * correct / total, checkpoint_dir, filename=f"checkpoint_epoch_finetune_on_TED_{epoch+1}.pth")
            curEpoch = 0
        # 在每個 epoch 結束後評估測試集
        #test_accuracy = evaluate(model, test_loader, device)
        test_accuracy = evaluate_noLabel(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()