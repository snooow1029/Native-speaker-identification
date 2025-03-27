import os
import torch
import numpy as np
import random
import gc
import warnings

import librosa
from tqdm import tqdm
from s3prl.nn import S3PRLUpstream
import torch.nn.functional as F
from binaryclass.model import LinearModel

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from captum.attr import IntegratedGradients

from PIL import Image
import io

# 忽略所有警告
warnings.filterwarnings("ignore")


def process_wav_file(wav_file, model, device):
    model.eval()
    feature = None  # Initialize feature to avoid UnboundLocalError
    
    if not wav_file.endswith((".wav", ".flac", ".mp3")):
        raise ValueError("Input file is not a audio file")
    
    # 轉換為 PyTorch 張量，並移動到指定裝置
    try:
        wav, sr = librosa.load(wav_file, sr=16000)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)
        wav_len = torch.LongTensor([wav_tensor.shape[1]]).to(device)

        # 抽取特徵
        with torch.no_grad():
            all_hs, all_hs_len = model(wav_tensor, wav_len)
        # 獲取最後一層特徵
        feature = all_hs[-1]
        feature = feature[0]
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
    # 釋放 GPU 記憶體並進行垃圾回收
    torch.cuda.empty_cache()
    gc.collect()
    return feature

def evaluate(model, feature, device):
    model.eval()  # 設置為評估模式

    with torch.no_grad():
        feature = feature.to(device)
        outputs = model(feature)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        if predicted == 1:
            print("Classified as a native speaker.")
        elif predicted == 0:
            print("Classified as a Non-native speaker.")
        native_prob = round(probabilities[0][1].item() * 100, 2)
        print(native_prob, "% Native (Get from SoftMax function)")
        
        return predicted, native_prob

def plot_smoothed_LineChart(framewise_error, approximation_error, peaks=None, outputName=None):
    data_np = framewise_error
    x_axis = np.arange(data_np.shape[0]) * 0.02

    plt.figure()  # 每次呼叫時建立一個新 figure
    plt.plot(x_axis, data_np)
    plt.scatter(x_axis[peaks], data_np[peaks], color='red', marker='o', label="Peaks")
    # plt.axhline(y = approximation_error, color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('timewise error')
    plt.title('Attribution plot in time')
    plt.savefig("/work/u1284878/hw2/Attribution_plot/" + outputName +".png")
    plt.close()  # 繪製完成後關閉當前 figure

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(4, 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    
    buf.seek(0)
    pil_img = Image.open(buf)
    return pil_img

def load_models(isWav = True):
    downstream_ckpt = "checkpoints/checkpoint_epoch_finetune2_20.pth"

    if isWav:
        print("Loading wavlm model... \n")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        upstream_model = S3PRLUpstream("wavlm_large").to(device)
        
        downstream_model = LinearModel(input_dim=1024, output_class_num=2).to(device)
        checkpoint = torch.load(downstream_ckpt, map_location=torch.device('cpu'))
        downstream_model.load_state_dict(checkpoint["model_state_dict"])
        
        return upstream_model, downstream_model, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        downstream_model = LinearModel(input_dim=1024, output_class_num=2).to(device)
        checkpoint = torch.load(downstream_ckpt, map_location=torch.device('cpu'))
        downstream_model.load_state_dict(checkpoint["model_state_dict"])
        
        return None, downstream_model, device

def attribution_score(feature, downstream_model):
    target_class_index = 0 # 0 為 非母語者之 label

    ig = IntegratedGradients(downstream_model)
    attributions, approximation_error = ig.attribute(feature, target=target_class_index,
                                        return_convergence_delta=True)

    return attributions, approximation_error


def get_feature(input_path, upstream_model, device, isWav = True):
    if isWav and input_path.endswith((".wav", ".mp3")):
        feature = process_wav_file(input_path, upstream_model, device)
        print("feature:",feature)
        feature = feature.unsqueeze(0)
    elif input_path.endswith(".npy"):
        data = np.load(input_path, allow_pickle=True).item()
        feature = data["feature"]
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device) #
    else:
        print("You are giving a wrong file")
        return

    return feature

def get_smooth_framewise_error(attributions, sigma=2, prominence=0.00003): 
    attributions = attributions.squeeze(0)
    framewise_error = torch.mean(attributions, dim=1)
    framewise_error_np = framewise_error.detach().cpu().numpy()
    smoothed_error = gaussian_filter1d(framewise_error_np, sigma=sigma)
    
    peaks, _ = find_peaks(smoothed_error, prominence=prominence)

    return smoothed_error, peaks # 回傳平滑化後的誤差 & 峰值索引(*0.02是秒數)

def get_framewise_error(attributions): # 目前以平均來施作
    attributions = attributions.squeeze(0)
    framewise_error = torch.mean(attributions, dim=1)
    return framewise_error
    
def plot_LineChart(framewise_error, approximation_error, outputName):
    data_np = framewise_error.cpu().numpy()
    x_axis = np.arange(data_np.shape[0]) * 0.02

    plt.figure()  # 每次呼叫時建立一個新 figure
    plt.plot(x_axis, data_np)
    plt.axhline(y = approximation_error, color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('timewise error')
    plt.title('Attribution plot in time')
    plt.savefig("/work/u1284878/hw2/Attribution_plot/" + outputName +".png")
    plt.close()  # 繪製完成後關閉當前 figure

def get_attribution_score(upstream_model, downstream_model, device, input_path, isWav=True):
    feature = get_feature(input_path, upstream_model, downstream_model, device, isWav = isWav)
    attributions, approximation_error = attribution_score(feature, downstream_model)
    return attributions, approximation_error


                