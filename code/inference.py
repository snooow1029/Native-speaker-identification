import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from s3prl.nn import S3PRLUpstream
import torch.nn.functional as F
import gc
from binaryclass.model import LinearModel
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
from PIL import Image
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import random

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

def load_models(downstream_ckpt, isWav = True):
    if isWav:
        print("Loading wavlm model... \n")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        upstream_model = S3PRLUpstream("wavlm_large").to(device)
        
        downstream_model = LinearModel(input_dim=1024, output_class_num=2).to(device)
        checkpoint = torch.load(downstream_ckpt)
        downstream_model.load_state_dict(checkpoint["model_state_dict"])
        
        return upstream_model, downstream_model, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        downstream_model = LinearModel(input_dim=1024, output_class_num=2).to(device)
        checkpoint = torch.load(downstream_ckpt)
        downstream_model.load_state_dict(checkpoint["model_state_dict"])
        
        return None, downstream_model, device
def attribution_score(feature, downstream_model):
    target_class_index = 0 # 0 為 非母語者之 label

    ig = IntegratedGradients(downstream_model)
    attributions, approximation_error = ig.attribute(feature, target=target_class_index,
                                        return_convergence_delta=True)

    return attributions, approximation_error


def get_feature(input_path, upstream_model, downstream_model, device, isWav = True):
    if isWav and input_path.endswith((".wav", ".mp3")):
        feature = process_wav_file(input_path, upstream_model, device)
        feature = feature.unsqueeze(0)
    elif input_path.endswith(".npy"):
        data = np.load(input_path, allow_pickle=True).item()
        feature = data["feature"]
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device) #
    else:
        print("You are giving a wrong file")
        return

    return feature

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
                

if __name__ == "__main__":
    # 改這裡的路徑，將欲辨識之語音路徑更改於此
    input_path = "/work/u1284878/hw2/dataset/l2arctic/HJK/HJK/wav/arctic_a0003.wav" # 輸入可以是 .wav .mp3 .npy 也可以是一個資料夾包含所有需要判斷之資料
    output_attribution_plot_path = "/work/u1284878/hw2/Attribution_plot/"

    downstream_ckpt = "/work/u1284878/hw2/code/checkpoints/checkpoint_epoch_add_TED3.pth"

    isFolder = False # 如果想一次處理整個資料夾的音檔           
    isWav = True # 如果輸入為 .wav .mp3 使用 isWav = True，若為 .npy，則使用 isWav = False

    if isFolder:
        total = 0
        totalCorrect = 0
        upstream_model, downstream_model, device = load_models(downstream_ckpt, isWav = isWav) # 載入模型
        last_folder = os.path.basename(os.path.normpath(input_path))   
        output_dir = output_attribution_plot_path + last_folder   
        os.makedirs(output_dir, exist_ok=True)
        for subdir in os.listdir(input_path):
            subdir_path = os.path.join(input_path, subdir)
            feature = get_feature(subdir_path, upstream_model, downstream_model, device, isWav = isWav) # 處理 Upstream

            attributions, approximation_error = attribution_score(feature, downstream_model)
            framewise_error = get_framewise_error(attributions)
            approximation_error = approximation_error.item()
    
            output_name = last_folder + "/" + subdir[0:8]

            plot_LineChart(framewise_error, approximation_error, output_name) # 處理 attribution error

            predicted, _ = evaluate(downstream_model, feature, device) # 處理預測結果
            totalCorrect += predicted
            total += 1
        print(((totalCorrect / total).item()) * 100, " %", "identified as Native speaker.")
        print(total)

    else:
        upstream_model, downstream_model, device = load_models(isWav = isWav)
        feature = get_feature(input_path, upstream_model, downstream_model, device, isWav = isWav)
        attributions, approximation_error = attribution_score(feature, downstream_model) # 處理 Upstream

        predicted, _ = evaluate(downstream_model, feature, device) # 處理預測結果

        framewise_error = get_framewise_error(attributions)
        approximation_error = approximation_error.item()
        plot_LineChart(framewise_error, approximation_error, "1") # 處理 attribution error


