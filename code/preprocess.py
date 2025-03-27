import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from s3prl.nn import S3PRLUpstream
import gc

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 設定資料夾
root_dir = "/work/u1284878/hw2/dataset/VCTK-Corpus/wav48" # "VCTK-Corpus/wav48" or "l2arctic"
output_dir = "/work/u1284878/hw2/feature/VCTK_remove_silence_add_noise"  # "VCTK" or "l2arctic"
os.makedirs(output_dir, exist_ok=True)

print("Loading wavlm model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = S3PRLUpstream("wavlm_large").to(device)
model.eval()


# 遍歷 root_dir 下所有的子資料夾
'''
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    if not os.path.isdir(subdir_path):
        continue
    
    for inner_subdir in os.listdir(subdir_path):

        inner_subdir_path = os.path.join(subdir_path, inner_subdir)
        wav_dir = os.path.join(inner_subdir_path, "wav")
        
        if not os.path.isdir(wav_dir):
            continue
        
        feature_dir = os.path.join(output_dir, subdir, inner_subdir)
        os.makedirs(feature_dir, exist_ok=True)
        
        # 獲取WAV文件列表
        wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]
        
        print(f"Processing {inner_subdir}: {len(wav_files)} files")


        for wav_file in tqdm(wav_files, desc=f"Processing {inner_subdir}"):
            npy_filename = os.path.basename(wav_file).replace(".wav", ".npy")
            npy_path = os.path.join(feature_dir, npy_filename)
            try:
                # 加載音頻文件
                wav, sr = librosa.load(wav_file, sr=16000)

                # 去除靜音部分
                wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)  # 30dB 閾值可調整

                # 確保音訊不為空
                if len(wav_trimmed) == 0:
                    print(f"Warning: {wav_file} is silent after trimming.")
                    continue
                if np.random.rand() < 0.5:
                    noise = np.random.normal(0, 0.005, wav_trimmed.shape)  # 均值0，標準差0.005，可調整
                    wav_trimmed = wav_trimmed + noise
                    wav_trimmed = np.clip(wav_trimmed, -1.0, 1.0)  # 避免超出[-1,1]範圍
                # 轉換為 PyTorch 張量，並移動到相同的裝置 (CPU/GPU)
                wav_tensor = torch.FloatTensor(wav_trimmed).unsqueeze(0).to(device)  # (1, samples)
                wav_len = torch.LongTensor([wav_tensor.shape[1]]).to(device)

                # 抽取特徵
                with torch.no_grad():
                    all_hs, all_hs_len = model(wav_tensor, wav_len)  # 確保 input 和 model 都在 GPU 上

                # 獲取最後一層特徵
                feature = all_hs[-1].cpu().numpy()  # 轉回 CPU 以便存檔

                # 保存特徵
                data = {
                    "feature": feature[0],  # 只有一個樣本，取第一個元素
                    "label": 0,
                }
                np.save(npy_path, data)

                # 清理內存
                del wav, wav_trimmed, wav_tensor, wav_len, all_hs, all_hs_len, feature

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

        # 確保釋放 GPU 記憶體
        torch.cuda.empty_cache()
        
        # 完成一個sub folder 後進行垃圾回收
        gc.collect()
        print(f"Finished processing {inner_subdir}")

print("All audio files processed successfully!")
'''
##############
#### VCTK ####
##############


# 遍歷 root_dir 下所有的子資料夾

for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    if not os.path.isdir(subdir_path):
        continue
    
    feature_dir = os.path.join(output_dir, subdir)
    os.makedirs(feature_dir, exist_ok=True)
    

    # 獲取WAV文件列表（限制最多20個）
    wav_dir = os.path.join(root_dir, subdir)
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")][:250]

    
    print(f"Processing {subdir}: {len(wav_files)} files")

    for wav_file in tqdm(wav_files, desc=f"Processing {subdir}"):
        npy_filename = os.path.basename(wav_file).replace(".wav", ".npy")
        npy_path = os.path.join(feature_dir, npy_filename)
        try:
            # 加載音頻文件 (.wav)
            wav, sr = librosa.load(wav_file, sr=16000)  # librosa 可以處理 wav 格式
            
            # 去掉音訊的頭尾靜音部分
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)  # top_db 可以調整為適合的閾值

            # 確保音訊不為空
            if len(wav_trimmed) == 0:
                print(f"Warning: {wav_file} is silent after trimming.")
                continue
            # 50% 機率加 Noise
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.005, wav_trimmed.shape)  # 均值0，標準差0.005，可調整
                wav_trimmed = wav_trimmed + noise
                wav_trimmed = np.clip(wav_trimmed, -1.0, 1.0)  # 避免超出[-1,1]範圍

            # 轉換為 PyTorch 張量，並移動到相同的裝置 (CPU/GPU)
            wav_tensor = torch.FloatTensor(wav_trimmed).unsqueeze(0).to(device)  # (1, samples)
            wav_len = torch.LongTensor([wav_tensor.shape[1]]).to(device)

            # 抽取特徵
            with torch.no_grad():
                all_hs, all_hs_len = model(wav_tensor, wav_len)  # 確保 input 和 model 都在 GPU 上

            # 獲取最後一層特徴
            feature = all_hs[-1].cpu().numpy()  # 轉回 CPU 以便存檔

            # 保存特徵
            data = {
                "feature": feature[0],  # 只有一個樣本，取第一個元素
                "label": 1,
            }
            np.save(npy_path, data)

            # 清理內存
            del wav, wav_tensor, wav_len, all_hs, all_hs_len, feature

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

        # 確保釋放 GPU 記憶體
        torch.cuda.empty_cache()

        # 完成一個子目錄後進行垃圾回收
        gc.collect()
    print(f"Finished processing {subdir}")

print("All audio files processed successfully!")


##############
####  TED ####
##############


# 遍歷 root_dir 下所有的子資料夾
'''
for wav_file in tqdm(os.listdir(root_dir), desc="Processing"):
    if not wav_file.endswith(".wav"):
        print("There is a not-wav file -- ", wav_file)
        continue
    npy_filename = os.path.basename(wav_file).replace(".wav", ".npy")
    full_path = os.path.join(root_dir ,wav_file)
    npy_path = os.path.join(output_dir, npy_filename)
    try:
        # 加載音頻文件
        wav, sr = librosa.load(full_path, sr=16000)
        wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)  # top_db 可以調整為適合的閾值

        # 確保音訊不為空
        if len(wav_trimmed) == 0:
            print(f"Warning: {wav_file} is silent after trimming.")
            continue

        # duration = len(wav) / sr

        # if duration > 15:
        #     print("Duration exceeds 10 seconds, skipping: ", wav_file)
        #     continue
        # 轉換為 PyTorch 張量，並移動到相同的裝置 (CPU/GPU)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)  # (1, samples)
        wav_len = torch.LongTensor([wav_tensor.shape[1]]).to(device)

        # 抽取特徵
        with torch.no_grad():
            all_hs, all_hs_len = model(wav_tensor, wav_len)  # 確保 input 和 model 都在 GPU 上

        # 獲取最後一層特徵
        feature = all_hs[-1].cpu().numpy()  # 轉回 CPU 以便存檔

        # 保存特徵
        data = {
            "feature": feature[0],  # 只有一個樣本，取第一個元素
            "label": 1,
        }
        np.save(npy_path, data)

        # 清理內存
        del wav, wav_tensor, wav_len, all_hs, all_hs_len, feature
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error processing {wav_file}: {e}")

    # 確保釋放 GPU 記憶體
    torch.cuda.empty_cache()

    # 完成一個sub folder 後進行垃圾回收
    gc.collect()

print("All audio files processed successfully!")
'''

