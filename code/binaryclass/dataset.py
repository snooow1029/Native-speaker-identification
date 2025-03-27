import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import textgrid


'''
def get_pronunciation_labels(textgrid_path, num_frames):
    """ 解析 TextGrid，將錯誤發音標註轉成 20ms 為單位的 label """
    tg = textgrid.TextGrid()
    try:
        tg.read(textgrid_path)
    except ValueError as e:
        print(f"Error reading TextGrid {textgrid_path}: {e}")
        # Return empty labels if TextGrid can't be read
        return torch.zeros(num_frames, dtype=torch.long)
    
    error_intervals = []
    ipa_tier_found = False

    for item in tg.tiers:
        if item.name == "IPA":  # 假設 IPA 層包含錯誤發音的標註
            ipa_tier_found = True
            for interval in item.intervals:
                if hasattr(interval, 'mark') and interval.mark != "":
                    # Add check to ensure interval is within bounds
                    if interval.maxTime <= tg.maxTime:
                        error_intervals.append((interval.minTime, interval.maxTime))
                    else:
                        print(f"Warning: Interval exceeds TextGrid bounds in {textgrid_path}")
    
    # 產生 20ms 為單位的錯誤標籤
    frame_rate = 0.02  # 20ms
    error_label = torch.zeros(num_frames, dtype=torch.long)

    for start, end in error_intervals:
        start_frame = int(start / frame_rate)
        end_frame = min(int(end / frame_rate), num_frames)  # Ensure we don't exceed num_frames
        error_label[start_frame:end_frame] = 1 #這個值是從saliency map估的

    return error_label

# 實作saliency loss
class L2ArcticDataset(Dataset):
    def __init__(self, root_dir, textgrid_dir = "/work/u1284878/hw2/dataset/l2arctic"):
        self.files = []
        # root_dir 是存放 .npy 文件的資料夾，例如: "/work/u1284878/hw2/feature/l2arctic/train"
        # textgrid_dir 是存放 .TextGrid 文件的資料夾，例如: "/work/u1284878/hw2/dataset/l2arctic"
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir,subdir)
            if os.path.isdir(subdir_path):
                # 在 textgrid_dir 中尋找對應的 TextGrid 文件
                textgrid_subdir_path = os.path.join(textgrid_dir, subdir, subdir, "annotation")
                
                if os.path.isdir(textgrid_subdir_path):
                    for file in os.listdir(subdir_path):
                        #print(file)
                        #print(subdir_path)
                        if file.endswith(".npy"):
                            textgrid_path = os.path.join(textgrid_subdir_path, file.replace(".npy", ".TextGrid"))
                            #print(textgrid_path)
                            if os.path.exists(textgrid_path):  # 只有當標註檔存在時才加入
                                self.files.append((os.path.join(subdir_path, file), textgrid_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy_path, textgrid_path = self.files[idx]
        data = np.load(npy_path, allow_pickle=True).item()
        feature = torch.tensor(data["feature"], dtype=torch.float32)

        # 取得發音錯誤標註
        error_label = get_pronunciation_labels(textgrid_path, feature.shape[0])
        
        # 設定 native/non-native label
        native_label = torch.tensor(0, dtype=torch.long)  # 可根據需要修改
        return feature, native_label, error_label


#實作saliency loss
class VCTKDataset(Dataset):
    def __init__(self, root_dir, frame_duration=0.02):
        """
        Initialize the dataset by traversing through the root directory 
        and collecting the paths of .npy files.
        
        Args:
            root_dir (str): Path to the root directory containing the dataset.
            frame_duration (float): Duration of each frame in seconds (default is 20ms or 0.02s).
        """
        self.files = []
        # Traverse all subdirectories and collect .npy files
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(".npy"):
                        self.files.append(os.path.join(subdir_path, file))

        self.frame_duration = frame_duration  # Duration of each frame (default is 20ms)

    def __len__(self):
        return len(self.files)

    def get_annotation_labels(self, num_seconds):
        """
        Generate annotation labels as a tensor of zeros based on the length of audio.
        The length of the annotation labels will be based on the number of 20ms frames 
        in the given duration (num_seconds).
        
        Args:
            num_seconds (float): Duration of the audio in seconds.
        
        Returns:
            torch.Tensor: A tensor of zeros with length equal to the number of frames.
        """
        num_frames = int(num_seconds / self.frame_duration)
        return torch.zeros(num_frames, dtype=torch.long)

    def __getitem__(self, idx):
        # Load the feature data from the .npy file
        data = np.load(self.files[idx], allow_pickle=True).item()
        feature = torch.tensor(data["feature"], dtype=torch.float32)

        # Assume that the time axis corresponds to the number of time frames.
        num_seconds = feature.size(0) * self.frame_duration

        # Generate annotation labels based on the duration of the audio
        anno_label = self.get_annotation_labels(num_seconds)

        # Create a dummy label (set to 1 as an example, can be replaced with actual label logic)
        label = torch.tensor(1, dtype=torch.long)

        return feature, label, anno_label
'''

class L2ArcticDataset(Dataset):
    def __init__(self, root_dir):
            self.files = []
            # 遍歷所有子目錄並收集 .npy 檔案
            for subdir in os.listdir(root_dir):
                subdir_path = os.path.join(root_dir, subdir, subdir)
                last_char = subdir[-1]
                # 此處可以忽略 classify 方法，label 改為 1
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith(".npy"):
                            self.files.append(os.path.join(subdir_path, file))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        feature = torch.tensor(data["feature"], dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)  
        return feature, label


class VCTKDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        # 遍歷所有子目錄並收集 .npy 檔案
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(".npy"):
                        self.files.append(os.path.join(subdir_path, file))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        feature = torch.tensor(data["feature"], dtype=torch.float32)
        label = torch.tensor(1, dtype=torch.long)  # 始終返回 1 作為 label  # 假設 label 是整數
        return feature, label

# 用來評估 簡單資料夾中 大量音檔
class SimpleData(Dataset):
    def __init__(self, root_dir):
        self.files = []
        if os.path.isdir(root_dir):
            for file in os.listdir(root_dir):
                if file.endswith(".npy"):
                    self.files.append(os.path.join(root_dir, file))
                else:
                    print("There is a file not .npy", file)
        else:
            print("this is Not a folder")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        feature = torch.tensor(data["feature"], dtype=torch.float32)
        label = torch.tensor(1, dtype=torch.long)  # 假設 label 是整數
        return feature, label



