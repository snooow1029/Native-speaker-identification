import os
import time
import random
import argparse
import subprocess
from tqdm import tqdm
import yt_dlp

# 設定輸出資料夾
OUTPUT_FOLDER = "/work/u1284878/ted_audio_wav"

def download_ted_youtube_audios(num_talks=100, output_folder=OUTPUT_FOLDER):
    """從 TED 的 YouTube 頻道下載音頻"""
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"開始從 YouTube 下載 {num_talks} 個 TED Talks 音頻片段...\n")
    
    # TED YouTube 頻道 URL (可以是頻道或播放清單)
    youtube_url = "https://www.youtube.com/c/TED/videos"
    
    # 首先獲取影片清單
    print("正在獲取 TED YouTube 頻道的影片清單...")
    
    ydl_opts_list = {
        "extract_flat": True,  # 不下載，只獲取清單
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    
    video_urls = []
    
    with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
        try:
            # 獲取頻道視頻清單
            info = ydl.extract_info(youtube_url, download=False)
            
            if "entries" in info:
                for entry in info["entries"]:
                    if len(video_urls) >= num_talks:
                        break
                    if entry.get("url"):
                        video_urls.append(entry["url"])
            
            # 如果沒有獲取足夠的視頻，可以嘗試使用 TED 的播放清單
            if len(video_urls) < num_talks:
                ted_playlists = [
                    "https://www.youtube.com/playlist?list=PLOGi5-fAu8bFHFD2vJ2Y5BFCb4lg3gFFy",  # TED Talks
                    "https://www.youtube.com/playlist?list=PLOGi5-fAu8bH_T9HhH9V2B5izEWZNG-PH"   # Most Popular TED Talks
                ]
                
                for playlist in ted_playlists:
                    if len(video_urls) >= num_talks:
                        break
                    try:
                        playlist_info = ydl.extract_info(playlist, download=False)
                        if "entries" in playlist_info:
                            for entry in playlist_info["entries"]:
                                if len(video_urls) >= num_talks:
                                    break
                                if entry.get("url") and entry["url"] not in video_urls:
                                    video_urls.append(entry["url"])
                    except Exception as e:
                        print(f"獲取播放清單 {playlist} 時出錯: {e}")
                        continue
            
        except Exception as e:
            print(f"獲取影片清單時出錯: {e}")
    
    print(f"成功獲取 {len(video_urls)} 個 TED Talk 影片連結")
    
    if not video_urls:
        print("無法獲取任何 TED Talk 連結，請檢查網絡連接或 YouTube 頻道 URL")
        return
    
    # 下載並處理每個視頻的音頻
    successful_downloads = 0
    failed_downloads = 0
    


    with tqdm(total=len(video_urls), desc="下載中", unit="file") as pbar:
        for i, video_url in enumerate(video_urls):
            try:
                # 設定 yt-dlp 選項：只下載音頻並轉為 wav 格式
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }],
                    "quiet": True,
                    "no_warnings": True,
                }
                
                # 下載音頻
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    title = info.get('title', f'unknown_title_{i}')
                    
                    # 處理文件名，移除不合法字符
                    safe_title = "".join([c for c in title if c.isalnum() or c in " ._-"]).strip()
                    input_filename = os.path.join(output_folder, f"{safe_title}.wav")
                    
                    # 檢查文件是否存在
                    if not os.path.exists(input_filename):
                        # 嘗試查找其他可能的文件名
                        possible_files = [f for f in os.listdir(output_folder) 
                                        if f.endswith('.wav') and not f.endswith('_10s.wav')]
                        if possible_files:
                            input_filename = os.path.join(output_folder, possible_files[-1])
                        else:
                            raise FileNotFoundError(f"無法找到下載的音頻文件: {safe_title}.wav")
                    
                    # 獲取音頻長度
                    cmd_get_duration = f'ffmpeg -i "{input_filename}" 2>&1 | grep Duration'
                    result = subprocess.run(cmd_get_duration, shell=True, capture_output=True, text=True)
                    duration_str = result.stdout.split()[1].strip(',')
                    duration = sum(x * float(t) for x, t in zip([3600, 60, 1], duration_str.split(":")))
                    
                    # 計算中間十秒的起始時間
                    start_time = max(0, (duration - 10) / 2)  # 確保不會超出範圍
                    
                    output_filename = os.path.join(output_folder, f"{safe_title}_10s.wav")
                    
                    # 使用 ffmpeg 剪輯中間 10 秒音頻
                    cmd = f'ffmpeg -y -i "{input_filename}" -ss {start_time} -t 10 -acodec pcm_s16le -ac 1 -ar 16000 "{output_filename}" -loglevel error'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"FFMPEG 錯誤: {result.stderr}")
                    
                    os.remove(input_filename)  # 刪除原始音頻文件
                    successful_downloads += 1

                    
            except Exception as e:
                print(f"\n下載或處理視頻 {i+1} 時出錯: {e}")
                failed_downloads += 1
            
            # 更新進度條
            pbar.update(1)
            pbar.set_postfix({"成功": successful_downloads, "失敗": failed_downloads})
            
            # 添加隨機延遲以避免被阻擋
            time.sleep(random.uniform(0.5, 2.0))
    
    print(f"\n下載完成！共下載 {successful_downloads} 個, 失敗 {failed_downloads} 個")
    print(f"所有 TED Talks 音頻已保存於: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 YouTube 下載 TED Talk 音頻")
    parser.add_argument("-n", "--num", type=int, default=1, help="設定要下載的數量")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_FOLDER, help="設定輸出資料夾")
    args = parser.parse_args()
    
    download_ted_youtube_audios(args.num, args.output)