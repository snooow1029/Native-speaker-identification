import os
import re
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_sentences(word):
    url = f"https://dictionary.cambridge.org/us/dictionary/english/{word}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Word: {word} web crawler error: {e}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    container = soup.find_all('span', class_='eg deg')
        
    sentences = [span.get_text() for span in container]
    return sentences

def get_pronunciation(word):
    base_url = "https://dictionary.cambridge.org/us/dictionary/english/"
    url = base_url + word
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Word: {word} web crawler error: {e}")
        return None, None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    us_block = soup.find("span", class_="us dpron-i")
    if not us_block:
        print("kk DNE")
        return None, None

    pron_span = us_block.find("span", class_="pron dpron")
    if pron_span:
        us_phonetic = pron_span.get_text(strip=True)
    else:
        us_phonetic = None

    audio_tag = us_block.find("audio")
    if audio_tag:
        source_tag = audio_tag.find("source", type="audio/mpeg")
        if source_tag:
            src = source_tag.get("src")
            if src.startswith("/"):
                us_mp3_url = "https://dictionary.cambridge.org" + src
            else:
                us_mp3_url = src
        else:
            us_mp3_url = None
    else:
        us_mp3_url = None

    saved_path = f"mp3/{word}.mp3"
    download_mp3(us_mp3_url, saved_path)
    return us_phonetic, saved_path

def download_mp3(url, filename):
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/90.0.4430.93 Safari/537.36")
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print("Failed to download the file. HTTP Status code:", response.status_code)
    
def get_target_words(sentence):
    return re.findall(r'\b\w+\b', sentence)

def get_padded_words(words1, words2):
    if len(words1) < len(words2):
        words1.extend([''] * (len(words2) - len(words1)))
    else:
        words2.extend([''] * (len(words1) - len(words2)))

    print(f'''
        After padding: len = {len(words1)}
        {words1}, {words2}
    ''')

    return words1, words2