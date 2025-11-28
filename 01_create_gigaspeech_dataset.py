#pip install pandas soundfile librosa tqdm

import os
import json
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import re


# --- CẤU HÌNH ---
ROOT_URL = '/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3'
ROOT_URL = Path(ROOT_URL)
AUDIO_FOLDER = ROOT_URL / "audios"      # Thư mục chứa file wav
CSV_FILE = ROOT_URL / "metadata.csv"    # File chứa transcript
OUTPUT_JSON = ROOT_URL / "gigaspeech2_format.json"
LANGUAGE = "vi"              # Mã ngôn ngữ

def create_gigaspeech_dataset():
    # 1. Đọc file transcript
    df = pd.read_csv(CSV_FILE)
    
    dataset_list = []

    print("Đang xử lý dữ liệu...")
    
    # 2. Duyệt qua từng dòng trong CSV
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row['filename']
        text = str(row['transcript']).strip()        
		#0-1 0 = folder, 1 = fileid

        filename_parts = re.split(r'[-]', filename)
		
        file_path = os.path.join(AUDIO_FOLDER, filename_parts[0], filename)
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            print(f"Cảnh báo: Không tìm thấy file {filename}")
            continue
            
        try:
            # 3. Lấy thông tin duration từ file audio
            # sf.info rất nhanh vì nó chỉ đọc header của file audio
            audio_info = sf.info(file_path)
            duration = audio_info.duration
            
            # 4. Tạo entry theo chuẩn
            entry = {
                "segment_id": os.path.splitext(filename)[0],
                "audio_path": file_path,
                "duration": float(f"{duration:.2f}"), # Làm tròn 2 chữ số thập phân
                "text": text,
                "lang": LANGUAGE,
                # Các trường bổ sung nếu cần
                "sample_rate": audio_info.samplerate 
            }
            
            dataset_list.append(entry)
            
        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

    # 5. Ghi ra file JSON
    # GigaSpeech thường dùng định dạng JSON tổng hoặc JSONL
    # Ở đây ta lưu JSON thường để dễ nhìn
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, ensure_ascii=False, indent=2)

    print(f"Xong! Đã tạo {len(dataset_list)} mẫu dữ liệu tại {OUTPUT_JSON}")

if __name__ == "__main__":
    create_gigaspeech_dataset()