import os
import shutil
from pathlib import Path

# --- CẤU HÌNH ---
ROOT_URL = Path("/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3")
BPE_VOCAB_FILE = ROOT_URL / "lang_bpe_500/bpe_500.vocab"  # File sinh ra cùng với bpe_500.model
LANG_DIR = ROOT_URL / "lang_bpe_500"               # Thư mục đích

def main():
    print(f"Đang tạo lang_dir tại: {LANG_DIR}")
    os.makedirs(LANG_DIR, exist_ok=True)

    # 1. Đọc file vocab từ SentencePiece
    # File này có dạng: <token> <score>
    with open(BPE_VOCAB_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. Tạo file tokens.txt
    # Chuẩn Icefall: <blk> là 0, các token khác bắt đầu từ 1
    tokens_path = os.path.join(LANG_DIR, "tokens.txt")
    
    with open(tokens_path, "w", encoding="utf-8") as f:
        # Ghi ký tự Blank đầu tiên (bắt buộc)
        f.write("<blk> 0\n")
        
        # Duyệt qua vocab cũ và shift ID lên +1
        for i, line in enumerate(lines):
            # Lấy token (cột đầu tiên), bỏ qua phần score
            token = line.strip().split()[0]
            
            # ID mới = ID cũ (i) + 1
            new_id = i + 1
            f.write(f"{token} {new_id}\n")

    print(f"--> Đã tạo {tokens_path}")
    print(f"--> Tổng số tokens (gồm blank): {len(lines) + 1}")

    # # 3. Copy file bpe.model vào thư mục lang luôn cho tiện
    # bpe_model_src = BPE_VOCAB_FILE.replace(".vocab", ".model")
    # if os.path.exists(bpe_model_src):
    #     shutil.copy(bpe_model_src, os.path.join(LANG_DIR, "bpe.model"))
    #     print("--> Đã copy bpe.model vào lang_dir")


    

    # 4. Tạo file L.pt (Lexicon rỗng) - Icefall yêu cầu file này tồn tại
    # Dù training stateless (không dùng graph) nhưng code check file này thường vẫn chạy
    # Để tạo L.pt chuẩn cần k2, nhưng ta có thể tạo file dummy nếu chỉ train Zipformer
    print("Lưu ý: Để tạo L.pt (Lexicon FSA), bạn cần cài đặt k2 và chạy script chuẩn.")
    print("Tuy nhiên với Zipformer/Transducer, tokens.txt là quan trọng nhất.")

if __name__ == "__main__":
    main()