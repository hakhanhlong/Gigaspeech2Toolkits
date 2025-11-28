import logging
import sentencepiece as spm
from lhotse import CutSet
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

def remove_empty_tokens(cuts_path, bpe_model_path, output_path):
    print(f"\n--- ĐANG QUÉT FILE: {cuts_path} ---")
    
    try:
        # Load kiểu Eager để xử lý an toàn
        cuts = CutSet.from_file(cuts_path).to_eager()
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # Load Tokenizer
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model_path)
    except Exception as e:
        print(f"❌ Lỗi load BPE: {e}")
        return

    good_cuts = []
    removed_count = 0

    for cut in cuts:
        try:
            # Lấy text
            if not cut.supervisions:
                removed_count += 1
                continue
                
            text = cut.supervisions[0].text
            
            # Nếu text None hoặc rỗng -> Bỏ
            if not text or str(text).strip() == "":
                removed_count += 1
                # print(f"⚠️ Bỏ ID {cut.id}: Text rỗng")
                continue

            # Encode qua BPE
            tokens = sp.encode(str(text))
            
            # --- QUAN TRỌNG NHẤT: Kiểm tra độ dài Token ---
            if len(tokens) == 0:
                removed_count += 1
                print(f"⚠️ Bỏ ID {cut.id}: Text '{text}' tạo ra 0 token.")
                continue
                
            good_cuts.append(cut)

        except Exception as e:
            removed_count += 1
            continue

    # Lưu lại
    if len(good_cuts) > 0:
        new_cuts = CutSet.from_cuts(good_cuts)
        new_cuts.to_file(output_path)
        print(f"✅ Đã giữ lại: {len(good_cuts)} câu.")
        print(f"🗑️  Đã xóa: {removed_count} câu rỗng.")
    else:
        print("❌ Cảnh báo: File sạch trơn không còn câu nào!")

if __name__ == "__main__":
    # Đường dẫn BPE chuẩn của bạn
    ROOT_URL = Path('/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3')

    BPE_MODEL = os.path.abspath(ROOT_URL / "lang_bpe_500/bpe_500.model")
    
    # Chỉ cần lọc tập VALID (dev) là quan trọng nhất lúc này
    # files_to_check = [
    #     "data/fbank/librispeech_cuts_dev-clean.jsonl.gz",
    #     "data/fbank/librispeech_cuts_dev-other.jsonl.gz",
    #     # Làm luôn tập train cho chắc
    #     "data/fbank/librispeech_cuts_train-clean-100.jsonl.gz" 
    # ]

    files_to_check = [
        ROOT_URL / "fbank/cuts_train_final.jsonl.gz",
        ROOT_URL / "fbank/cuts_valid.jsonl.gz",
        ROOT_URL / "fbank/cuts_valid_dev_other.jsonl.gz" # File bạn fake lúc nãy
    ]

    for f in files_to_check:
        remove_empty_tokens(f, BPE_MODEL, f)