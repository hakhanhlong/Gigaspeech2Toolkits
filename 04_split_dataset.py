import logging
import math
from pathlib import Path
from lhotse import CutSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def split_dataset(input_path, output_dir, valid_ratio=0.1):
    # Chuyển đổi string sang Path object để tránh lỗi đường dẫn
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Đang đọc dữ liệu từ: {input_path}")
    
    # --- FIX QUAN TRỌNG: .to_eager() ---
    # Lệnh này ép Lhotse load toàn bộ metadata vào RAM.
    # Nếu không có nó, bạn không thể dùng len() hoặc slicing [:] chính xác.
    try:
        cuts = CutSet.from_file(input_path).to_eager()
    except Exception as e:
        logging.error(f"Không đọc được file. Kiểm tra đường dẫn! Lỗi: {e}")
        return

    total_count = len(cuts)
    if total_count == 0:
        logging.error("File dữ liệu rỗng! Vui lòng kiểm tra lại bước prepare_data.")
        return

    # Xáo trộn ngẫu nhiên
    logging.info("Đang xáo trộn dữ liệu...")
    cuts = cuts.shuffle()

    # Tính toán số lượng
    valid_count = math.ceil(total_count * valid_ratio)
    train_count = total_count - valid_count

    logging.info("train_count = " + str(train_count))
    logging.info("valid_count = " + str(valid_count))
    

    # Cắt dữ liệu (Slicing)
    # Vì đã dùng .to_eager(), ta có thể cắt như list bình thường
    train_cuts_list = cuts[0:train_count]
    logging.info("train_cuts_list = " + str(len(train_cuts_list)))
    valid_cuts_list = cuts[train_count:]
    logging.info("valid_cuts_list = " + str(len(valid_cuts_list)))

    train_cuts = CutSet.from_cuts(train_cuts_list)
    valid_cuts = CutSet.from_cuts(valid_cuts_list)

    logging.info(f"Tổng số: {total_count}")
    logging.info(f"-> Train: {len(train_cuts_list)} (Lưu tại: cuts_train_final.jsonl.gz)")
    logging.info(f"-> Valid: {len(valid_cuts_list)} (Lưu tại: cuts_valid.jsonl.gz)")



    # Lưu file
    # Subset train
    train_cuts.to_file(output_dir / "cuts_train_final.jsonl.gz")
    # Subset valid
    valid_cuts.to_file(output_dir / "cuts_valid.jsonl.gz")
    
    logging.info("✅ Đã chia xong và lưu thành công!")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN TẠI ĐÂY ---
    ROOT_URL = Path("/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3") 
    INPUT_FILE = ROOT_URL / "fbank/cuts_train.jsonl.gz"
    OUTPUT_DIR = ROOT_URL / "fbank"
    
    # Kiểm tra file đầu vào có tồn tại không trước khi chạy
    if Path(INPUT_FILE).exists():
        split_dataset(INPUT_FILE, OUTPUT_DIR, valid_ratio=0.1)
    else:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_FILE}'.")
        print("Hãy kiểm tra lại xem file cuts_train.jsonl.gz đang nằm ở đâu (trong thư mục data_lhotse hay bên trong feats?)")