import logging
import shutil
from pathlib import Path
from lhotse import CutSet, Fbank, FbankConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')

def recompute_features_fresh(input_json, output_dir):
    input_path = Path(input_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- ĐANG LÀM MỚI FEATURE CHO: {input_path.name} ---")

    # 1. Load dữ liệu JSON hiện tại (đã qua các bước sửa lỗi trước đó)
    try:
        cuts = CutSet.from_file(input_path).to_eager()
    except Exception as e:
        print(f"❌ Lỗi load file: {e}")
        return

    print(f"-> Số lượng câu: {len(cuts)}")

    # 2. Cấu hình Feature Extractor (Phải chuẩn 80 mel bins cho Zipformer)
    extractor = Fbank(FbankConfig(num_mel_bins=80))

    # 3. Tính toán lại Feature
    # Lưu vào folder mới để tránh xung đột
    new_feats_dir = output_dir / "feats_recomputed"
    new_feats_dir.mkdir(exist_ok=True)

    print("-> Đang tính toán lại Fbank (có thể mất vài phút)...")
    try:
        # Hàm này sẽ tự động update metadata trong 'cuts' để khớp với feature mới
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=new_feats_dir,
            num_jobs=1, # Tăng lên nếu máy mạnh
            #storage_type="lilcom_files"
        )
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất đặc trưng: {e}")
        print("Gợi ý: Kiểm tra xem file .wav gốc có còn tồn tại không?")
        return

    # 4. Lưu đè lại file JSON
    # Bây giờ JSON và Feature trên đĩa đã đồng bộ hoàn toàn
    cuts.to_file(input_path)
    print(f"✅ Đã đồng bộ xong! Đã lưu đè file JSON tại: {input_path}")

if __name__ == "__main__":
    # Thư mục chứa dữ liệu của bạn
    ROOT_URL = Path('/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3')
    DATA_DIR = ROOT_URL / "fbank"
    
    # Danh sách file cần làm mới (Train và Valid)
    files_to_process = [
        ROOT_URL / "fbank/cuts_train_final.jsonl.gz",
        ROOT_URL / "fbank/cuts_valid.jsonl.gz",
        ROOT_URL / "fbank/cuts_valid_dev_other.jsonl.gz" # File bạn fake lúc nãy
    ]

    for f in files_to_process:
        if f.exists():
            recompute_features_fresh(f, DATA_DIR)
        else:
            print(f"⚠️ Không tìm thấy file: {f}")