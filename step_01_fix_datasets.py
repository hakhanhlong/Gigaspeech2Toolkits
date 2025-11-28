import logging
from pathlib import Path
from lhotse import CutSet

logging.basicConfig(level=logging.INFO)

def fix_supervision_overflow(cut_set_path, output_path):
    logging.info(f"Đang xử lý: {cut_set_path}")
    
    # 1. Load dữ liệu và ép vào RAM
    try:
        cuts = CutSet.from_file(cut_set_path).to_eager()
    except Exception as e:
        logging.error(f"Không tìm thấy file {cut_set_path}. Bỏ qua.")
        return

    fixed_count = 0
    removed_count = 0
    
    # 2. Hàm sửa lỗi cho từng Cut
    def sanitize_cut(cut):
        nonlocal fixed_count, removed_count
        new_supervisions = []
        is_modified = False
        
        for sup in cut.supervisions:
            # Trường hợp 1: Nhãn bắt đầu sau khi audio đã hết -> Xóa luôn
            if sup.start >= cut.duration:
                removed_count += 1
                is_modified = True
                continue
            
            # Trường hợp 2 (Lỗi của bạn): Nhãn kết thúc sau khi audio hết -> Xén bớt
            if sup.end > cut.duration:
                # Giới hạn độ dài lại cho vừa khít với audio
                old_dur = sup.duration
                sup.duration = cut.duration - sup.start
                
                # Cảnh báo nhẹ nếu bị cắt quá nhiều (> 0.1s)
                if old_dur - sup.duration > 0.1:
                    logging.warning(f"Cắt bớt {old_dur - sup.duration:.3f}s của nhãn trong cut {cut.id}")
                
                fixed_count += 1
                is_modified = True
            
            new_supervisions.append(sup)
        
        if is_modified:
            cut.supervisions = new_supervisions
        return cut

    # 3. Áp dụng sửa lỗi
    cuts = cuts.map(sanitize_cut)
    
    # 4. Loại bỏ các Cut không còn nhãn nào (nếu có)
    cuts = cuts.filter(lambda c: len(c.supervisions) > 0)
    
    # 5. Lưu đè lại file cũ (hoặc file mới)
    cuts.to_file(output_path)
    logging.info(f"✅ Hoàn tất! Đã sửa {fixed_count} lỗi, xóa {removed_count} nhãn thừa.")
    logging.info(f"Đã lưu tại: {output_path}")

if __name__ == "__main__":
    ROOT_URL = Path('/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3')
    # --- DANH SÁCH FILE CẦN SỬA ---
    # Hãy điền đường dẫn đến file data của bạn (trong thư mục data/fbank/)
    datasets_to_fix = [
        ROOT_URL / "fbank/cuts_train_final.jsonl.gz",
        ROOT_URL / "fbank/cuts_valid.jsonl.gz",
        ROOT_URL / "fbank/cuts_valid_dev_other.jsonl.gz" # File bạn fake lúc nãy
    ]
    
    for path in datasets_to_fix:
        fix_supervision_overflow(path, path) # Lưu đè lên chính nó