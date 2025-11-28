import lhotse
from lhotse import Fbank, FbankConfig
import logging
from pathlib import Path


# Cấu hình logging để theo dõi quá trình
logging.basicConfig(level=logging.INFO)


def feature_fbank(ROOT_URL):    

    storage_feats_path = Path(ROOT_URL / "feats")
    storage_feats_path.mkdir(parents=True, exist_ok=True)

    # Load manifests vừa tạo
    recordings = lhotse.load_manifest(ROOT_URL / "manifests/recordings_train.jsonl.gz")
    supervisions = lhotse.load_manifest(ROOT_URL / "manifests/supervisions_train.jsonl.gz")


    # Cấu hình Fbank giống Kaldi (80 mel bins là chuẩn cho Icefall/k2)
    extractor = Fbank(FbankConfig(num_mel_bins=80))

    # Tính toán và lưu vào thư mục 'feats'
    cuts = lhotse.CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    
    cuts = cuts.compute_and_store_features(
        extractor = extractor,
        storage_path = ROOT_URL / "feats/",
        num_jobs=1  # Tăng số này lên nếu máy bạn nhiều CPU core
    )


    storage_fbank_path = Path(ROOT_URL / "fbank")
    storage_fbank_path.mkdir(parents=True, exist_ok=True)

    # Lưu CutSet (đây là input cuối cùng cho Icefall)
    cuts.to_file(storage_fbank_path / "cuts_train.jsonl.gz")    


if __name__ == "__main__":    
    ROOT_URL = Path("/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3")
    try:
        feature_fbank(ROOT_URL)
    except Exception as e:
        print(f"Lỗi: {e}")

    