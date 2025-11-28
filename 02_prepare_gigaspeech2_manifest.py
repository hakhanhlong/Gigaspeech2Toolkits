import json
import logging
from pathlib import Path
from lhotse import RecordingSet, SupervisionSet, Recording, SupervisionSegment, AudioSource
import os

def prepare_gigaspeech2(root_dir, output_dir, lang='vi'):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    
    # Đọc file metadata (Giả sử format json của GigaSpeech 2)
    # Bạn cần kiểm tra tên file metadata chính xác trong bộ dataset bạn tải
    meta_path = root_dir / "gigaspeech2_format.json" 
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    recordings = []
    supervisions = []
    
    print(f"Đang xử lý dữ liệu cho ngôn ngữ: {lang}...")

    for item in data:
        # Lọc theo ngôn ngữ (nếu dataset gộp nhiều ngôn ngữ)
        if item.get('lang') != lang: 
            continue

        # 1. Tạo Recording Object
        # item['audio_path'] phải là đường dẫn tương đối hoặc tuyệt đối
        audio_path = root_dir / item['audio_path'] 

        if not os.path.exists(audio_path):
            continue
                    
        recording = Recording.from_file(audio_path, recording_id=item['segment_id'])
        recordings.append(recording)
        
        # 2. Tạo Supervision Object (Label)
        text = item['text'] # Transcript
        
        # Xử lý text cơ bản (viết thường)
        #text = text.lower().strip()
        
        segment = SupervisionSegment(
            id=item['segment_id'],
            recording_id=item['segment_id'],
            start=0.0,
            duration=item['duration'],
            channel=0,
            language=lang,
            #speaker=item.get('speaker', 'unknown'),
            text=text
        )
        supervisions.append(segment)

    # Lưu Manifests
    recording_set = RecordingSet.from_recordings(recordings)
    print(f"-> Tìm thấy {len(recordings)} file âm thanh (recording_set).")
    supervision_set = SupervisionSet.from_segments(supervisions)
    print(f"-> Đã tạo {len(supervisions)} nhãn dữ liệu (supervision_set).")
    
    # # Chia train/dev/test (Ví dụ đơn giản: lấy 1000 mẫu làm dev/test)
    # # Thực tế GigaSpeech 2 thường chia sẵn, bạn nên check metadata field 'split'
    # # Ở đây tôi chia ngẫu nhiên để demo code chạy được:
    # splits = recording_set.split(num_splits=20, shuffle=True)
    
    # # Lưu tập Test/Dev (5%)
    # test_rec = splits[0]
    # test_sup = supervision_set.filter(lambda s: s.recording_id in test_rec)
    # test_rec.to_file(output_dir / "recordings_test.jsonl.gz")
    # test_sup.to_file(output_dir / "supervisions_test.jsonl.gz")
    
    # # Lưu tập Train (95% còn lại)
    # train_rec = RecordingSet.from_recordings([r for s in splits[1:] for r in s])

    # train_sup = supervision_set.filter(lambda s: s.recording_id in train_rec)


    recording_set.to_file(output_dir / "recordings_train.jsonl.gz")
    supervision_set.to_file(output_dir / "supervisions_train.jsonl.gz")

    print("Hoàn tất tạo manifest!")

if __name__ == "__main__":
    # Thay đổi đường dẫn này theo máy của bạn
    ROOT_URL = Path('/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3')
    prepare_gigaspeech2(
        root_dir = ROOT_URL, 
        output_dir = ROOT_URL / 'manifests',
        lang='vi'
    )