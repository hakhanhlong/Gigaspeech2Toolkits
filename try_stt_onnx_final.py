import sherpa_onnx
import soundfile as sf
import numpy as np
import librosa # Cần cài: pip install librosa

def main():
    # --- 1. CẤU HÌNH (Sửa đường dẫn của bạn vào đây) ---
    base_dir = "/mnt/c/AILAB/MODELS/exp/dev_vi_10h"
    tokens_path = "/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3/lang_bpe_500/tokens.txt"
    wav_file = "/mnt/c/AILAB/AUDIO_TEST_STT/speech.wav" # File âm thanh để test

    print(f"--- Đang khởi tạo model GPU từ: {base_dir} ---")

    # --- 2. KHỞI TẠO MODEL (GPU) ---
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{base_dir}/encoder-epoch-30-avg-5.onnx",
            decoder=f"{base_dir}/decoder-epoch-30-avg-5.onnx",
            joiner=f"{base_dir}/joiner-epoch-30-avg-5.onnx",
            tokens=tokens_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cuda", # <--- QUAN TRỌNG: Chạy trên GPU
            debug=False
        )
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return

    # --- 3. ĐỌC VÀ RESAMPLE AUDIO (Quan trọng để không bị lỗi lặp từ) ---
    print(f"--- Đang đọc và xử lý file: {wav_file} ---")
    try:
        # Dùng librosa để load và tự động ép về 16000Hz
        audio, sr = librosa.load(wav_file, sr=16000)
    except Exception as e:
        print(f"❌ Lỗi đọc file wav (Bạn đã cài librosa chưa?): {e}")
        return

    # --- 4. NHẬN DẠNG ---
    print("--- Đang nhận dạng... ---")
    s = recognizer.create_stream()
    s.accept_waveform(16000, audio) # Luôn truyền 16000 vào đây
    recognizer.decode_stream(s)
    
    # --- 5. KẾT QUẢ ---
    text = s.result.text
    print("\n" + "="*40)
    print("📝 KẾT QUẢ CUỐI CÙNG:")
    print("="*40)
    print(text)
    print("="*40 + "\n")

if __name__ == "__main__":
    main()