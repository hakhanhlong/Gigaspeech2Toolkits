import sherpa_onnx
import soundfile as sf

def main():
    # --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
    model_dir = "/mnt/c/AILAB/MODELS/exp/dev_vi_10h"
    tokens_path = "/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3/lang_bpe_500/tokens.txt"
    wav_file = "/mnt/c/AILAB/DATASET/AUDIO_TEST_STT/speech.wav" # File âm thanh để test

    print(f"--- Đang khởi tạo model từ: {model_dir} ---")

    # --- 2. KHỞI TẠO RECOGNIZER (CÁCH ĐƠN GIẢN) ---
    # Hàm này tự động gom các config lại, bạn không lo thiếu attribute nữa
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder-epoch-30-avg-5.onnx",
            decoder=f"{model_dir}/decoder-epoch-30-avg-5.onnx",
            joiner=f"{model_dir}/joiner-epoch-30-avg-5.onnx",
            tokens=tokens_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search", # hoặc "modified_beam_search"
            provider="cuda" # Đổi thành "cuda" nếu muốn chạy GPU
        )
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    # --- 3. ĐỌC FILE WAV ---
    print(f"--- Đang đọc file: {wav_file} ---")
    try:
        audio, sample_rate = sf.read(wav_file, dtype="float32")
    except Exception as e:
        print(f"❌ Không tìm thấy file wav: {e}")
        return

    # Kiểm tra sample rate
    if sample_rate != 16000:
        print(f"⚠️ Cảnh báo: File âm thanh là {sample_rate}Hz. Model cần 16000Hz.")
        # Lưu ý: Nếu sai sample rate, kết quả sẽ sai hoàn toàn. 
        # Sherpa-onnx python không tự resample, bạn cần resample file wav trước.

    # --- 4. NHẬN DẠNG ---
    s = recognizer.create_stream()
    s.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(s)
    
    # Lấy kết quả
    text = s.result.text
    
    print("\n" + "="*30)
    print("📝 KẾT QUẢ NHẬN DẠNG:")
    print("="*30)
    print(text)
    print("="*30 + "\n")

if __name__ == "__main__":
    main()