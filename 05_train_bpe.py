import sentencepiece as spm
from pathlib import Path

#readme:
# Create file transcript.txt
# jq -r '.[].text' gigaspeech2_format.json > transcript.txt

def train_bpe(input_text_file, model_prefix, vocab_size):
    # Đảm bảo bạn có file text chứa toàn bộ nội dung transcript
    input_text = input_text_file
    model_prefix = model_prefix
    vocab_size = vocab_size  # Với dữ liệu nhỏ (<100h), 500 là đủ. Dữ liệu lớn thì dùng 5000-8000.

    # Train model
    spm.SentencePieceTrainer.train(
        input=input_text,
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        input_sentence_size=1000000,
        user_defined_symbols=["<blk>"] # Symbol bắt buộc của RNN-T       
    )

    print(f"Đã tạo xong {model_prefix}.model và {model_prefix}.vocab")

if __name__ == "__main__":
    # Đường dẫn file bạn vừa tạo ở bước trước
    ROOT_URL = Path("/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3")
    
    INPUT_FILE_TEXT = ROOT_URL / "transcript.txt"
    ROOT_PATH_MODEL_PREFIX = Path(ROOT_URL / 'lang_bpe_500')
    ROOT_PATH_MODEL_PREFIX.mkdir(parents=True, exist_ok=True)
    MODEL_PREFIX = ROOT_PATH_MODEL_PREFIX / 'bpe_500'
    VOCAB_SIZE = 500  # Với dữ liệu nhỏ (<100h), 500 là đủ. Dữ liệu lớn thì dùng 5000-8000.
    
    try:
        train_bpe(INPUT_FILE_TEXT, MODEL_PREFIX, VOCAB_SIZE)
    except Exception as e:
        print(f"Lỗi: {e}")