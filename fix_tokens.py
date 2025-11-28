# fix_tokens.py
path = "/mnt/c/AILAB/DATASET/GigaSpeech2/data/vi3/lang_bpe_500/tokens.txt"

# Đọc file lên
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

seen_symbols = set()
new_lines = []

print("Đang quét lỗi...")
for line in lines:
    parts = line.strip().split()
    if len(parts) < 2: continue
    
    symbol = parts[0]
    
    # Nếu chưa gặp symbol này bao giờ -> Thêm vào danh sách
    if symbol not in seen_symbols:
        seen_symbols.add(symbol)
        new_lines.append(line)
    else:
        # Nếu gặp rồi -> Bỏ qua (Đây chính là dòng gây lỗi)
        print(f"❌ Đã xóa dòng trùng lặp: {line.strip()}")

# Ghi đè lại file
with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("✅ Đã sửa xong file tokens.txt! Bạn có thể chạy lại lệnh export.")