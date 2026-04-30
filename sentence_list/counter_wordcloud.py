import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ================= CONFIG =================
RAW_PATH = "raw_data/raw_comment.csv"
CLEAN_PATH = "clean_data/clean_comment.csv"
OUTPUT_DIR = "plots"

TEXT_COLUMN = "comment"   # đổi nếu tên cột khác

# ================= CREATE OUTPUT DIR =================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
raw_df = pd.read_csv(RAW_PATH)
clean_df = pd.read_csv(CLEAN_PATH)

# ================= COUNT COMMENTS =================
raw_count = raw_df.shape[0]
clean_count = clean_df.shape[0]

print(f"Số comment trước khi làm sạch: {raw_count}")
print(f"Số comment sau khi làm sạch: {clean_count}")

# ================= BAR CHART =================
labels = ['Raw', 'Clean']
values = [raw_count, clean_count]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values)

# Hiển thị số trên cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             str(height), ha='center', va='bottom')

plt.title("So sánh số lượng comment trước và sau khi làm sạch")
plt.ylabel("Số lượng comment")

plt.savefig(os.path.join(OUTPUT_DIR, "comment_comparison.png"))
plt.close()

# ================= WORDCLOUD =================
# Gộp toàn bộ text
text_data = clean_df[TEXT_COLUMN].dropna().astype(str)
text = " ".join(text_data)

# Tạo wordcloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=200
).generate(text)

# Hiển thị & lưu
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.title("WordCloud sau khi làm sạch dữ liệu")

plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"))
plt.close()

print("Đã lưu biểu đồ vào thư mục plots/")