import re
import csv
import os

try:
    import emoji
except ImportError:  # emoji lib not installed
    emoji = None

NORMALIZATION_DICT = {
    "k": "không",
    "ko": "không",
    "kg": "không",
    "khong": "không",
    "hok": "không",
    "hông": "không",
    "dk": "được",
    "dc": "được",
    "đc": "được",
    "vs": "với",
    "vs": "với",
    "vn": "việt nam",
    "tp": "thành phố",
    "vc": "vãi",
    "vcl": "vãi",
    "vv": "vân vân",
    "j": "gì",
    "ntn": "như thế nào",
    "nb": "nhưng",
    # tiếng anh
    "u": "you",
    "r": "are",
    "y": "why",
    "idk": "i don't know",
    "lol": "lol",
    "omg": "oh my god",
    "thx": "thanks",
    "pls": "please",
    "ty": "thank you",
}

# Xóa emoji
def remove_emojis(text: str) -> str:
    if emoji is not None:
        return emoji.replace_emoji(text, replace="")
    # fall back to a very broad regex that catches most emojis (not perfect)
    return re.sub(r"[\U00010000-\U0010ffff]", "", text)

# Xóa links (http/https/www)
def remove_links(text: str) -> str:
    # drop http/https and www links
    return re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)

# Xóa các ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng
def remove_special_chars(text: str) -> str:
    return re.sub(
        r"[^0-9A-Za-zÀ-ỹà-ỹĐđ\s]",
        "",
        text,
    )

# Xóa các ký tự lặp lại (ví dụ: "rồiiii" -> "rồi", "luoonn" -> "luôn")
def collapse_elongations(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", text)

# Chuẩn hóa các từ viết tắt và biến thể thành dạng chuẩn
def normalize_tokens(text: str) -> str:
    words = text.split()
    for idx, w in enumerate(words):
        key = w.lower()
        if key in NORMALIZATION_DICT:
            words[idx] = NORMALIZATION_DICT[key]
        else:
            words[idx] = key
    return " ".join(words)

# Xác định xem một comment có phải là spam hay không 
# (rỗng, chỉ có dấu chấm, hoặc không còn chữ cái/số nào sau khi loại bỏ)
def is_spam(text: str) -> bool:
    if not text or not text.strip():
        return True
    if re.fullmatch(r"[\.\s]+", text):
        return True
    if not re.search(r"[0-9A-Za-zÀ-ỹà-ỹĐđ]", text):
        return True
    return False

# Tiền xử lý comment
def preprocess_comment(original: str) -> str | None:
    """Clean a single comment string.

    Returns the cleaned string, or ``None`` if the comment should be
    discarded (spam/empty after normalization).
    """
    if original is None:
        return None

    text = original.strip()
    text = remove_links(text)
    text = remove_emojis(text)
    text = collapse_elongations(text)
    text = remove_special_chars(text)
    text = normalize_tokens(text)

    if is_spam(text):
        return None
    return text.strip()

DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__), "raw_data", "raw_comment.csv"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "clean_data", "clean_comment.csv"
)

# Thực thi tiền xử lý 
def process_file(input_path: str | None = None, output_path: str | None = None) -> tuple[int, int]:
    """Clean comments from ``input_path`` writing results to ``output_path``.

    Returns ``(kept, dropped)`` counts. Uses default paths if arguments
    are ``None``.
    """
    if input_path is None:
        input_path = DEFAULT_INPUT
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, newline="", encoding="utf-8") as fin, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["video_id", "comment"])
        writer.writeheader()

        kept = 0
        dropped = 0
        for row in reader:
            cleaned = preprocess_comment(row.get("comment", ""))
            if cleaned:
                writer.writerow({"video_id": row.get("video_id", ""), "comment": cleaned})
                kept += 1
            else:
                dropped += 1

    return kept, dropped


if __name__ == "__main__":
    kept, dropped = process_file()
    print(f"Đã tiền xử lý xong: Số comment sạch là {kept}, đã loại bỏ {dropped}")
    print(f"Đã lưu comment sạch vào : {DEFAULT_OUTPUT}")
