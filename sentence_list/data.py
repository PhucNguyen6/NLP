import re
import csv
import os
import time
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
import emoji
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0
    
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

EN_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    
    "what","which","who","whom","this","that","these","those",
    
    "am","is","are","was","were","be","been","being",
    "have","has","had","having",
    "do","does","did","doing",
    
    "a","an","the",
    
    "and","but","if","or","because","as","until","while",
    
    "of","at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under",
    
    "again","further","then","once",
    
    "here","there","when","where","why","how",
    
    "all","any","both","each","few","more","most","other","some","such",
    
    "no","nor","not","only","own","same","so","than","too","very",
    
    "can","will","just","don","should","now"
}

VI_STOPWORDS = {
    "và","là","có","của","cho","một","những","các","được","trong","khi","đó",
    "này","kia","ấy","đó","đây",
    
    "tôi","ta","chúng_tôi","chúng_ta","bạn","các_bạn","họ","nó",
    
    "làm","đang","đã","sẽ","cũng","vẫn","còn","bị","được",
    
    "với","về","từ","đến","trên","dưới","giữa","qua","lại",
    
    "rằng","thì","mà","nhưng","hay","hoặc","vì","nên",
    
    "rất","hơi","khá","quá","lắm","nhiều","ít",
    
    "mỗi","từng","cả","tất_cả",
    
    "cái","con","chiếc","việc","điều",
    
    "ở","ra","vào","lên","xuống",
    
    "đâu","nào","sao","vậy","à","ừ","ờ",
    
    "cùng","theo","do","bởi","nếu","thì","lúc","khi",
    
    "đi","đến","về","lại","ra","vào",
    
    "ai","gì","nào","bao_nhiêu","bao_lâu",
    
    "thế_nào","như_thế_nào",
    
    "đây_là","kia_là"
}

NORMALIZATION_DICT = {
    "k": "không",
    "r": "rồi",
    "rùi": "rồi",
    "luon": "luôn",
    'lun': 'luôn',
    "cx": "cũng",
    "ko": "không",
    "kg": "không",
    "thik": "thích",
    "thikh": "thích",
    "thich": "thích",
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
    "cụa": "của",
    "ntn": "như thế nào",
    "nb": "nhưng",
    "nm": "nhưng mà",
    "mxh": "mạng xã hội",
    # tiếng anh
    "u": "you",
    "r": "are",
    "y": "why",
    "idk": "i don't know",
    "lol": "lol",
    "omg": "oh my god",
    "thx": "thanks",
    "pls": "please",
    "pl" : "please",
    "ty": "thank you",
    "btw": "by the way",
    "anw": "anyway",
    
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

# Phát hiện ngôn ngữ của comment, chỉ giữ lại tiếng Việt (vi) hoặc tiếng Anh (en), và cả hai (mix)
def detect_language(text: str) -> str | None:
    """Detect language; only keep Vietnamese (`vi`) or English (`en`)."""
    if not text:
        return None

    # Prefer langdetect when available
    if detect is not None:
        try:
            lang = detect(text)
            if lang in ("vi", "en"):
                return lang
            return None
        except Exception:
            pass
    
    vi_chars = bool(re.search(r"[ăắằẵẳâấầẫẩáàãảạêếềễểệéèẽẻẹôốồỗổộơớờỡởợưứừữửựđ]", text, flags=re.IGNORECASE))
    en_chars = bool(re.search(r"[a-zA-Z]", text))

    if vi_chars and en_chars:
        return "mix"
    if vi_chars and not en_chars:
        return "vi"
    if en_chars and not vi_chars:
        return "en"
    return None

# Xóa stopwords 
def remove_stopwords(text: str, lang: str = "mixed") -> str:
    """Remove stopwords based on language type."""
    words = text.split()
    
    if lang == "en":
        filtered_words = [w for w in words if w not in EN_STOPWORDS]
    elif lang == "vi":
        filtered_words = [w for w in words if w not in VI_STOPWORDS]
    else:  # mixed or unknown
        filtered_words = [w for w in words if w not in EN_STOPWORDS and w not in VI_STOPWORDS]
    
    return " ".join(filtered_words)

# Stemming & Lemmatization cho tiếng Anh và Việt
def stem_and_lemmatize(text: str, lang: str = "mixed") -> str:
    words = text.split()
    processed_words = []
    
    if lang == "en":
        for word in words:
            lemmatized = lemmatizer.lemmatize(word, pos='v')  # lemmatize as verb first
            lemmatized = lemmatizer.lemmatize(lemmatized, pos='n')  # then as noun
            stemmed = stemmer.stem(lemmatized)
            processed_words.append(stemmed)
    elif lang == "vi":
        processed_words = words
    else:
        for word in words:
            lemmatized = lemmatizer.lemmatize(word, pos='v')
            lemmatized = lemmatizer.lemmatize(lemmatized, pos='n')
            stemmed = stemmer.stem(lemmatized)
            processed_words.append(stemmed)
    
    return " ".join(processed_words)

# Tiền xử lý comment
def preprocess_comment(original: str) -> tuple[str | None, str | None]:
    """Clean a single comment string.

    Returns a tuple (cleaned_text, lang) or (None, None) if the comment should be
    discarded (spam/empty/non-supported language).
    """
    if original is None:
        return None, None

    text = original.strip()
    text = remove_links(text)
    text = remove_emojis(text)
    text = collapse_elongations(text)
    text = remove_special_chars(text)
    text = normalize_tokens(text)

    # Detect language early to apply language-specific processing
    lang = detect_language(text)
    if lang is None:
        return None, None

    # Remove stopwords based on detected language
    text = remove_stopwords(text, lang)
    
    # Apply stemming and lemmatization based on language
    text = stem_and_lemmatize(text, lang)

    if is_spam(text):
        return None, None

    return text.strip(), lang

from config import BASE_DIR, DATA_DIR, CRAWLING_CONFIG

RAW_DATA_DIR = BASE_DIR / "raw_data"
DEFAULT_INPUT = str(RAW_DATA_DIR / "raw_comment.csv")
DEFAULT_OUTPUT = str(DATA_DIR / "clean_comment.csv")

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
        writer = csv.DictWriter(fout, fieldnames=["video_id", "comment", "lang"])
        writer.writeheader()

        kept = 0
        dropped = 0
        for row in reader:
            cleaned, lang = preprocess_comment(row.get("comment", ""))
            if cleaned is not None and lang is not None:
                writer.writerow({"video_id": row.get("video_id", ""), "comment": cleaned, "lang": lang})
                kept += 1
            else:
                dropped += 1

    return kept, dropped


# --- YouTube crawl (gộp từ crawl_sentence.py) ---
def crawl_youtube_comments(video_id, max_results=1000, max_retries=5, retry_backoff=2):
    from googleapiclient.discovery import build

    api_key = CRAWLING_CONFIG.get("youtube_api_key") or __import__("os").getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu YOUTUBE_API_KEY trong .env")
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
    while request and len(comments) < max_results:
        attempt = 0
        while attempt < max_retries:
            try:
                response = request.execute()
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise
                time.sleep(retry_backoff * attempt)
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"].get("textOriginal", "")
            comments.append({"video_id": video_id, "comment": text})
        if len(comments) >= max_results:
            break
        request = youtube.commentThreads().list_next(request, response)
    return comments[:max_results]


def read_video_ids(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video ID file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_crawl(
    video_list: str | None = None,
    output_path: str | None = None,
    max_per_video: int = 20000,
) -> str:
    import time
    from pathlib import Path

    video_list = video_list or str(BASE_DIR / "youtube_id_music.txt")
    output_path = output_path or DEFAULT_INPUT
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_comments = []
    for vid in read_video_ids(video_list):
        print(f"Đang crawl video: {vid}")
        all_comments.extend(crawl_youtube_comments(vid, max_results=max_per_video))
        time.sleep(1)
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["video_id", "comment"])
        writer.writeheader()
        writer.writerows(all_comments)
    print(f"Đã lưu {len(all_comments)} comments → {output_path}")
    return output_path


def run_preprocess(input_path: str | None = None, output_path: str | None = None) -> tuple[int, int]:
    kept, dropped = process_file(input_path, output_path)
    print(f"Tiền xử lý: giữ {kept}, loại {dropped} → {output_path or DEFAULT_OUTPUT}")
    return kept, dropped
