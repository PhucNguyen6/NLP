# -*- coding: utf-8 -*-
"""
test_models.py
─────────────────────────────────────────────────────────────────────────────
Crawl dữ liệu test từ YouTube và test 3 mô hình SVM đã train

Pipeline:
  1. Đọc YouTube video IDs từ youtube_id_music_test.txt
  2. Crawl comments từ tất cả các video
  3. Preprocess dữ liệu (xóa emoji, link, ký tự đặc biệt, etc.)
  4. Encode dữ liệu bằng 3 phương pháp (TF-IDF, Doc2Vec, XLM-RoBERTa)
  5. Load 3 mô hình SVM đã train + 3 scaler
  6. Dự đoán cảm xúc trên dữ liệu test
  7. Lưu kết quả vào results/test_predictions.csv
  8. Vẽ plot thống kê so sánh 3 mô hình

Chạy:
    python test_models.py

Ghi chú:
    - Video IDs được đọc từ youtube_id_music_test.txt
    - Nếu không có YOUTUBE_API_KEY, sẽ sử dụng dữ liệu mẫu
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
import csv
from dotenv import load_dotenv

from preprocessing_comment import (
    remove_emojis, remove_links, remove_special_chars, collapse_elongations,
    NORMALIZATION_DICT, EN_STOPWORDS, VI_STOPWORDS
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_CSV = os.path.join(SCRIPT_DIR, "test_data", "test_comments.csv")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
MODELS_DIR   = os.path.join(RESULTS_DIR, "svm_models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
ENCODED_DIR  = os.path.join(SCRIPT_DIR, "encoded_data")
os.makedirs(os.path.dirname(TEST_DATA_CSV), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Cấu hình ─────────────────────────────────────────────────────────────────
CLASS_LABELS = ["criticism", "neutral", "praise"]
COLORS = {
    "TF-IDF":      "#2196F3",     # xanh dương
    "Doc2Vec":     "#4CAF50",     # xanh lá
    "XLM-RoBERTa": "#FF9800",     # cam
}

# XLM-RoBERTa config
MODEL_NAME  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE  = 32
MAX_LENGTH  = 128
LABEL_REMAP = {
    "negative": "criticism",
    "neutral":  "neutral",
    "positive": "praise",
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. Crawl data từ YouTube
# ═════════════════════════════════════════════════════════════════════════════

def read_video_ids_from_file(file_path):
    """Đọc danh sách video IDs từ file."""
    if not os.path.exists(file_path):
        logging.warning(f"Không tìm thấy file: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            video_ids = [line.strip() for line in f if line.strip()]
        logging.info(f"✓ Đọc {len(video_ids)} video IDs từ {file_path}")
        return video_ids
    except Exception as e:
        logging.error(f"Lỗi đọc file video IDs: {e}")
        return []

def crawl_youtube_comments(video_id, max_results=1000, max_retries=5, retry_backoff=2):
    """Crawl comments từ một video YouTube."""
    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    
    if not API_KEY:
        logging.error("Không tìm thấy YOUTUBE_API_KEY trong .env")
        raise ValueError("YOUTUBE_API_KEY not found")
    
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
    except Exception as e:
        logging.error(f"Lỗi build YouTube API client: {e}")
        raise
    
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < max_results:
        attempt = 0
        while attempt < max_retries:
            try:
                response = request.execute()
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    logging.error(f"Quá nhiều lỗi cho video {video_id}: {e}")
                    raise
                wait_time = retry_backoff * attempt
                logging.warning(f"Lỗi API (attempt {attempt}/{max_retries}): {e} – retry sau {wait_time}s")
                time.sleep(wait_time)

        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"].get("textOriginal", "")
            comments.append({
                "video_id": video_id,
                "comment": text
            })

        if len(comments) >= max_results:
            break

        request = youtube.commentThreads().list_next(request, response)

    return comments[:max_results]

# ═════════════════════════════════════════════════════════════════════════════
# 2. Preprocess dữ liệu
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_text(text):
    """Preprocess comment text."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = remove_emojis(text)
    text = remove_links(text)
    text = remove_special_chars(text)
    text = collapse_elongations(text)
    
    # Normalize abbreviations
    words = text.split()
    words = [NORMALIZATION_DICT.get(w, w) for w in words]
    text = " ".join(words)
    
    return text.strip()

# ═════════════════════════════════════════════════════════════════════════════
# 3. Load trained models, scalers, và vectorizers
# ═════════════════════════════════════════════════════════════════════════════

def load_tfidf_model():
    """Load TF-IDF vectorizer từ training data."""
    tfidf_pkl = os.path.join(ENCODED_DIR, "tfidf_labeled.pkl")
    if not os.path.exists(tfidf_pkl):
        logging.error(f"Không tìm thấy {tfidf_pkl}")
        return None
    
    try:
        payload = joblib.load(tfidf_pkl)
        # Tạo lại vectorizer từ vocabulary
        tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        # Fit trên training data để lấy vocabulary
        train_texts = payload.get("texts", [])
        if not train_texts:
            logging.error("Không tìm thấy training texts trong tfidf_labeled.pkl")
            return None
        
        logging.info(f"Re-fitting TF-IDF trên {len(train_texts)} training texts...")
        tfidf.fit(train_texts)
        logging.info(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
        return tfidf
    except Exception as e:
        logging.error(f"Lỗi load TF-IDF: {e}")
        return None

def load_doc2vec_model():
    """Load Doc2Vec model từ training data."""
    d2v_pkl = os.path.join(ENCODED_DIR, "doc2vec_labeled.pkl")
    if not os.path.exists(d2v_pkl):
        logging.error(f"Không tìm thấy {d2v_pkl}")
        return None
    
    try:
        payload = joblib.load(d2v_pkl)
        train_texts = payload.get("texts", [])
        
        if not train_texts:
            logging.error("Không tìm thấy training texts trong doc2vec_labeled.pkl")
            return None
        
        # Re-train Doc2Vec model từ training texts
        D2V_VECTOR_SIZE = 100
        D2V_WINDOW = 5
        D2V_MIN_COUNT = 1
        D2V_WORKERS = 4
        D2V_EPOCHS = 40
        
        logging.info(f"Re-training Doc2Vec trên {len(train_texts)} training texts...")
        tagged_data = [
            TaggedDocument(words=text.split(), tags=[str(i)])
            for i, text in enumerate(train_texts)
        ]
        
        d2v_model = Doc2Vec(
            vector_size=D2V_VECTOR_SIZE,
            window=D2V_WINDOW,
            min_count=D2V_MIN_COUNT,
            workers=D2V_WORKERS,
            epochs=D2V_EPOCHS,
            dm=1,
            seed=42,
        )
        d2v_model.build_vocab(tagged_data)
        d2v_model.train(
            tagged_data,
            total_examples=d2v_model.corpus_count,
            epochs=d2v_model.epochs,
        )
        logging.info(f"Doc2Vec training completed, vocab size: {len(d2v_model.wv.key_to_index)}")
        return d2v_model
    except Exception as e:
        logging.error(f"Lỗi load Doc2Vec: {e}")
        return None

def load_xlmroberta_model():
    """Load XLM-RoBERTa model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# ═════════════════════════════════════════════════════════════════════════════
# 4. Encode test data bằng 3 phương pháp
# ═════════════════════════════════════════════════════════════════════════════

def encode_tfidf(texts, tfidf_vectorizer):
    """Encode texts using TF-IDF."""
    try:
        X = tfidf_vectorizer.transform(texts).toarray()
        return X.astype(np.float32)
    except Exception as e:
        logging.error(f"Lỗi encode TF-IDF: {e}")
        return None

def encode_doc2vec(texts, d2v_model):
    """Encode texts using Doc2Vec."""
    try:
        vectors = []
        for text in texts:
            vector = d2v_model.infer_vector(text.split())
            vectors.append(vector)
        return np.array(vectors, dtype=np.float32)
    except Exception as e:
        logging.error(f"Lỗi encode Doc2Vec: {e}")
        return None

def encode_xlmroberta(texts, tokenizer, model, device):
    """Encode texts using XLM-RoBERTa."""
    try:
        vectors = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i+BATCH_SIZE]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                # Lấy [CLS] token (hidden state cuối cùng)
                cls_vectors = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                vectors.extend(cls_vectors)
        return np.array(vectors, dtype=np.float32)
    except Exception as e:
        logging.error(f"Lỗi encode XLM-RoBERTa: {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# 5. Load trained SVM models + scalers
# ═════════════════════════════════════════════════════════════════════════════

def load_svm_model_and_scaler(method_name):
    """Load SVM model và scaler cho một phương pháp."""
    model_path = os.path.join(MODELS_DIR, f"svm_model_{method_name.lower().replace('-', '_')}.pkl")
    
    if not os.path.exists(model_path):
        logging.error(f"Không tìm thấy model cho {method_name} tại {model_path}")
        return None, None
    
    try:
        model_pkg = joblib.load(model_path)
        svm_model = model_pkg.get("model")
        scaler = model_pkg.get("scaler")
        
        if svm_model is None or scaler is None:
            logging.error(f"Model package cho {method_name} không có 'model' hoặc 'scaler'")
            return None, None
        
        logging.info(f"✓ Loaded SVM model + scaler cho {method_name}")
        return svm_model, scaler
    except Exception as e:
        logging.error(f"Lỗi load model {method_name}: {e}")
        return None, None

# ═════════════════════════════════════════════════════════════════════════════
# 6. Main test function
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Main function để test models."""
    
    logging.info("=" * 80)
    logging.info("START TESTING 3 SVM MODELS")
    logging.info("=" * 80)
    
    # ── Bước 1: Crawl data từ YouTube ──────────────────────────────────────
    logging.info("\n[1/6] Crawling YouTube comments...")
    
    # Đọc danh sách video IDs từ file
    video_id_file = os.path.join(SCRIPT_DIR, "youtube_id_music_test.txt")
    video_ids_from_file = read_video_ids_from_file(video_id_file)
    
    all_comments = []
    use_sample_data = True
    
    # Chỉ lấy video ID từ file
    if video_ids_from_file:
        logging.info(f"Found {len(video_ids_from_file)} video IDs in {video_id_file}")
        
        # In danh sách video IDs
        logging.info("Available video IDs:")
        for idx, vid in enumerate(video_ids_from_file, 1):
            logging.info(f"  [{idx}] {vid}")
        
        # Crawl từ tất cả video IDs
        logging.info(f"Crawling comments from {len(video_ids_from_file)} video(s)...")
        for idx, vid in enumerate(video_ids_from_file, 1):
            try:
                logging.info(f"[{idx}/{len(video_ids_from_file)}] Crawling: {vid}")
                comments = crawl_youtube_comments(vid, max_results=500)
                all_comments.extend(comments)
                logging.info(f"  ✓ Crawled {len(comments)} comments from {vid}")
                time.sleep(1)  # Tránh bị rate limit
            except Exception as e:
                logging.warning(f"Failed to crawl {vid}: {e}")
        
        if all_comments:
            use_sample_data = False
    else:
        logging.info("No video IDs found in file, using sample data")
    
    # Lưu dữ liệu crawl nếu thành công
    if all_comments:
        os.makedirs(os.path.dirname(TEST_DATA_CSV), exist_ok=True)
        with open(TEST_DATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "comment"])
            writer.writeheader()
            writer.writerows(all_comments)
        logging.info(f"✓ Crawled total {len(all_comments)} comments → {TEST_DATA_CSV}")
    
    # Sử dụng sample data nếu không crawl được
    if use_sample_data:
        logging.info("Using sample data for test...")
        sample_comments = [
            {"video_id": "sample", "comment": "Rất hay, tuyệt vời, thích lắm!"},
            {"video_id": "sample", "comment": "Không thích, chán, buồn quá"},
            {"video_id": "sample", "comment": "Bình thường, ok, trung bình"},
            {"video_id": "sample", "comment": "Tuyệt diệu, xuất sắc, quá đỉnh"},
            {"video_id": "sample", "comment": "Tệ, tồi, ghét lắm"},
            {"video_id": "sample", "comment": "Ổn thôi, không comment"},
        ]
        with open(TEST_DATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "comment"])
            writer.writeheader()
            writer.writerows(sample_comments)
        logging.info(f"✓ Created sample data: {len(sample_comments)} comments")
    
    # ── Bước 2: Load dữ liệu test ──────────────────────────────────────────
    logging.info("\n[2/6] Loading test data...")
    df = pd.read_csv(TEST_DATA_CSV)
    test_comments = df["comment"].astype(str).tolist()
    logging.info(f"✓ Loaded {len(test_comments)} test comments")
    
    # ── Bước 3: Preprocess dữ liệu ─────────────────────────────────────────
    logging.info("\n[3/6] Preprocessing test data...")
    processed_comments = [preprocess_text(c) for c in tqdm(test_comments, desc="Preprocess")]
    logging.info(f"✓ Preprocessed {len(processed_comments)} comments")
    
    # ── Bước 4: Load models, scalers, và vectorizers ────────────────────────
    logging.info("\n[4/6] Loading models, scalers, and vectorizers...")
    
    # Load TF-IDF
    tfidf_vec = load_tfidf_model()
    svm_tfidf, scaler_tfidf = load_svm_model_and_scaler("TF-IDF")
    
    # Load Doc2Vec
    d2v_model = load_doc2vec_model()
    svm_d2v, scaler_d2v = load_svm_model_and_scaler("Doc2Vec")
    
    # Load XLM-RoBERTa
    xlm_tokenizer, xlm_model, xlm_device = load_xlmroberta_model()
    svm_xlm, scaler_xlm = load_svm_model_and_scaler("XLM-RoBERTa")
    
    # Check if all models loaded successfully
    if not all([tfidf_vec, svm_tfidf, d2v_model, svm_d2v, xlm_model, svm_xlm]):
        logging.error("Lỗi: không tìm thấy một hoặc nhiều model")
        return
    
    logging.info("✓ All models loaded successfully")
    
    # ── Bước 5: Encode dữ liệu test ────────────────────────────────────────
    logging.info("\n[5/6] Encoding test data...")
    
    # TF-IDF encoding
    logging.info("  Encoding with TF-IDF...")
    X_tfidf = encode_tfidf(processed_comments, tfidf_vec)
    if X_tfidf is None:
        logging.error("Lỗi TF-IDF encoding")
        return
    X_tfidf_scaled = scaler_tfidf.transform(X_tfidf)
    
    # Doc2Vec encoding
    logging.info("  Encoding with Doc2Vec...")
    X_d2v = encode_doc2vec(processed_comments, d2v_model)
    if X_d2v is None:
        logging.error("Lỗi Doc2Vec encoding")
        return
    X_d2v_scaled = scaler_d2v.transform(X_d2v)
    
    # XLM-RoBERTa encoding
    logging.info("  Encoding with XLM-RoBERTa...")
    X_xlm = encode_xlmroberta(processed_comments, xlm_tokenizer, xlm_model, xlm_device)
    if X_xlm is None:
        logging.error("Lỗi XLM-RoBERTa encoding")
        return
    X_xlm_scaled = scaler_xlm.transform(X_xlm)
    
    logging.info(f"✓ Encoded data shapes: TF-IDF {X_tfidf_scaled.shape}, Doc2Vec {X_d2v_scaled.shape}, XLM-RoBERTa {X_xlm_scaled.shape}")
    
    # ── Bước 6: Dự đoán cảm xúc ───────────────────────────────────────────
    logging.info("\n[6/6] Predicting sentiments...")
    
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    
    # Predictions
    y_pred_tfidf = svm_tfidf.predict(X_tfidf_scaled)
    y_pred_d2v = svm_d2v.predict(X_d2v_scaled)
    y_pred_xlm = svm_xlm.predict(X_xlm_scaled)
    
    # Convert to labels
    y_pred_tfidf_labels = le.inverse_transform(y_pred_tfidf)
    y_pred_d2v_labels = le.inverse_transform(y_pred_d2v)
    y_pred_xlm_labels = le.inverse_transform(y_pred_xlm)
    
    logging.info("✓ Predictions completed")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Lưu kết quả vào CSV
    # ─────────────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "original_comment": test_comments,
        "processed_comment": processed_comments,
        "tfidf_prediction": y_pred_tfidf_labels,
        "doc2vec_prediction": y_pred_d2v_labels,
        "xlmroberta_prediction": y_pred_xlm_labels,
    })
    
    results_csv = os.path.join(RESULTS_DIR, "test_predictions.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8")
    logging.info(f"\n✓ Results saved to {results_csv}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Thống kê kết quả
    # ─────────────────────────────────────────────────────────────────────────
    logging.info("\n" + "=" * 80)
    logging.info("THỐNG KÊ KẾT QUẢ")
    logging.info("=" * 80)
    
    for method, predictions in [
        ("TF-IDF", y_pred_tfidf_labels),
        ("Doc2Vec", y_pred_d2v_labels),
        ("XLM-RoBERTa", y_pred_xlm_labels)
    ]:
        logging.info(f"\n{method}:")
        for label in CLASS_LABELS:
            count = np.sum(predictions == label)
            pct = 100.0 * count / len(predictions)
            logging.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Vẽ plots
    # ─────────────────────────────────────────────────────────────────────────
    logging.info("\n" + "=" * 80)
    logging.info("VẼPLOT THỐNG KÊ")
    logging.info("=" * 80)
    
    # Plot 1: Số lượng mỗi cảm xúc cho 3 phương pháp
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sentiment Distribution - Test Data (3 Methods)", fontsize=14, fontweight="bold")
    
    predictions_all = [y_pred_tfidf_labels, y_pred_d2v_labels, y_pred_xlm_labels]
    method_names = ["TF-IDF", "Doc2Vec", "XLM-RoBERTa"]
    
    for idx, (ax, preds, method) in enumerate(zip(axes, predictions_all, method_names)):
        counts = [np.sum(preds == label) for label in CLASS_LABELS]
        colors_list = ["#f44336", "#4CAF50", "#FFC107"]  # red, green, orange
        ax.bar(CLASS_LABELS, counts, color=colors_list, edgecolor="black", linewidth=1.5)
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.1)
        for i, count in enumerate(counts):
            ax.text(i, count + 1, str(count), ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "test_sentiment_distribution.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    logging.info(f"✓ Saved: {plot1_path}")
    plt.close()
    
    # Plot 2: So sánh giữa các phương pháp (grouped bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(CLASS_LABELS))
    width = 0.25
    
    counts_tfidf = [np.sum(y_pred_tfidf_labels == label) for label in CLASS_LABELS]
    counts_d2v = [np.sum(y_pred_d2v_labels == label) for label in CLASS_LABELS]
    counts_xlm = [np.sum(y_pred_xlm_labels == label) for label in CLASS_LABELS]
    
    ax.bar(x_pos - width, counts_tfidf, width, label="TF-IDF", color=COLORS["TF-IDF"], edgecolor="black")
    ax.bar(x_pos, counts_d2v, width, label="Doc2Vec", color=COLORS["Doc2Vec"], edgecolor="black")
    ax.bar(x_pos + width, counts_xlm, width, label="XLM-RoBERTa", color=COLORS["XLM-RoBERTa"], edgecolor="black")
    
    ax.set_xlabel("Sentiment Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Prediction Count", fontsize=11, fontweight="bold")
    ax.set_title("Sentiment Predictions Comparison (3 Methods)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CLASS_LABELS)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "test_predictions_comparison.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    logging.info(f"✓ Saved: {plot2_path}")
    plt.close()
    
    # Plot 3: Pie charts cho mỗi phương pháp
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sentiment Distribution Pie Charts", fontsize=14, fontweight="bold")
    
    for idx, (ax, preds, method) in enumerate(zip(axes, predictions_all, method_names)):
        counts = [np.sum(preds == label) for label in CLASS_LABELS]
        colors_list = ["#f44336", "#4CAF50", "#FFC107"]
        wedges, texts, autotexts = ax.pie(
            counts, labels=CLASS_LABELS, autopct="%1.1f%%",
            colors=colors_list, startangle=90, textprops={"fontsize": 10}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax.set_title(method, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "test_sentiment_pies.png")
    plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
    logging.info(f"✓ Saved: {plot3_path}")
    plt.close()
    
    logging.info("\n" + "=" * 80)
    logging.info("TEST HOÀN TẤT!")
    logging.info("=" * 80)
    logging.info(f"\nResults:")
    logging.info(f"  - CSV: {results_csv}")
    logging.info(f"  - Plots: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
