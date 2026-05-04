# -*- coding: utf-8 -*-
"""
encode_label_tfidf.py
─────────────────────────────────────────────────────────────────────────────
Bước 1 – Mã hoá TF-IDF + Gán nhãn tự động
  • Vectorise toàn bộ comment bằng TF-IDF (500 features, 1-2 gram)
  • Phân cụm KMeans (k=3) trên không gian TF-IDF
  • Ánh xạ mỗi cụm → nhãn cảm xúc (praise / neutral / criticism)
    bằng cách chấm điểm lexicon trên mẫu đại diện của từng cụm
  • Lưu toàn bộ đặc trưng + nhãn + metadata → encoded_data/tfidf_labeled.pkl

─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ── import lexicon scorer để ánh xạ cụm → nhãn ──────────────────────────────
from sentiment_define import analyze_sentiment

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TF-IDF] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(SCRIPT_DIR, "clean_data",   "clean_comment.csv")
OUTPUT_PKL  = os.path.join(SCRIPT_DIR, "encoded_data", "tfidf_labeled.pkl")
os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tải dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Đang tải dữ liệu từ %s …", INPUT_CSV)
df    = pd.read_csv(INPUT_CSV)
texts = df["comment"].astype(str).tolist()
langs = df["lang"].astype(str).tolist()
logging.info("Tải xong %d comment.", len(texts))

# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF Vectorisation
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Khởi tạo TF-IDF (max_features=500, ngram_range=(1,2)) …")
t_enc_start = time.perf_counter()

tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    sublinear_tf=True,          # log(1+tf) – giảm ảnh hưởng tần suất cao
    min_df=2,                   # bỏ token quá hiếm
)
X = tfidf.fit_transform(texts).toarray()          # (n, 500)

encode_time = time.perf_counter() - t_enc_start
vocab_size  = len(tfidf.vocabulary_)
logging.info("TF-IDF shape: %s | vocab: %d | thời gian: %.2fs",
             X.shape, vocab_size, encode_time)

# ─────────────────────────────────────────────────────────────────────────────
# 3. KMeans Clustering (k = 3)
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Chạy KMeans k=3 …")
t_clus_start = time.perf_counter()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=500)
cluster_ids = kmeans.fit_predict(X)

cluster_time = time.perf_counter() - t_clus_start
logging.info("Phân cụm xong trong %.2fs.", cluster_time)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Ánh xạ cụm → nhãn bằng lexicon scoring
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Tính điểm lexicon trên từng cụm để ánh xạ nhãn …")

SAMPLE_PER_CLUSTER = 300
cluster_scores: dict[int, float] = {}

for cid in range(3):
    idxs   = np.where(cluster_ids == cid)[0]
    sample = idxs[:min(SAMPLE_PER_CLUSTER, len(idxs))]
    scores = []
    for i in sample:
        try:
            result = analyze_sentiment(texts[i], language=langs[i])
            scores.append(result.get("score", 0.0))
        except Exception:
            scores.append(0.0)
    cluster_scores[cid] = float(np.mean(scores)) if scores else 0.0
    logging.info("  Cụm %d: mean_score=%.4f  size=%d",
                 cid, cluster_scores[cid], len(idxs))

# Sắp xếp theo điểm: thấp → criticism, giữa → neutral, cao → praise
sorted_cids = sorted(cluster_scores, key=lambda k: cluster_scores[k])
label_map: dict[int, str] = {
    sorted_cids[0]: "criticism",
    sorted_cids[1]: "neutral",
    sorted_cids[2]: "praise",
}
logging.info("Ánh xạ cụm → nhãn: %s", label_map)

y = np.array([label_map[cid] for cid in cluster_ids])

# ─────────────────────────────────────────────────────────────────────────────
# 5. Thống kê phân phối nhãn
# ─────────────────────────────────────────────────────────────────────────────
label_dist = {lbl: int((y == lbl).sum()) for lbl in ["praise", "neutral", "criticism"]}
logging.info("Phân phối nhãn: %s", label_dist)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Đếm tham số
#    – TF-IDF không có tham số "học được" theo nghĩa deep learning,
#      nhưng vocabulary size là kích thước không gian đặc trưng (mô hình thống kê)
# ─────────────────────────────────────────────────────────────────────────────
encoder_params = vocab_size   # số token trong từ điển

# ─────────────────────────────────────────────────────────────────────────────
# 7. Lưu payload
# ─────────────────────────────────────────────────────────────────────────────
payload = {
    "X":     X,                 # numpy (n, 500)
    "y":     y,                 # numpy str labels
    "texts": texts,
    "langs": langs,
    "metadata": {
        "method":          "TF-IDF",
        "n_samples":       len(texts),
        "n_features":      X.shape[1],
        "vocab_size":      vocab_size,
        "encoder_params":  encoder_params,
        "encode_time_s":   encode_time,
        "cluster_time_s":  cluster_time,
        "total_time_s":    encode_time + cluster_time,
        "label_dist":      label_dist,
        "cluster_scores":  cluster_scores,
        "label_map":       label_map,
        "tfidf_config": {
            "max_features": 500,
            "ngram_range":  (1, 2),
            "sublinear_tf": True,
            "min_df":       2,
        },
    },
}
joblib.dump(payload, OUTPUT_PKL, compress=3)
logging.info("Đã lưu → %s", OUTPUT_PKL)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Tóm tắt
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TF-IDF ENCODING & LABELING – HOÀN THÀNH")
print("=" * 60)
print(f"  Số mẫu         : {len(texts):,}")
print(f"  Số đặc trưng   : {X.shape[1]}")
print(f"  Kích thước vocab: {vocab_size:,}")
print(f"  Tham số encoder : {encoder_params:,}")
print(f"  Thời gian encode: {encode_time:.2f}s")
print(f"  Thời gian cluster: {cluster_time:.2f}s")
print(f"  Phân phối nhãn  : {label_dist}")
print(f"  Lưu tại         : {OUTPUT_PKL}")
print("=" * 60 + "\n")
