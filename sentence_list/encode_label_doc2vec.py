# -*- coding: utf-8 -*-
"""
encode_label_doc2vec.py
─────────────────────────────────────────────────────────────────────────────
Bước 2 – Mã hoá Doc2Vec + Gán nhãn tự động
  • Huấn luyện Doc2Vec (vector_size=100, window=5, epochs=40)
  • Suy diễn vector đại diện cho mỗi comment
  • Phân cụm KMeans (k=3) trên không gian Doc2Vec
  • Ánh xạ mỗi cụm → nhãn cảm xúc (praise / neutral / criticism)
    bằng cách chấm điểm lexicon trên mẫu đại diện của từng cụm
  • Lưu toàn bộ đặc trưng + nhãn + metadata → encoded_data/doc2vec_labeled.pkl

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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans

# ── import lexicon scorer để ánh xạ cụm → nhãn ──────────────────────────────
from sentiment_define import analyze_sentiment

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Doc2Vec] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(SCRIPT_DIR, "clean_data",   "clean_comment.csv")
OUTPUT_PKL  = os.path.join(SCRIPT_DIR, "encoded_data", "doc2vec_labeled.pkl")
os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình Doc2Vec
# ─────────────────────────────────────────────────────────────────────────────
D2V_VECTOR_SIZE = 100
D2V_WINDOW      = 5
D2V_MIN_COUNT   = 1
D2V_WORKERS     = 4
D2V_EPOCHS      = 40

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tải dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Đang tải dữ liệu từ %s …", INPUT_CSV)
df    = pd.read_csv(INPUT_CSV)
texts = df["comment"].astype(str).tolist()
langs = df["lang"].astype(str).tolist()
logging.info("Tải xong %d comment.", len(texts))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Huấn luyện Doc2Vec
# ─────────────────────────────────────────────────────────────────────────────
logging.info(
    "Huấn luyện Doc2Vec (vector_size=%d, window=%d, epochs=%d) …",
    D2V_VECTOR_SIZE, D2V_WINDOW, D2V_EPOCHS,
)
t_enc_start = time.perf_counter()

tagged_data = [
    TaggedDocument(words=text.split(), tags=[str(i)])
    for i, text in enumerate(texts)
]

d2v_model = Doc2Vec(
    vector_size=D2V_VECTOR_SIZE,
    window=D2V_WINDOW,
    min_count=D2V_MIN_COUNT,
    workers=D2V_WORKERS,
    epochs=D2V_EPOCHS,
    dm=1,          # Distributed Memory (PV-DM)
    seed=42,
)
d2v_model.build_vocab(tagged_data)
d2v_model.train(
    tagged_data,
    total_examples=d2v_model.corpus_count,
    epochs=d2v_model.epochs,
)

encode_time = time.perf_counter() - t_enc_start
logging.info("Huấn luyện Doc2Vec xong trong %.2fs.", encode_time)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Suy diễn vector cho toàn bộ corpus
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Suy diễn vector cho %d comment …", len(texts))
t_infer_start = time.perf_counter()

X = np.array([
    d2v_model.infer_vector(text.split(), epochs=20)
    for text in texts
], dtype=np.float32)                                  # (n, 100)

infer_time = time.perf_counter() - t_infer_start
total_encode_time = encode_time + infer_time
logging.info("Suy diễn xong trong %.2fs. Shape: %s", infer_time, X.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 4. KMeans Clustering (k = 3)
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Chạy KMeans k=3 …")
t_clus_start = time.perf_counter()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=500)
cluster_ids = kmeans.fit_predict(X)

cluster_time = time.perf_counter() - t_clus_start
logging.info("Phân cụm xong trong %.2fs.", cluster_time)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ánh xạ cụm → nhãn bằng lexicon scoring
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

sorted_cids = sorted(cluster_scores, key=lambda k: cluster_scores[k])
label_map: dict[int, str] = {
    sorted_cids[0]: "criticism",
    sorted_cids[1]: "neutral",
    sorted_cids[2]: "praise",
}
logging.info("Ánh xạ cụm → nhãn: %s", label_map)

y = np.array([label_map[cid] for cid in cluster_ids])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Thống kê phân phối nhãn
# ─────────────────────────────────────────────────────────────────────────────
label_dist = {lbl: int((y == lbl).sum()) for lbl in ["praise", "neutral", "criticism"]}
logging.info("Phân phối nhãn: %s", label_dist)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Đếm tham số Doc2Vec
#    – Word vectors: vocab_size × vector_size
#    – Context weights (DM): vocab_size × vector_size
#    – Tổng ≈ 2 × vocab_size × vector_size
# ─────────────────────────────────────────────────────────────────────────────
vocab_size     = len(d2v_model.wv.key_to_index)
encoder_params = vocab_size * D2V_VECTOR_SIZE * 2   # word vecs + context vecs
logging.info("Vocab: %d | Tham số encoder: ~%d", vocab_size, encoder_params)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Lưu payload
# ─────────────────────────────────────────────────────────────────────────────
payload = {
    "X":     X,
    "y":     y,
    "texts": texts,
    "langs": langs,
    "metadata": {
        "method":          "Doc2Vec",
        "n_samples":       len(texts),
        "n_features":      X.shape[1],
        "vocab_size":      vocab_size,
        "encoder_params":  encoder_params,
        "encode_time_s":   total_encode_time,
        "cluster_time_s":  cluster_time,
        "total_time_s":    total_encode_time + cluster_time,
        "label_dist":      label_dist,
        "cluster_scores":  cluster_scores,
        "label_map":       label_map,
        "d2v_config": {
            "vector_size": D2V_VECTOR_SIZE,
            "window":      D2V_WINDOW,
            "min_count":   D2V_MIN_COUNT,
            "epochs":      D2V_EPOCHS,
            "dm":          1,
        },
    },
}
joblib.dump(payload, OUTPUT_PKL, compress=3)
logging.info("Đã lưu → %s", OUTPUT_PKL)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Tóm tắt
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Doc2Vec ENCODING & LABELING – HOÀN THÀNH")
print("=" * 60)
print(f"  Số mẫu          : {len(texts):,}")
print(f"  Số đặc trưng    : {X.shape[1]}")
print(f"  Kích thước vocab : {vocab_size:,}")
print(f"  Tham số encoder  : ~{encoder_params:,}")
print(f"  Thời gian encode : {total_encode_time:.2f}s")
print(f"  Thời gian cluster: {cluster_time:.2f}s")
print(f"  Phân phối nhãn   : {label_dist}")
print(f"  Lưu tại          : {OUTPUT_PKL}")
print("=" * 60 + "\n")
