# -*- coding: utf-8 -*-
"""
encode_label_xlmroberta.py
─────────────────────────────────────────────────────────────────────────────
Bước 3 – Mã hoá XLM-RoBERTa + Gán nhãn tự động
  • Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
    (hỗ trợ đa ngôn ngữ bao gồm tiếng Anh và tiếng Việt)
  • Gán nhãn: lấy trực tiếp từ đầu ra phân loại của mô hình
      negative → criticism | neutral → neutral | positive → praise
  • Đặc trưng: vector [CLS] từ lớp ẩn cuối cùng (768 chiều)
  • Xử lý theo batch để tiết kiệm bộ nhớ; tự động dùng GPU nếu có
  • Lưu toàn bộ đặc trưng + nhãn + metadata → encoded_data/xlmroberta_labeled.pkl

Chạy độc lập:
    python encode_label_xlmroberta.py

Yêu cầu:
    pip install transformers torch
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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [XLM-RoBERTa] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(SCRIPT_DIR, "clean_data",   "clean_comment.csv")
OUTPUT_PKL  = os.path.join(SCRIPT_DIR, "encoded_data", "xlmroberta_labeled.pkl")
os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

# ── Cấu hình ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE  = 32
MAX_LENGTH  = 128          # comment ngắn; 128 là đủ và nhanh hơn 512

# Ánh xạ nhãn model → nhãn dự án
# cardiffnlp model id2label: 0=negative, 1=neutral, 2=positive
LABEL_REMAP = {
    "negative": "criticism",
    "neutral":  "neutral",
    "positive": "praise",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tải dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
logging.info("Đang tải dữ liệu từ %s …", INPUT_CSV)
df    = pd.read_csv(INPUT_CSV)
texts = df["comment"].astype(str).tolist()
langs = df["lang"].astype(str).tolist()
logging.info("Tải xong %d comment.", len(texts))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Khởi tạo model & tokenizer
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Thiết bị: %s", device)

logging.info("Tải model %s …", MODEL_NAME)
t_load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,   # lấy CLS embedding
)
model.to(device)
model.eval()

load_time = time.perf_counter() - t_load_start
logging.info("Tải model xong trong %.2fs.", load_time)

# ── Đếm tham số ──────────────────────────────────────────────────────────────
encoder_params = sum(p.numel() for p in model.parameters())
logging.info("Tổng tham số model: %d (~%.1fM)", encoder_params, encoder_params / 1e6)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference theo batch: lấy nhãn + CLS embedding
# ─────────────────────────────────────────────────────────────────────────────
logging.info(
    "Bắt đầu inference batch_size=%d, max_length=%d …",
    BATCH_SIZE, MAX_LENGTH,
)
t_enc_start = time.perf_counter()

all_embeddings: list[np.ndarray] = []
all_labels:     list[str]         = []

n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

with torch.no_grad():
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]

        # Tokenise
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward pass
        outputs = model(**encoded)

        # ── Nhãn: argmax của logits ──────────────────────────────────────────
        pred_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
        for pid in pred_ids:
            raw_label = model.config.id2label[pid].lower()
            all_labels.append(LABEL_REMAP.get(raw_label, "neutral"))

        # ── Embedding: CLS token của lớp ẩn cuối (hidden_states[-1][:, 0, :])
        last_hidden = outputs.hidden_states[-1]      # (B, seq_len, 768)
        cls_vecs    = last_hidden[:, 0, :].cpu().numpy().astype(np.float32)
        all_embeddings.append(cls_vecs)

        # Log tiến độ mỗi 50 batch
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            logging.info(
                "  Batch %d/%d – %d/%d mẫu",
                batch_idx + 1, n_batches, end, len(texts),
            )

encode_time = time.perf_counter() - t_enc_start
logging.info("Inference xong trong %.2fs.", encode_time)

X = np.vstack(all_embeddings)            # (n, 768)
y = np.array(all_labels)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Thống kê phân phối nhãn
# ─────────────────────────────────────────────────────────────────────────────
label_dist = {lbl: int((y == lbl).sum()) for lbl in ["praise", "neutral", "criticism"]}
logging.info("Phân phối nhãn: %s", label_dist)
logging.info("Shape đặc trưng: %s", X.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Lưu payload
# ─────────────────────────────────────────────────────────────────────────────
payload = {
    "X":     X,
    "y":     y,
    "texts": texts,
    "langs": langs,
    "metadata": {
        "method":          "XLM-RoBERTa",
        "n_samples":       len(texts),
        "n_features":      X.shape[1],
        "encoder_params":  encoder_params,
        "encode_time_s":   encode_time,
        "load_time_s":     load_time,
        "total_time_s":    encode_time + load_time,
        "label_dist":      label_dist,
        "label_remap":     LABEL_REMAP,
        "device":          str(device),
        "xlmr_config": {
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "hidden_size": 768,
            "embedding":  "CLS token (last hidden state)",
        },
    },
}
joblib.dump(payload, OUTPUT_PKL, compress=3)
logging.info("Đã lưu → %s", OUTPUT_PKL)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Tóm tắt
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  XLM-RoBERTa ENCODING & LABELING – HOÀN THÀNH")
print("=" * 65)
print(f"  Số mẫu           : {len(texts):,}")
print(f"  Số đặc trưng     : {X.shape[1]}")
print(f"  Tham số encoder  : {encoder_params:,} (~{encoder_params/1e6:.1f}M)")
print(f"  Thời gian tải    : {load_time:.2f}s")
print(f"  Thời gian encode : {encode_time:.2f}s")
print(f"  Thiết bị         : {device}")
print(f"  Phân phối nhãn   : {label_dist}")
print(f"  Lưu tại          : {OUTPUT_PKL}")
print("=" * 65 + "\n")
