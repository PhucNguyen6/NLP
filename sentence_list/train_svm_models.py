# -*- coding: utf-8 -*-
"""
train_svm_models.py
─────────────────────────────────────────────────────────────────────────────
Huấn luyện SVM trên 3 bộ đặc trưng (TF-IDF / Doc2Vec / XLM-RoBERTa)
và lưu model + xuất biểu đồ so sánh.

Pipeline cho mỗi phương pháp:
  1. Tải encoded_data/<method>_labeled.pkl
  2. Phân tách Train/Val/Test = 7 / 2 / 1 (stratified)
  3. Chuẩn hoá đặc trưng (StandardScaler)
  4. Huấn luyện SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
  5. LƯU MODEL + pickling
  6. Tính metrics trên tập Val + Test
  7. Đếm tham số SVM

Kết quả xuất:
  results/svm_models/svm_model_*.pkl          – model SVM đã train
  results/svm_comparison_metrics.csv          – bảng tổng hợp metrics
  results/svm_comparison_summary.txt          – báo cáo văn bản chi tiết
  plots/01_metrics_comparison.png             – biểu đồ cột nhóm các metric
  plots/02_time_params.png                    – thời gian train & số tham số
  plots/03_confusion_matrices.png             – ma trận nhầm lẫn (3 phương pháp)
  plots/04_learning_curves.png                – đường cong học (3 phương pháp)
  plots/05_metrics_heatmap.png                – heatmap tương quan metrics

Chạy độc lập (sau khi đã chạy 3 file encode_label_*.py):
    python train_svm_models.py
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SVM] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)

# Helper print function with flush
def pprint(msg: str):
    """Print with auto-flush."""
    print(msg, flush=True)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ENCODED_DIR  = os.path.join(SCRIPT_DIR, "encoded_data")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
MODELS_DIR   = os.path.join(RESULTS_DIR, "svm_models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# ── Datasets ─────────────────────────────────────────────────────────────────
DATASETS = {
    "TF-IDF":       os.path.join(ENCODED_DIR, "tfidf_labeled.pkl"),
    "Doc2Vec":      os.path.join(ENCODED_DIR, "doc2vec_labeled.pkl"),
    "XLM-RoBERTa":  os.path.join(ENCODED_DIR, "xlmroberta_labeled.pkl"),
}

# ── Màu sắc nhất quán cho 3 phương pháp ──────────────────────────────────────
COLORS = {
    "TF-IDF":      "#2196F3",   # xanh dương
    "Doc2Vec":     "#4CAF50",   # xanh lá
    "XLM-RoBERTa": "#FF9800",   # cam
}
CLASS_LABELS = ["criticism", "neutral", "praise"]

# ═════════════════════════════════════════════════════════════════════════════
# Hàm tiện ích
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính Accuracy / Precision / Recall / F1 (macro)."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def count_svm_params(svm: SVC, n_features: int) -> int:
    """
    Đếm tham số SVM ở dạng dual:
        support_vectors_  : (n_sv, n_features)
        dual_coef_        : (n_classes-1, n_sv)
        intercept_        : (n_classes*(n_classes-1)/2,)
    """
    n_sv      = int(sum(svm.n_support_))
    n_classes = len(svm.classes_)
    sv_params   = n_sv * n_features
    dual_params = (n_classes - 1) * n_sv
    bias_params = int(n_classes * (n_classes - 1) / 2)
    return sv_params + dual_params + bias_params


def print_dataset_statistics(y_split, split_name: str, class_labels: list, le=None):
    """
    Thống kê chi tiết phân bố các lớp trong tập dữ liệu.
    
    Args:
        y_split: mảng nhãn
        split_name: tên tập (Train/Val/Test)
        class_labels: danh sách tên các lớp
        le: LabelEncoder (nếu muốn hiển thị tên lớp thay vì số)
    """
    unique, counts = np.unique(y_split, return_counts=True)
    total = len(y_split)
    
    logging.info(f"┌─ {split_name.upper()} SET STATISTICS ─────────────────────────")
    logging.info(f"│ Tổng mẫu: {total}")
    
    for label_idx, count in zip(unique, counts):
        pct = 100 * count / total
        class_name = class_labels[label_idx] if label_idx < len(class_labels) else f"Class {label_idx}"
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        logging.info(f"│ {class_name:<15} : {count:>6} ({pct:>5.1f}%) {bar:<30}")
    
    logging.info(f"└─ {'─'*60}")
    return dict(zip(unique, counts))


def split_data(X: np.ndarray, y: np.ndarray):
    """Chia 7 / 2 / 1 (train / val / test), stratified."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=2/9, random_state=42, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═════════════════════════════════════════════════════════════════════════════
# Vòng lặp chính: huấn luyện và đánh giá từng phương pháp
# ═════════════════════════════════════════════════════════════════════════════
all_results: dict[str, dict] = {}

for idx, (method_name, pkl_path) in enumerate(DATASETS.items(), 1):
    print(f"\n[{idx}/{len(DATASETS)}] Đang xử lý phương pháp: {method_name}")

    # ── Kiểm tra file ────────────────────────────────────────────────────────
    if not os.path.exists(pkl_path):
        logging.warning("Không tìm thấy file: %s – Bỏ qua.", pkl_path)
        continue

    # ── 1. Tải dữ liệu đã mã hoá ────────────────────────────────────────────
    logging.info("Tải dữ liệu từ %s …", pkl_path)
    print(f"  → Tải dữ liệu...")
    payload  = joblib.load(pkl_path)
    X_raw    = payload["X"].astype(np.float32)
    y_raw    = payload["y"]
    meta     = payload["metadata"]

    logging.info("Shape đặc trưng: %s | Số mẫu: %d", X_raw.shape, len(y_raw))
    print(f"  ✓ Dữ liệu: {X_raw.shape} ({len(y_raw)} mẫu)")

    # ── 2. Mã hoá nhãn ──────────────────────────────────────────────────────
    print(f"  → Mã hoá nhãn...")
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    y_enc = le.transform(y_raw)
    print(f"  ✓ Mã hoá: {len(CLASS_LABELS)} lớp")

    # ── 3. Phân tách Train / Val / Test ─────────────────────────────────────
    print(f"  → Phân tách dữ liệu (7/2/1)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_raw, y_enc)
    logging.info(
        "Split → Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    print(f"  ✓ Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # ── In thống kê chi tiết về Validation & Test set ──────────────────────────
    print(f"\n  ╔═ VALIDATION SET STATISTICS ═╗")
    val_dist = print_dataset_statistics(y_val, "Validation", CLASS_LABELS, le)
    
    print(f"\n  ╔═ TEST SET STATISTICS (10% INTERNAL) ═╗")
    test_dist = print_dataset_statistics(y_test, "Test", CLASS_LABELS, le)
    print()

    # ── 4. Chuẩn hoá đặc trưng ──────────────────────────────────────────────
    print(f"  → Chuẩn hoá (StandardScaler)...")
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_val_sc    = scaler.transform(X_val)
    X_test_sc   = scaler.transform(X_test)
    print(f"  ✓ Chuẩn hoá xong")

    # ── 5. Huấn luyện SVM ───────────────────────────────────────────────────
    print(f"  → Huấn luyện SVM (RBF kernel, CPU) ...")
    logging.info("Huấn luyện SVM (RBF kernel, CPU) …")
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42,
    )

    t_train_start = time.perf_counter()
    svm.fit(X_train_sc, y_train)
    svm_train_time = time.perf_counter() - t_train_start
    logging.info("Huấn luyện SVM xong trong %.2fs.", svm_train_time)
    print(f"  ✓ SVM train time: {svm_train_time:.2f}s")

    # ── 6. Dự đoán ──────────────────────────────────────────────────────────
    print(f"  → Dự đoán trên Val & Test...")
    y_pred_val  = svm.predict(X_val_sc)
    y_pred_test = svm.predict(X_test_sc)
    print(f"  ✓ Dự đoán xong")

    # ── 7. Metrics ──────────────────────────────────────────────────────────
    print(f"  → Tính metrics...")
    val_metrics  = compute_metrics(y_val, y_pred_val)
    test_metrics = compute_metrics(y_test, y_pred_test)
    cm_test      = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])

    logging.info(
        "Val  → Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f",
        *val_metrics.values(),
    )
    logging.info(
        "Test → Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f",
        *test_metrics.values(),
    )

    # ── 8. Đếm tham số SVM ──────────────────────────────────────────────────
    n_sv      = int(sum(svm.n_support_))
    svm_params = count_svm_params(svm, X_train_sc.shape[1])
    enc_params = meta.get("encoder_params", 0)
    total_params = enc_params + svm_params

    logging.info(
        "Support vectors: %d | SVM params: %d | Total params: %d",
        n_sv, svm_params, total_params,
    )
    print(f"  ✓ Support vectors: {n_sv} | SVM params: {svm_params:,}")

    # ── 9. LƯU MODEL SVM ─────────────────────────────────────────────────────
    print(f"  → Lưu model SVM...")
    model_filename = f"svm_model_{method_name.lower().replace('-', '_')}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    model_pkg = {
        "model": svm,
        "scaler": scaler,
        "label_encoder": le,
        "method": method_name,
        "metrics_test": test_metrics,
        "metrics_val": val_metrics,
        "n_features": X_train_sc.shape[1],
        "class_labels": CLASS_LABELS,
    }
    joblib.dump(model_pkg, model_path)
    logging.info("Model lưu → %s", model_path)
    print(f"  ✓ Model: {model_path}")

    # ── 10. Learning Curve ──────────────────────────────────────────────────
    print(f"  → Learning curve (5-fold CV)...")
    logging.info("Tính learning curve (CPU) …")
    train_sizes_frac = np.linspace(0.1, 1.0, 8)
    lc_sizes, lc_train_scores, lc_val_scores = learning_curve(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        X_train_sc, y_train,
        train_sizes=train_sizes_frac,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    print(f"  ✓ Learning curve xong")

    # ── Lưu kết quả ──────────────────────────────────────────────────────────
    all_results[method_name] = {
        "val_metrics":       val_metrics,
        "test_metrics":      test_metrics,
        "cm_test":           cm_test,
        "svm_train_time_s":  svm_train_time,
        "encoder_time_s":    meta.get("encode_time_s", meta.get("total_time_s", 0)),
        "encoder_params":    enc_params,
        "svm_params":        svm_params,
        "total_params":      total_params,
        "n_sv":              n_sv,
        "n_features":        X_train_sc.shape[1],
        "n_train":           len(X_train),
        "n_val":             len(X_val),
        "n_test":            len(X_test),
        "label_dist":        meta.get("label_dist", {}),
        "val_dist":          val_dist,
        "test_dist":         test_dist,
        "lc_sizes":          lc_sizes,
        "lc_train_scores":   lc_train_scores,
        "lc_val_scores":     lc_val_scores,
        "le":                le,
        "model_path":        model_path,
        "classification_report": classification_report(
            y_test, y_pred_test,
            target_names=CLASS_LABELS,
            zero_division=0,
        ),
    }

if not all_results:
    raise RuntimeError("Không có phương pháp nào được tải thành công. "
                       "Hãy chạy các file encode_label_*.py trước.")

methods = list(all_results.keys())

# ═════════════════════════════════════════════════════════════════════════════
# Xuất CSV tổng hợp metrics
# ═════════════════════════════════════════════════════════════════════════════
rows = []
for m, res in all_results.items():
    for split_name, mdict in [("val", res["val_metrics"]), ("test", res["test_metrics"])]:
        rows.append({
            "method":         m,
            "split":          split_name,
            "accuracy":       mdict["accuracy"],
            "precision":      mdict["precision"],
            "recall":         mdict["recall"],
            "f1":             mdict["f1"],
            "svm_train_time": res["svm_train_time_s"],
            "encoder_time":   res["encoder_time_s"],
            "encoder_params": res["encoder_params"],
            "svm_params":     res["svm_params"],
            "total_params":   res["total_params"],
            "n_sv":           res["n_sv"],
            "n_features":     res["n_features"],
        })

metrics_df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "svm_comparison_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
logging.info("Metrics CSV → %s", csv_path)

# ═════════════════════════════════════════════════════════════════════════════
# Xuất báo cáo văn bản
# ═════════════════════════════════════════════════════════════════════════════
txt_path = os.path.join(RESULTS_DIR, "svm_comparison_summary.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SVM COMPARISON REPORT – TF-IDF vs Doc2Vec vs XLM-RoBERTa\n")
    f.write("=" * 80 + "\n\n")
    f.write("NOTA: Test set là 10% từ phân chia dữ liệu stratified (KHÔNG dùng youtube_id_music_test.txt)\n")
    f.write("=" * 80 + "\n\n")

    for m, res in all_results.items():
        f.write(f"{'─'*60}\n")
        f.write(f"  Phương pháp : {m}\n")
        f.write(f"{'─'*60}\n")
        f.write(f"  Đặc trưng         : {res['n_features']} chiều\n")
        f.write(f"  Tham số encoder   : {res['encoder_params']:,}\n")
        f.write(f"  Tham số SVM       : {res['svm_params']:,}\n")
        f.write(f"  Support vectors   : {res['n_sv']:,}\n")
        f.write(f"  Tổng tham số      : {res['total_params']:,}\n")
        f.write(f"  Thời gian encode  : {res['encoder_time_s']:.2f}s\n")
        f.write(f"  Thời gian SVM     : {res['svm_train_time_s']:.2f}s\n")
        f.write(f"  Model file        : {res['model_path']}\n")
        
        f.write(f"\n  --- Phân bố Validation Set ---\n")
        f.write(f"  Tổng mẫu: {res['n_val']}\n")
        if "val_dist" in res and res["val_dist"]:
            total_val = res['n_val']
            for class_idx in range(len(CLASS_LABELS)):
                count = res["val_dist"].get(class_idx, 0)
                pct = 100 * count / total_val if total_val > 0 else 0
                f.write(f"    {CLASS_LABELS[class_idx]:<15}: {count:>6} ({pct:>5.1f}%)\n")
        
        f.write(f"\n  --- Phân bố Test Set (10% Internal) ---\n")
        f.write(f"  Tổng mẫu: {res['n_test']}\n")
        if "test_dist" in res and res["test_dist"]:
            total_test = res['n_test']
            for class_idx in range(len(CLASS_LABELS)):
                count = res["test_dist"].get(class_idx, 0)
                pct = 100 * count / total_test if total_test > 0 else 0
                f.write(f"    {CLASS_LABELS[class_idx]:<15}: {count:>6} ({pct:>5.1f}%)\n")
        
        f.write(f"\n  --- Validation ---\n")
        for k, v in res["val_metrics"].items():
            f.write(f"    {k:<12}: {v:.4f}\n")
        f.write(f"\n  --- Test (10% Internal) ---\n")
        for k, v in res["test_metrics"].items():
            f.write(f"    {k:<12}: {v:.4f}\n")
        f.write(f"\n  Classification Report (Test):\n")
        f.write(res["classification_report"])
        f.write("\n\n")

logging.info("Báo cáo → %s", txt_path)

print(f"\n{'='*70}")
print(f"BIỂU ĐỒ HÓATIZATION")
print(f"{'='*70}")

# ═════════════════════════════════════════════════════════════════════════════
# ── BIỂU ĐỒ 1: So sánh Metrics (grouped bar chart) ──────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
print(f"[1/5] Biểu đồ so sánh Metrics...")
metric_keys  = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("So sánh Metrics SVM – TF-IDF / Doc2Vec / XLM-RoBERTa",
             fontsize=14, fontweight="bold", y=1.02)

for ax_idx, (split_key, split_title) in enumerate([("val", "Validation"), ("test", "Test")]):
    ax = axes[ax_idx]
    n_methods  = len(methods)
    n_metrics  = len(metric_keys)
    bar_width  = 0.22
    x          = np.arange(n_metrics)

    for i, m in enumerate(methods):
        vals = [all_results[m][f"{split_key}_metrics"][k] for k in metric_keys]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width,
                      label=m, color=COLORS[m], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Tập {split_title}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "01_metrics_comparison.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
logging.info("Biểu đồ 1 → %s", p1)
print(f"  ✓ Lưu: 01_metrics_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# ── BIỂU ĐỒ 2: Thời gian train & Số tham số ─────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
print(f"[2/5] Biểu đồ thời gian & tham số...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Thời gian & Số tham số – TF-IDF / Doc2Vec / XLM-RoBERTa (SVM)",
             fontsize=14, fontweight="bold")

# ── 2a: Thời gian encoding (stacked bar) ─────────────────────────────────────
ax = axes[0]
enc_times = [all_results[m]["encoder_time_s"] for m in methods]
svm_times = [all_results[m]["svm_train_time_s"] for m in methods]
x_pos = np.arange(len(methods))

bars1 = ax.bar(x_pos, enc_times, color=[COLORS[m] for m in methods],
               alpha=0.85, label="Encode time", edgecolor="white")
bars2 = ax.bar(x_pos, svm_times, bottom=enc_times,
               color=[COLORS[m] for m in methods], alpha=0.5,
               hatch="//", label="SVM train time", edgecolor="white")

for i, (e, s) in enumerate(zip(enc_times, svm_times)):
    ax.text(i, e + s + max(enc_times) * 0.02, f"{e+s:.1f}s",
            ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Giây (s)", fontsize=10)
ax.set_title("Thời gian (Encode + SVM Train)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 2b: Số tham số encoder (log scale) ──────────────────────────────────────
ax = axes[1]
enc_params_vals = [all_results[m]["encoder_params"] for m in methods]
bars = ax.bar(x_pos, enc_params_vals,
              color=[COLORS[m] for m in methods], edgecolor="white", alpha=0.85)
for bar, v in zip(bars, enc_params_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.05,
            f"{v/1e6:.2f}M" if v >= 1e6 else f"{v:,}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_yscale("log")
ax.set_ylabel("Số tham số (log scale)", fontsize=10)
ax.set_title("Tham số Encoder", fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 2c: Số tham số SVM (Support Vectors) ─────────────────────────────────────
ax = axes[2]
svm_params_vals = [all_results[m]["svm_params"] for m in methods]
n_sv_vals       = [all_results[m]["n_sv"] for m in methods]
bars = ax.bar(x_pos, svm_params_vals,
              color=[COLORS[m] for m in methods], edgecolor="white", alpha=0.85)
for bar, v, sv in zip(bars, svm_params_vals, n_sv_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.05,
            f"{v:,}\n({sv} SVs)",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Số tham số SVM", fontsize=10)
ax.set_title("Tham số SVM (dual form)", fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, "02_time_params.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
logging.info("Biểu đồ 2 → %s", p2)
print(f"  ✓ Lưu: 02_time_params.png")

# ═════════════════════════════════════════════════════════════════════════════
# ── BIỂU ĐỒ 3: Confusion Matrices (1×N subplots) ────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
print(f"[3/5] Ma trận nhầm lẫn...")
n_methods = len(methods)
fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
if n_methods == 1:
    axes = [axes]
fig.suptitle("Ma trận Nhầm lẫn – Tập Test (SVM)", fontsize=13, fontweight="bold")

for ax, m in zip(axes, methods):
    cm = all_results[m]["cm_test"]
    # Normalise theo hàng (true label)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", ax=ax,
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    # Thêm số tuyệt đối bên cạnh tỉ lệ
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="grey")

    acc  = all_results[m]["test_metrics"]["accuracy"]
    f1   = all_results[m]["test_metrics"]["f1"]
    ax.set_title(f"{m}\nAcc={acc:.3f}  F1={f1:.3f}", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)

plt.tight_layout()
p3 = os.path.join(PLOTS_DIR, "03_confusion_matrices.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
logging.info("Biểu đồ 3 → %s", p3)
print(f"  ✓ Lưu: 03_confusion_matrices.png")

# ═════════════════════════════════════════════════════════════════════════════
# ── BIỂU ĐỒ 4: Learning Curves (F1 macro) ───────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
print(f"[4/5] Learning curves...")
fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), sharey=True)
if n_methods == 1:
    axes = [axes]
fig.suptitle("Learning Curves (F1-macro) – SVM", fontsize=13, fontweight="bold")

for ax, m in zip(axes, methods):
    sizes       = all_results[m]["lc_sizes"]
    tr_mean     = all_results[m]["lc_train_scores"].mean(axis=1)
    tr_std      = all_results[m]["lc_train_scores"].std(axis=1)
    cv_mean     = all_results[m]["lc_val_scores"].mean(axis=1)
    cv_std      = all_results[m]["lc_val_scores"].std(axis=1)
    col         = COLORS[m]

    ax.plot(sizes, tr_mean, "o-",  color=col,         label="Train score", linewidth=1.8)
    ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color=col)

    ax.plot(sizes, cv_mean, "s--", color=col, alpha=0.7, label="CV score",    linewidth=1.8)
    ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.1, color=col)

    ax.set_xlabel("Số mẫu train", fontsize=9)
    ax.set_ylabel("F1-macro", fontsize=9)
    ax.set_title(m, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
p4 = os.path.join(PLOTS_DIR, "04_learning_curves.png")
plt.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
logging.info("Biểu đồ 4 → %s", p4)
print(f"  ✓ Lưu: 04_learning_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# ── BIỂU ĐỒ 5: Heatmap tương quan Metrics (Test) ────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
print(f"[5/5] Heatmap metrics...")
heat_data = pd.DataFrame(
    {m: {k: all_results[m]["test_metrics"][k] for k in metric_keys} for m in methods}
).T                                                # (methods, metrics)
heat_data.columns = metric_labels

fig, ax = plt.subplots(figsize=(8, max(3, len(methods) * 1.2)))
sns.heatmap(
    heat_data, annot=True, fmt=".4f", ax=ax,
    cmap="RdYlGn", vmin=0, vmax=1,
    linewidths=0.8, linecolor="white",
    annot_kws={"size": 11, "weight": "bold"},
    cbar_kws={"shrink": 0.7, "label": "Score"},
)
ax.set_title("Heatmap Metrics – Tập Test (SVM)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10, rotation=0)
plt.tight_layout()

p5 = os.path.join(PLOTS_DIR, "05_metrics_heatmap.png")
plt.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()
logging.info("Biểu đồ 5 → %s", p5)
print(f"  ✓ Lưu: 05_metrics_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# ── In bảng tổng kết cuối ───────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*82}")
print(f"  KẾT QUẢ TỔNG HỢP – SVM (Tập Test)")
print(f"{'='*82}")
header = f"{'Method':<16} {'Feat':>6} {'Enc-Params':>14} {'SVM-Params':>12} "
header += f"{'EncTime':>8} {'SVMTime':>8} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}"
print(header)
print("-" * 82)
for m, res in all_results.items():
    tm  = res["test_metrics"]
    row = (
        f"{m:<16} "
        f"{res['n_features']:>6,} "
        f"{res['encoder_params']:>14,} "
        f"{res['svm_params']:>12,} "
        f"{res['encoder_time_s']:>7.2f}s "
        f"{res['svm_train_time_s']:>7.2f}s "
        f"{tm['accuracy']:>7.4f} "
        f"{tm['precision']:>7.4f} "
        f"{tm['recall']:>7.4f} "
        f"{tm['f1']:>7.4f}"
    )
    print(row)
print("=" * 82)
print(f"\nFile đã lưu:")
print(f"  Models  : {MODELS_DIR}")
print(f"  CSV     : {csv_path}")
print(f"  Report  : {txt_path}")
for p in [p1, p2, p3, p4, p5]:
    print(f"  Plot    : {p}")
print()
