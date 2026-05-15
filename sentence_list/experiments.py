# -*- coding: utf-8 -*-
"""Pipeline thống nhất: mã hoá (TF-IDF / Doc2Vec / XLM-RoBERTa) + huấn luyện SVM,
đánh giá train/validation/test (7/2/1), xuất metrics, biểu đồ và JSON tham số.

Ví dụ:
  python sentence_list/experiments.py encode --method all
  python sentence_list/experiments.py train
  python sentence_list/experiments.py all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_define import analyze_sentiment

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EXP] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
CLEAN_COMMENT_CSV = BASE / "clean_data" / "clean_comment.csv"
ENCODED_DIR = BASE / "encoded_data"
RESULTS_DIR = BASE / "results"
MODELS_DIR = RESULTS_DIR / "svm_models"
PLOTS_DIR = BASE / "plots"

CLASS_LABELS = ["criticism", "neutral", "praise"]
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Phân chia stratified 7 / 2 / 1 (train / val / test)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_STATE = 42

SVM_CONFIG: dict[str, Any] = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,
    "random_state": RANDOM_STATE,
}

# (file_key, tên hiển thị, tên file .pkl đã encode) — file_key khớp inference_pipeline (svm_model_{file_key}.pkl)
ENCODED_SPECS: list[tuple[str, str, str]] = [
    ("tf_idf", "TF-IDF", "tfidf_labeled.pkl"),
    ("doc2vec", "Doc2Vec", "doc2vec_labeled.pkl"),
    ("xlm_roberta", "XLM-RoBERTa", "xlmroberta_labeled.pkl"),
]

COLORS = {
    "TF-IDF": "#2196F3",
    "Doc2Vec": "#4CAF50",
    "XLM-RoBERTa": "#FF9800",
}

ENCODED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_comments():
    df = pd.read_csv(CLEAN_COMMENT_CSV)
    texts = df["comment"].astype(str).tolist()
    langs = df["lang"].astype(str).tolist() if "lang" in df.columns else ["vi"] * len(texts)
    return texts, langs


def map_clusters_to_labels(cluster_ids, texts, langs):
    cluster_scores = {}
    for cid in sorted(set(cluster_ids)):
        idxs = np.where(cluster_ids == cid)[0][:300]
        scores = []
        for i in idxs:
            try:
                scores.append(analyze_sentiment(texts[i], language=langs[i]).get("score", 0.0))
            except Exception:
                scores.append(0.0)
        cluster_scores[cid] = float(np.mean(scores)) if scores else 0.0
    sorted_cids = sorted(cluster_scores, key=lambda c: cluster_scores[c])
    label_map = {sorted_cids[0]: "criticism", sorted_cids[1]: "neutral", sorted_cids[2]: "praise"}
    y = np.array([label_map[c] for c in cluster_ids])
    return y, cluster_scores, label_map


def encode_tfidf(texts, langs):
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    x = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    encode_time = time.perf_counter() - start
    cluster_start = time.perf_counter()
    cluster_ids = KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=500).fit_predict(x)
    y, cluster_scores, label_map = map_clusters_to_labels(cluster_ids, texts, langs)
    cluster_time = time.perf_counter() - cluster_start
    payload = {
        "X": x,
        "y": y,
        "texts": texts,
        "langs": langs,
        "metadata": {
            "method": "TF-IDF",
            "n_samples": len(texts),
            "n_features": x.shape[1],
            "vocab_size": len(vectorizer.vocabulary_),
            "encoder_params": len(vectorizer.vocabulary_),
            "encode_time_s": encode_time,
            "cluster_time_s": cluster_time,
            "total_time_s": encode_time + cluster_time,
            "cluster_scores": cluster_scores,
            "label_map": label_map,
            "label_dist": {k: int((y == k).sum()) for k in ["praise", "neutral", "criticism"]},
            "tfidf_config": {"max_features": 500, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 2},
        },
    }
    joblib.dump(payload, ENCODED_DIR / "tfidf_labeled.pkl", compress=3)
    joblib.dump(vectorizer, ENCODED_DIR / "tfidf_vectorizer.pkl")


def encode_doc2vec(texts, langs):
    config = {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 40, "dm": 1}
    start = time.perf_counter()
    tagged = [TaggedDocument(words=t.split(), tags=[str(i)]) for i, t in enumerate(texts)]
    model = Doc2Vec(
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        workers=4,
        epochs=config["epochs"],
        dm=config["dm"],
        seed=42,
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    x = np.array([model.infer_vector(t.split(), epochs=20) for t in texts], dtype=np.float32)
    encode_time = time.perf_counter() - start
    cluster_start = time.perf_counter()
    cluster_ids = KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=500).fit_predict(x)
    y, cluster_scores, label_map = map_clusters_to_labels(cluster_ids, texts, langs)
    cluster_time = time.perf_counter() - cluster_start
    vocab_size = len(model.wv.key_to_index)
    payload = {
        "X": x,
        "y": y,
        "texts": texts,
        "langs": langs,
        "metadata": {
            "method": "Doc2Vec",
            "n_samples": len(texts),
            "n_features": x.shape[1],
            "vocab_size": vocab_size,
            "encoder_params": vocab_size * config["vector_size"] * 2,
            "encode_time_s": encode_time,
            "cluster_time_s": cluster_time,
            "total_time_s": encode_time + cluster_time,
            "cluster_scores": cluster_scores,
            "label_map": label_map,
            "label_dist": {k: int((y == k).sum()) for k in ["praise", "neutral", "criticism"]},
            "d2v_config": config,
        },
    }
    joblib.dump(payload, ENCODED_DIR / "doc2vec_labeled.pkl", compress=3)
    model.save(str(ENCODED_DIR / "doc2vec_model.model"))


def encode_xlmroberta(texts, langs):
    del langs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.to(device)
    model.eval()
    label_remap = {"negative": "criticism", "neutral": "neutral", "positive": "praise"}
    bs = 32
    all_emb, all_labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            pred_ids = out.logits.argmax(dim=-1).cpu().numpy()
            for pid in pred_ids:
                all_labels.append(label_remap.get(model.config.id2label[pid].lower(), "neutral"))
            cls = out.hidden_states[-1][:, 0, :].cpu().numpy().astype(np.float32)
            all_emb.append(cls)
    x = np.vstack(all_emb)
    y = np.array(all_labels)
    encode_time = time.perf_counter() - start
    params = sum(p.numel() for p in model.parameters())
    payload = {
        "X": x,
        "y": y,
        "texts": texts,
        "langs": ["vi"] * len(texts),
        "metadata": {
            "method": "XLM-RoBERTa",
            "n_samples": len(texts),
            "n_features": x.shape[1],
            "encoder_params": int(params),
            "encode_time_s": encode_time,
            "total_time_s": encode_time,
            "label_dist": {k: int((y == k).sum()) for k in ["praise", "neutral", "criticism"]},
            "label_remap": label_remap,
            "device": str(device),
        },
    }
    joblib.dump(payload, ENCODED_DIR / "xlmroberta_labeled.pkl", compress=3)


def run_encode(method: str):
    texts, langs = load_comments()
    if method in ("tfidf", "all"):
        logger.info("Encoding TF-IDF...")
        encode_tfidf(texts, langs)
    if method in ("doc2vec", "all"):
        logger.info("Encoding Doc2Vec...")
        encode_doc2vec(texts, langs)
    if method in ("xlmroberta", "all"):
        logger.info("Encoding XLM-RoBERTa...")
        encode_xlmroberta(texts, langs)


def split_train_val_test_721(X: np.ndarray, y: np.ndarray):
    """Stratified 70% / 20% / 10%: tách test 10%, rồi từ 90% còn lại lấy val = 2/9."""
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=RANDOM_STATE, stratify=y_tmp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def count_svm_params(svm: SVC, n_features: int) -> int:
    n_sv = int(sum(svm.n_support_))
    n_classes = len(svm.classes_)
    sv_params = n_sv * n_features
    dual_params = (n_classes - 1) * n_sv
    bias_params = int(n_classes * (n_classes - 1) / 2)
    return sv_params + dual_params + bias_params


def _label_counts(y_arr: np.ndarray) -> dict[str, int]:
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    if y_arr.dtype == object or y_arr.dtype.kind in ("U", "S"):
        y_enc = le.transform(np.asarray(y_arr, dtype=str))
    else:
        y_enc = np.asarray(y_arr, dtype=int)
    out: dict[str, int] = {}
    for i, name in enumerate(CLASS_LABELS):
        out[name] = int((y_enc == i).sum())
    return out


def _svc_params_json(svm: SVC) -> dict[str, Any]:
    raw = svm.get_params(deep=False)
    out = {}
    for k, v in raw.items():
        if callable(v):
            out[k] = getattr(v, "__name__", str(v))
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


def _json_dict_key(k: Any) -> str | int | float | bool | None:
    """Chuyen khoa dict thanh kieu ma json.dump chap nhan (khong chap nhan numpy.int32)."""
    if k is None or isinstance(k, str):
        return k
    if isinstance(k, bool):
        return k
    if isinstance(k, (int, float)):
        return k
    if isinstance(k, np.generic):
        return k.item()
    return str(k)


def sanitize_for_json(obj: Any) -> Any:
    """Deep-copy cau truc sang kieu JSON-safe (khoa + gia tri, numpy, tuple, set)."""
    if isinstance(obj, dict):
        return {_json_dict_key(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, set):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return obj


def run_train():
    """Huấn luyện SVM cho từng encoder đã lưu: metrics train/val/test, plots, CSV, TXT, JSON."""
    all_results: dict[str, dict[str, Any]] = {}

    for file_key, display_name, pkl_name in ENCODED_SPECS:
        pkl_path = ENCODED_DIR / pkl_name
        if not pkl_path.exists():
            logger.warning("Bỏ qua (không có file): %s", pkl_path)
            continue

        logger.info("Huấn luyện SVM — %s", display_name)
        payload = joblib.load(pkl_path)
        X_raw = payload["X"].astype(np.float32)
        y_raw = payload["y"]
        meta = payload["metadata"]

        le = LabelEncoder()
        le.fit(CLASS_LABELS)
        y_enc = le.transform(y_raw)

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_721(X_raw, y_enc)
        n_tot = len(y_enc)
        logger.info(
            "Split 7/2/1 → train=%d (%.1f%%) | val=%d (%.1f%%) | test=%d (%.1f%%)",
            len(X_train),
            100 * len(X_train) / n_tot,
            len(X_val),
            100 * len(X_val) / n_tot,
            len(X_test),
            100 * len(X_test) / n_tot,
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        svm = SVC(**SVM_CONFIG)
        t0 = time.perf_counter()
        svm.fit(X_train_sc, y_train)
        svm_train_time = time.perf_counter() - t0

        y_pred_train = svm.predict(X_train_sc)
        y_pred_val = svm.predict(X_val_sc)
        y_pred_test = svm.predict(X_test_sc)

        train_metrics = compute_metrics(y_train, y_pred_train)
        val_metrics = compute_metrics(y_val, y_pred_val)
        test_metrics = compute_metrics(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])

        n_sv = int(sum(svm.n_support_))
        svm_params = count_svm_params(svm, X_train_sc.shape[1])
        enc_params = int(meta.get("encoder_params", 0))

        train_sizes_frac = np.linspace(0.1, 1.0, 8)
        lc_sizes, lc_train_scores, lc_val_scores = learning_curve(
            SVC(**SVM_CONFIG),
            X_train_sc,
            y_train,
            train_sizes=train_sizes_frac,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )

        model_path = MODELS_DIR / f"svm_model_{file_key}.pkl"
        model_pkg = {
            "model": svm,
            "scaler": scaler,
            "label_encoder": le,
            "method": file_key,
            "metrics_train": train_metrics,
            "metrics_val": val_metrics,
            "metrics_test": test_metrics,
            "n_features": int(X_train_sc.shape[1]),
            "class_labels": CLASS_LABELS,
            "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        }
        joblib.dump(model_pkg, model_path)

        all_results[display_name] = {
            "file_key": file_key,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "cm_test": cm_test,
            "svm_train_time_s": svm_train_time,
            "encoder_time_s": float(meta.get("encode_time_s", meta.get("total_time_s", 0))),
            "encoder_params": enc_params,
            "svm_params": svm_params,
            "total_params": enc_params + svm_params,
            "n_sv": n_sv,
            "n_features": X_train_sc.shape[1],
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "label_dist_full": meta.get("label_dist", {}),
            "train_label_counts": _label_counts(y_train),
            "val_label_counts": _label_counts(y_val),
            "test_label_counts": _label_counts(y_test),
            "lc_sizes": lc_sizes,
            "lc_train_scores": lc_train_scores,
            "lc_val_scores": lc_val_scores,
            "model_path": str(model_path),
            "classification_report_test": classification_report(
                y_test, y_pred_test, target_names=CLASS_LABELS, zero_division=0
            ),
            "metadata_encoder": {k: v for k, v in meta.items() if k != "cluster_scores"},
        }

    if not all_results:
        raise RuntimeError("Không có dữ liệu đã mã hoá. Chạy: python sentence_list/experiments.py encode --method all")

    methods = list(all_results.keys())

    # --- CSV metrics (train / val / test) ---
    rows = []
    for m, res in all_results.items():
        for split_name, mdict in [
            ("train", res["train_metrics"]),
            ("val", res["val_metrics"]),
            ("test", res["test_metrics"]),
        ]:
            rows.append(
                {
                    "method": m,
                    "split": split_name,
                    "accuracy": mdict["accuracy"],
                    "precision": mdict["precision"],
                    "recall": mdict["recall"],
                    "f1": mdict["f1"],
                    "svm_train_time_s": res["svm_train_time_s"],
                    "encoder_time_s": res["encoder_time_s"],
                    "encoder_params": res["encoder_params"],
                    "svm_params": res["svm_params"],
                    "total_params": res["total_params"],
                    "n_sv": res["n_sv"],
                    "n_features": res["n_features"],
                    "n_train": res["n_train"],
                    "n_val": res["n_val"],
                    "n_test": res["n_test"],
                }
            )
    metrics_df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "svm_comparison_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    # --- TXT báo cáo ---
    txt_path = RESULTS_DIR / "svm_comparison_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SVM — TF-IDF / Doc2Vec / XLM-RoBERTa (stratified 7/2/1)\n")
        f.write("=" * 80 + "\n\n")
        for m, res in all_results.items():
            f.write(f"{'─' * 60}\n  Phương pháp: {m}\n{'─' * 60}\n")
            f.write(f"  Đặc trưng: {res['n_features']} chiều\n")
            f.write(f"  Tham số encoder: {res['encoder_params']:,} | SVM: {res['svm_params']:,} | SV: {res['n_sv']}\n")
            f.write(f"  Thời gian encode: {res['encoder_time_s']:.2f}s | SVM: {res['svm_train_time_s']:.2f}s\n")
            f.write(f"  Model: {res['model_path']}\n\n")
            for split in ("train", "val", "test"):
                mm = res[f"{split}_metrics"]
                f.write(f"  --- {split.upper()} ---\n")
                for k, v in mm.items():
                    f.write(f"    {k:<12}: {v:.4f}\n")
            f.write("\n  Phân bố nhãn (train): " + str(res["train_label_counts"]) + "\n")
            f.write("  Phân bố nhãn (val):   " + str(res["val_label_counts"]) + "\n")
            f.write("  Phân bố nhãn (test):  " + str(res["test_label_counts"]) + "\n\n")
            f.write("  Classification report (TEST):\n")
            f.write(res["classification_report_test"] + "\n\n")

    # --- JSON tham số huấn luyện + kết quả ---
    svm_template = SVC(**SVM_CONFIG)
    training_report = {
        "data_split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "sklearn_train_test_split": {
                "step1_test_size": TEST_RATIO,
                "step2_val_fraction_of_trainval": VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            },
        },
        "feature_scaling": {"class": "sklearn.preprocessing.StandardScaler", "fit_on": "train_only"},
        "svm_hyperparameters": _svc_params_json(svm_template),
        "svm_config_used": {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in SVM_CONFIG.items()},
        "learning_curve": {
            "cv": 5,
            "scoring": "f1_macro",
            "train_sizes_fraction": train_sizes_frac.tolist(),
            "n_jobs": -1,
        },
        "class_labels": CLASS_LABELS,
        "methods": {},
    }
    for m, res in all_results.items():
        training_report["methods"][m] = {
            "file_key": res["file_key"],
            "counts": {"n_train": res["n_train"], "n_val": res["n_val"], "n_test": res["n_test"]},
            "metrics": {
                "train": res["train_metrics"],
                "val": res["val_metrics"],
                "test": res["test_metrics"],
            },
            "parameters": {
                "encoder_params": res["encoder_params"],
                "svm_params_estimate": res["svm_params"],
                "total_params_estimate": res["total_params"],
                "n_support_vectors": res["n_sv"],
                "n_features": res["n_features"],
            },
            "timing_seconds": {"encoder": res["encoder_time_s"], "svm_fit": res["svm_train_time_s"]},
            "model_path": res["model_path"],
            "encoder_metadata": res["metadata_encoder"],
        }
    json_path = RESULTS_DIR / "training_run_report.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sanitize_for_json(training_report), jf, ensure_ascii=False, indent=2, default=str)

    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Plot 1: metrics train / val / test
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("So sánh metrics SVM — Train / Validation / Test", fontsize=14, fontweight="bold", y=1.02)
    for ax_idx, (split_key, split_title) in enumerate([("train", "Train"), ("val", "Validation"), ("test", "Test")]):
        ax = axes[ax_idx]
        n_methods = len(methods)
        bar_width = 0.22
        x = np.arange(len(metric_keys))
        for i, m in enumerate(methods):
            vals = [all_results[m][f"{split_key}_metrics"][k] for k in metric_keys]
            offset = (i - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width,
                label=m,
                color=COLORS[m],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_title(split_title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p1 = PLOTS_DIR / "01_metrics_train_val_test.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: time + params
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Thời gian & tham số", fontsize=14, fontweight="bold")
    x_pos = np.arange(len(methods))
    enc_times = [all_results[m]["encoder_time_s"] for m in methods]
    svm_times = [all_results[m]["svm_train_time_s"] for m in methods]
    ax = axes[0]
    ax.bar(x_pos, enc_times, color=[COLORS[m] for m in methods], alpha=0.85, label="Encode", edgecolor="white")
    ax.bar(x_pos, svm_times, bottom=enc_times, color=[COLORS[m] for m in methods], alpha=0.5, hatch="//", label="SVM", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Giây")
    ax.set_title("Encode + SVM")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    enc_p = [all_results[m]["encoder_params"] for m in methods]
    bars = ax.bar(x_pos, enc_p, color=[COLORS[m] for m in methods], edgecolor="white")
    for bar, v in zip(bars, enc_p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, f"{v/1e6:.2f}M" if v >= 1e6 else f"{v:,}", ha="center", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Tham số encoder (log)")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    svm_p = [all_results[m]["svm_params"] for m in methods]
    n_svs = [all_results[m]["n_sv"] for m in methods]
    bars = ax.bar(x_pos, svm_p, color=[COLORS[m] for m in methods], edgecolor="white")
    for bar, v, sv in zip(bars, svm_p, n_svs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, f"{v:,}\n({sv} SV)", ha="center", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_title("Tham số SVM (dạng dual)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = PLOTS_DIR / "02_time_params.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: confusion matrices (test)
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    fig.suptitle("Ma trận nhầm lẫn — Test", fontsize=13, fontweight="bold")
    for ax, m in zip(axes, methods):
        cm = all_results[m]["cm_test"]
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
        )
        for i in range(len(CLASS_LABELS)):
            for j in range(len(CLASS_LABELS)):
                ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})", ha="center", va="center", fontsize=7, color="grey")
        acc = all_results[m]["test_metrics"]["accuracy"]
        f1 = all_results[m]["test_metrics"]["f1"]
        ax.set_title(f"{m}\nAcc={acc:.3f} F1={f1:.3f}", fontsize=10)
    plt.tight_layout()
    p3 = PLOTS_DIR / "03_confusion_matrices_test.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: learning curves
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]
    fig.suptitle("Learning curve (F1-macro, 5-fold CV trên tập train)", fontsize=13, fontweight="bold")
    for ax, m in zip(axes, methods):
        sizes = all_results[m]["lc_sizes"]
        tr_mean = all_results[m]["lc_train_scores"].mean(axis=1)
        tr_std = all_results[m]["lc_train_scores"].std(axis=1)
        cv_mean = all_results[m]["lc_val_scores"].mean(axis=1)
        cv_std = all_results[m]["lc_val_scores"].std(axis=1)
        col = COLORS[m]
        ax.plot(sizes, tr_mean, "o-", color=col, label="Train (CV)", linewidth=1.8)
        ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color=col)
        ax.plot(sizes, cv_mean, "s--", color=col, alpha=0.7, label="Val (CV)", linewidth=1.8)
        ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.1, color=col)
        ax.set_xlabel("Số mẫu train (trong fold)")
        ax.set_ylabel("F1-macro")
        ax.set_title(m)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p4 = PLOTS_DIR / "04_learning_curves.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 5: heatmap test metrics
    heat_data = pd.DataFrame({m: {k: all_results[m]["test_metrics"][k] for k in metric_keys} for m in methods}).T
    heat_data.columns = metric_labels
    fig, ax = plt.subplots(figsize=(8, max(3, len(methods) * 1.2)))
    sns.heatmap(heat_data, annot=True, fmt=".4f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.8)
    ax.set_title("Heatmap — Test")
    plt.tight_layout()
    p5 = PLOTS_DIR / "05_metrics_heatmap_test.png"
    plt.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 6: accuracy train vs val vs test (grouped by method)
    fig, ax = plt.subplots(figsize=(10, 5))
    splits = ["train", "val", "test"]
    x_m = np.arange(len(methods))
    w = 0.25
    for si, sp in enumerate(splits):
        offs = (si - 1) * w
        accs = [all_results[m][f"{sp}_metrics"]["accuracy"] for m in methods]
        ax.bar(x_m + offs, accs, width=w, label=sp.capitalize())
        for i, v in enumerate(accs):
            ax.text(x_m[i] + offs, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x_m)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy: Train / Validation / Test (7/2/1)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p6 = PLOTS_DIR / "06_accuracy_splits.png"
    plt.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Đã lưu CSV: %s", csv_path)
    logger.info("Đã lưu TXT: %s", txt_path)
    logger.info("Đã lưu JSON tham số & kết quả: %s", json_path)
    for p in (p1, p2, p3, p4, p5, p6):
        logger.info("Plot: %s", p)


def main():
    parser = argparse.ArgumentParser(description="Encode + train SVM (7/2/1), plots & JSON report.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="Mã hoá TF-IDF / Doc2Vec / XLM-RoBERTa.")
    p_encode.add_argument("--method", choices=["tfidf", "doc2vec", "xlmroberta", "all"], default="all")

    sub.add_parser("train", help="Huấn luyện SVM, xuất metrics/plots/JSON.")
    sub.add_parser("all", help="encode(all) rồi train.")

    args = parser.parse_args()
    if args.command == "encode":
        run_encode(args.method)
    elif args.command == "train":
        run_train()
    elif args.command == "all":
        run_encode("all")
        run_train()


if __name__ == "__main__":
    main()
