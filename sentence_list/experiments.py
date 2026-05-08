"""Unified experiments entrypoint for encode/train/benchmark.

Usage examples:
  python sentence_list/experiments.py encode --method all
  python sentence_list/experiments.py train
  python sentence_list/experiments.py all
"""

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_define import analyze_sentiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EXP] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
CLEAN_COMMENT_CSV = BASE / "clean_data" / "clean_comment.csv"
ENCODED_DIR = BASE / "encoded_data"
RESULTS_DIR = BASE / "results"
MODELS_DIR = RESULTS_DIR / "svm_models"

CLASS_LABELS = ["criticism", "neutral", "praise"]
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

ENCODED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    # Use slow tokenizer to avoid auto-conversion issues on SentencePiece models.
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


def train_one(method_key: str, pkl_path: Path):
    payload = joblib.load(pkl_path)
    x = payload["X"].astype(np.float32)
    y_raw = payload["y"]
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    y = le.transform(y_raw)
    x_tmp, x_test, y_tmp, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=2 / 9, random_state=42, stratify=y_tmp
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_precision": float(precision_score(y_val, val_pred, average="macro", zero_division=0)),
        "val_recall": float(recall_score(y_val, val_pred, average="macro", zero_division=0)),
        "val_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
    }
    model_pkg = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "method": method_key,
        "metrics_test": {k[5:]: v for k, v in metrics.items() if k.startswith("test_")},
        "metrics_val": {k[4:]: v for k, v in metrics.items() if k.startswith("val_")},
        "n_features": int(x.shape[1]),
        "class_labels": CLASS_LABELS,
    }
    out = MODELS_DIR / f"svm_model_{method_key}.pkl"
    joblib.dump(model_pkg, out)
    return {"method": method_key, **metrics}


def run_train():
    datasets = {
        "tf_idf": ENCODED_DIR / "tfidf_labeled.pkl",
        "doc2vec": ENCODED_DIR / "doc2vec_labeled.pkl",
        "xlm_roberta": ENCODED_DIR / "xlmroberta_labeled.pkl",
    }
    rows = []
    for key, path in datasets.items():
        if not path.exists():
            logger.warning("Missing encoded file: %s", path)
            continue
        logger.info("Training SVM for %s", key)
        rows.append(train_one(key, path))
    if not rows:
        raise RuntimeError("No dataset available for training.")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "svm_comparison_metrics.csv", index=False)
    (RESULTS_DIR / "svm_comparison_summary.txt").write_text(
        metrics_df.to_string(index=False), encoding="utf-8"
    )
    logger.info("Saved training summary to %s", RESULTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Unified experiments for sentence_list.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="Encode dataset into tfidf/doc2vec/xlmroberta artifacts.")
    p_encode.add_argument("--method", choices=["tfidf", "doc2vec", "xlmroberta", "all"], default="all")

    sub.add_parser("train", help="Train SVM models from encoded artifacts.")
    sub.add_parser("all", help="Run encode(all) then train.")

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
