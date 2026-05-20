# -*- coding: utf-8 -*-
"""
Đánh giá RAG: truy vấn mẫu, thống kê similarity/latency, xuất JSON + TXT + biểu đồ.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DICT_DIR, EMBEDDING_CONFIG, LOGGING_CONFIG, RAG_CONFIG
from database import get_db_manager
from config import DICT_CHUNKS_JSON
from experiments import sanitize_for_json
from rag_retriever import get_rag_retriever

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
RAG_RESULTS_DIR = BASE / "results" / "rag"
RAG_PLOTS_DIR = BASE / "plots" / "rag"

MODEL_LABELS = {
    "xlmroberta": "XLM-RoBERTa",
    "doc2vec": "Doc2Vec",
    "tfidf": "TF-IDF",
}
COLORS = {
    "XLM-RoBERTa": "#FF9800",
    "Doc2Vec": "#4CAF50",
    "TF-IDF": "#2196F3",
}


def load_sample_queries(max_queries: int = 150, random_state: int = 42) -> list[str]:
    path = DATA_DIR / "clean_comment.csv"
    if not path.exists():
        return [
            "Bài hát này rất hay!",
            "Không thích giai điệu này.",
            "Bình thường, không có gì đặc biệt.",
        ]
    df = pd.read_csv(path, encoding="utf-8")
    col = "comment" if "comment" in df.columns else df.columns[0]
    texts = df[col].astype(str).str.strip()
    texts = texts[texts != ""]
    if len(texts) <= max_queries:
        return texts.tolist()
    return texts.sample(n=max_queries, random_state=random_state).tolist()


def load_dictionary_chunk_stats() -> dict[str, Any]:
    if not DICT_CHUNKS_JSON.exists():
        return {"available": False, "path": str(DICT_CHUNKS_JSON)}
    with open(DICT_CHUNKS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    chunks = data.get("chunks", [])
    lang_counts: dict[str, int] = {}
    for c in chunks:
        lang = c.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    return {
        "available": True,
        "path": str(DICT_CHUNKS_JSON),
        "total_chunks": len(chunks),
        "metadata": meta,
        "by_language": lang_counts,
    }


def evaluate_retrieval_for_model(
    queries: list[str],
    model: str,
    top_k: int,
    threshold: float,
) -> dict[str, Any]:
    retriever = get_rag_retriever(top_k=top_k, similarity_threshold=threshold)
    top1_sims: list[float] = []
    topk_mean_sims: list[float] = []
    latencies_ms: list[float] = []
    n_results_list: list[int] = []
    per_query: list[dict[str, Any]] = []

    for q in tqdm(queries, desc=f"RAG/{model}", unit="query"):
        t0 = time.perf_counter()
        docs = retriever.retrieve_context(q, top_k=top_k, model=model)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        n_results_list.append(len(docs))
        sims = [float(d.get("similarity", 0.0)) for d in docs]
        if sims:
            top1_sims.append(sims[0])
            topk_mean_sims.append(float(np.mean(sims)))
        else:
            top1_sims.append(0.0)
            topk_mean_sims.append(0.0)
        per_query.append(
            {
                "query_preview": q[:120],
                "n_retrieved": len(docs),
                "top1_similarity": sims[0] if sims else None,
                "latency_ms": round(elapsed_ms, 2),
            }
        )

    top1_arr = np.array(top1_sims, dtype=np.float64)
    threshold_pass = float(np.mean(top1_arr >= threshold)) if len(top1_arr) else 0.0

    return {
        "model": model,
        "label": MODEL_LABELS.get(model, model),
        "n_queries": len(queries),
        "top_k": top_k,
        "similarity_threshold": threshold,
        "metrics": {
            "avg_top1_similarity": float(np.mean(top1_arr)) if len(top1_arr) else 0.0,
            "median_top1_similarity": float(np.median(top1_arr)) if len(top1_arr) else 0.0,
            "std_top1_similarity": float(np.std(top1_arr)) if len(top1_arr) else 0.0,
            "avg_topk_mean_similarity": float(np.mean(topk_mean_sims)) if topk_mean_sims else 0.0,
            "threshold_pass_rate": threshold_pass,
            "avg_n_retrieved": float(np.mean(n_results_list)) if n_results_list else 0.0,
            "queries_with_any_hit": int(sum(1 for n in n_results_list if n > 0)),
            "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            "p50_latency_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        },
        "distributions": {
            "top1_similarities": top1_sims,
            "latencies_ms": latencies_ms,
            "n_retrieved": n_results_list,
        },
        "sample_queries": per_query[:20],
    }


def evaluate_dictionary_retrieval(
    queries: list[str],
    model: str,
    dict_top_k: int,
    dict_threshold: float,
) -> dict[str, Any]:
    retriever = get_rag_retriever()
    top1_sims: list[float] = []
    latencies_ms: list[float] = []
    n_hits: list[int] = []

    for q in tqdm(queries, desc=f"DICT/{model}", unit="query"):
        t0 = time.perf_counter()
        docs = retriever.retrieve_dictionary_context(q, top_k=dict_top_k, model=model)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        n_hits.append(len(docs))
        sims = [float(d.get("similarity", 0.0)) for d in docs]
        top1_sims.append(sims[0] if sims else 0.0)

    arr = np.array(top1_sims, dtype=np.float64)
    return {
        "model": model,
        "n_queries": len(queries),
        "dict_top_k": dict_top_k,
        "dictionary_similarity_threshold": dict_threshold,
        "metrics": {
            "avg_top1_similarity": float(np.mean(arr)) if len(arr) else 0.0,
            "threshold_pass_rate": float(np.mean(arr >= dict_threshold)) if len(arr) else 0.0,
            "avg_n_retrieved": float(np.mean(n_hits)) if n_hits else 0.0,
            "queries_with_any_hit": int(sum(1 for n in n_hits if n > 0)),
            "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        },
        "distributions": {"top1_similarities": top1_sims, "latencies_ms": latencies_ms},
    }


def evaluate_hybrid_for_model(
    queries: list[str],
    model: str,
    top_k: int,
    dict_top_k: int,
    threshold: float,
    comment_m: dict[str, Any] | None = None,
    dict_m: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chỉ đo latency hybrid; tái dùng metric RAG/DICT đã tính (tránh chạy lại 150 query x3)."""
    retriever = get_rag_retriever(top_k=top_k, similarity_threshold=threshold)
    if comment_m is None:
        comment_m = evaluate_retrieval_for_model(queries, model, top_k, threshold)
    if dict_m is None:
        dict_m = evaluate_dictionary_retrieval(
            queries, model, dict_top_k, float(RAG_CONFIG.get("dictionary_similarity_threshold", 0.25))
        )
    latencies: list[float] = []
    for q in tqdm(queries, desc=f"HYBRID/{model}", unit="query"):
        t0 = time.perf_counter()
        retriever.retrieve_hybrid_context(q, top_k=top_k, dict_top_k=dict_top_k, model=model)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        "model": model,
        "comments": {k: v for k, v in comment_m.items() if k != "distributions"},
        "dictionary": {k: v for k, v in dict_m.items() if k != "distributions"},
        "hybrid_latency_ms": {
            "avg": float(np.mean(latencies)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "distributions": {
            "comment_top1": comment_m.get("distributions", {}).get("top1_similarities", []),
            "dictionary_top1": dict_m.get("distributions", {}).get("top1_similarities", []),
        },
    }


def plot_rag_results(all_models: dict[str, dict[str, Any]], db_stats: dict[str, Any]) -> list[str]:
    RAG_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    labels = [all_models[m]["label"] for m in all_models]
    colors = [COLORS.get(lbl, "#888888") for lbl in labels]

    # 1 — Similarity & latency bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RAG — Similarity & thời gian truy vấn", fontsize=13, fontweight="bold")
    avg_sim = [all_models[m]["metrics"]["avg_top1_similarity"] for m in all_models]
    avg_lat = [all_models[m]["metrics"]["avg_latency_ms"] for m in all_models]
    x = np.arange(len(labels))
    axes[0].bar(x, avg_sim, color=colors, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Cosine similarity (top-1)")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(RAG_CONFIG["similarity_threshold"], color="red", linestyle="--", label="threshold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, avg_lat, color=colors, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("ms / truy vấn")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = RAG_PLOTS_DIR / "01_rag_similarity_latency.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(p1))

    # 2 — Histogram top-1 similarity
    fig, axes = plt.subplots(1, len(all_models), figsize=(5 * len(all_models), 4))
    if len(all_models) == 1:
        axes = [axes]
    fig.suptitle("Phân bố similarity top-1", fontsize=13, fontweight="bold")
    for ax, (mkey, res) in zip(axes, all_models.items()):
        sims = res["distributions"]["top1_similarities"]
        ax.hist(sims, bins=20, color=COLORS.get(res["label"], "#888"), alpha=0.85, edgecolor="white")
        ax.axvline(RAG_CONFIG["similarity_threshold"], color="red", linestyle="--")
        ax.set_title(res["label"])
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Số truy vấn")
    plt.tight_layout()
    p2 = RAG_PLOTS_DIR / "02_top1_similarity_histogram.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(p2))

    # 3 — Threshold pass rate & avg retrieved
    fig, ax = plt.subplots(figsize=(8, 5))
    pass_rates = [all_models[m]["metrics"]["threshold_pass_rate"] for m in all_models]
    avg_n = [all_models[m]["metrics"]["avg_n_retrieved"] for m in all_models]
    w = 0.35
    ax.bar(x - w / 2, pass_rates, width=w, label="Tỷ lệ ≥ threshold (top-1)", color=colors, alpha=0.9)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, avg_n, width=w, label="TB số doc retrieve", color=colors, alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tỷ lệ")
    ax2.set_ylabel("Số document")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Tỷ lệ vượt ngưỡng & số kết quả trả về")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p3 = RAG_PLOTS_DIR / "03_threshold_pass_and_retrieved.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(p3))

    # 4 — Latency boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    data_lat = [all_models[m]["distributions"]["latencies_ms"] for m in all_models]
    bp = ax.boxplot(data_lat, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Phân bố thời gian truy vấn RAG")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p4 = RAG_PLOTS_DIR / "04_retrieval_latency_boxplot.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(p4))

    # 5 — DB label distribution
    dist = db_stats.get("label_distribution") or {}
    if dist:
        fig, ax = plt.subplots(figsize=(6, 5))
        keys = list(dist.keys())
        vals = [dist[k] for k in keys]
        ax.pie(vals, labels=keys, autopct="%1.1f%%", startangle=140)
        ax.set_title("Phân bố nhãn trong embeddings DB")
        plt.tight_layout()
        p5 = RAG_PLOTS_DIR / "05_db_sentiment_distribution.png"
        plt.savefig(p5, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(str(p5))

    return saved


def write_text_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "=" * 72,
        "BÁO CÁO ĐÁNH GIÁ RAG",
        "=" * 72,
        f"Thời gian: {report.get('created_at', '')}",
        f"Số truy vấn mẫu: {report.get('n_queries', 0)}",
        "",
        "--- Tham số RAG ---",
    ]
    for k, v in report.get("rag_config", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("--- Cơ sở dữ liệu ---")
    db = report.get("database", {})
    lines.append(f"  Tổng embeddings: {db.get('total_embeddings', 0)}")
    lines.append(f"  Tổng từ điển (DB): {db.get('total_words', 0)}")
    lines.append(f"  Phân bố nhãn: {db.get('label_distribution', {})}")
    lines.append("")
    dict_chunk = report.get("dictionary_chunks", {})
    if dict_chunk.get("available"):
        lines.append("--- Chunk từ điển (JSON) ---")
        lines.append(f"  File: {dict_chunk.get('path')}")
        lines.append(f"  Tổng chunk: {dict_chunk.get('total_chunks', 0)}")
        lines.append(f"  Theo ngôn ngữ: {dict_chunk.get('by_language', {})}")
        lines.append("")
    lines.append("--- Kết quả theo encoder ---")
    for mkey, res in report.get("models", {}).items():
        lines.append(f"\n  [{res.get('label', mkey)}]")
        met = res.get("metrics", {})
        lines.append(f"    avg top-1 similarity : {met.get('avg_top1_similarity', 0):.4f}")
        lines.append(f"    threshold pass rate  : {met.get('threshold_pass_rate', 0):.4f}")
        lines.append(f"    avg n retrieved      : {met.get('avg_n_retrieved', 0):.2f}")
        lines.append(f"    avg latency (ms)     : {met.get('avg_latency_ms', 0):.2f}")
        lines.append(f"    p95 latency (ms)       : {met.get('p95_latency_ms', 0):.2f}")
    lines.append("\n--- Retrieve từ điển (hybrid, theo encoder) ---")
    for mkey, res in report.get("dictionary_retrieval_by_encoder", {}).items():
        met = res.get("metrics", {})
        lines.append(
            f"  [{mkey}] sim={met.get('avg_top1_similarity', 0):.4f} "
            f"hits={met.get('queries_with_any_hit', 0)}/{res.get('n_queries', 0)}"
        )
    lines.append("")
    lines.append("Biểu đồ:")
    for p in report.get("plots", []):
        lines.append(f"  - {p}")
    lines.append("=" * 72)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(
    max_queries: int = 150,
    models: list[str] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    RAG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = models or list(EMBEDDING_CONFIG["models"])
    top_k = top_k or int(RAG_CONFIG["top_k"])
    threshold = threshold if threshold is not None else float(RAG_CONFIG["similarity_threshold"])

    queries = load_sample_queries(max_queries=max_queries)
    db_stats: dict[str, Any] = {}
    try:
        db = get_db_manager()
        db_stats = db.get_statistics()
    except Exception as e:
        logger.warning("Không kết nối DB: %s", e)
        db_stats = {"error": str(e), "total_embeddings": 0}

    if db_stats.get("total_embeddings", 0) == 0:
        logger.warning("DB không có embedding — metric retrieval có thể bằng 0.")

    all_models: dict[str, dict[str, Any]] = {}
    for model in models:
        logger.info("Đánh giá RAG — model=%s", model)
        try:
            res = evaluate_retrieval_for_model(queries, model, top_k, threshold)
            # Bỏ distribution khỏi bản JSON chính (giữ trong file riêng nếu cần)
            slim = {k: v for k, v in res.items() if k != "distributions"}
            all_models[model] = res
        except Exception as e:
            logger.error("Lỗi model %s: %s", model, e)
            all_models[model] = {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "error": str(e),
                "metrics": {},
            }

    dict_top_k = int(RAG_CONFIG.get("dict_top_k", 3))
    dict_thresh = float(RAG_CONFIG.get("dictionary_similarity_threshold", 0.25))
    hybrid_models: dict[str, Any] = {}
    dict_by_encoder: dict[str, Any] = {}
    for model in models:
        try:
            logger.info("Đánh giá từ điển — model=%s", model)
            dict_by_encoder[model] = evaluate_dictionary_retrieval(
                queries, model, dict_top_k, dict_thresh
            )
            comment_res = all_models.get(model, {})
            hybrid_models[model] = evaluate_hybrid_for_model(
                queries,
                model,
                top_k,
                dict_top_k,
                threshold,
                comment_m=comment_res if "error" not in comment_res else None,
                dict_m=dict_by_encoder[model],
            )
        except Exception as e:
            hybrid_models[model] = {"error": str(e)}

    dict_chunks = load_dictionary_chunk_stats()
    plot_inputs = {m: all_models[m] for m in all_models if "distributions" in all_models[m]}
    plots = plot_rag_results(plot_inputs, db_stats) if plot_inputs else []

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(queries),
        "rag_config": dict(RAG_CONFIG),
        "embedding_config": {
            "primary_model": EMBEDDING_CONFIG["primary_model"],
            "models": EMBEDDING_CONFIG["models"],
        },
        "database": db_stats,
        "dictionary_chunks": dict_chunks,
        "hybrid_retrieval": {
            m: {k: v for k, v in hybrid_models[m].items() if k != "distributions"}
            for m in hybrid_models
        },
        "dictionary_retrieval_by_encoder": {
            m: {k: v for k, v in dict_by_encoder[m].items() if k != "distributions"}
            for m in dict_by_encoder
        },
        "models": {
            m: {k: v for k, v in all_models[m].items() if k != "distributions"}
            for m in all_models
        },
        "plots": plots,
    }

    json_path = RAG_RESULTS_DIR / "rag_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(report), f, ensure_ascii=False, indent=2)

    txt_path = RAG_RESULTS_DIR / "rag_evaluation_summary.txt"
    write_text_summary(report, txt_path)

    # Lưu phân bố similarity cho phân tích sau
    dist_path = RAG_RESULTS_DIR / "rag_similarity_distributions.json"
    dist_payload = {
        m: all_models[m].get("distributions", {})
        for m in all_models
        if "distributions" in all_models[m]
    }
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(dist_payload), f, ensure_ascii=False, indent=2)

    logger.info("Báo cáo RAG: %s", json_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá RAG và xuất báo cáo + biểu đồ")
    parser.add_argument("--max-queries", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="xlmroberta doc2vec tfidf",
    )
    args = parser.parse_args()

    try:
        report = run_evaluation(
            max_queries=args.max_queries,
            models=args.models,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        print("\n✓ Hoàn tất đánh giá RAG")
        print(f"  JSON: {RAG_RESULTS_DIR / 'rag_evaluation_report.json'}")
        print(f"  TXT:  {RAG_RESULTS_DIR / 'rag_evaluation_summary.txt'}")
        print(f"  Plots: {RAG_PLOTS_DIR}/")
        for m, res in report.get("models", {}).items():
            if "error" in res:
                print(f"  [{m}] Lỗi: {res['error']}")
            else:
                met = res.get("metrics", {})
                print(
                    f"  [{m}] sim={met.get('avg_top1_similarity', 0):.3f} "
                    f"lat={met.get('avg_latency_ms', 0):.1f}ms"
                )
        return 0
    except Exception as e:
        logger.exception("Đánh giá RAG thất bại: %s", e)
        print(f"✗ {e}")
        return 1


# Gọi qua: python -m sentence_list.cli db eval
