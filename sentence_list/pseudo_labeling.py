# -*- coding: utf-8 -*-
"""
Gán nhãn giả (pseudo-label) cho TF-IDF / Doc2Vec — thay thế KMeans thuần
để tránh lệch lớp (vd. gần hết criticism/neutral, rất ít praise).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import ENCODED_DIR, PSEUDO_LABEL_CONFIG
from sentiment_define import analyze_sentiment

logger = logging.getLogger(__name__)

CLASS_LABELS = ("criticism", "neutral", "praise")


def _lexicon_scores_and_labels(texts: list[str], langs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(texts), dtype=np.float32)
    labels = np.empty(len(texts), dtype=object)
    for i, text in enumerate(texts):
        lang = langs[i] if i < len(langs) else "vi"
        try:
            result = analyze_sentiment(text, language=lang)
            scores[i] = float(result.get("score", 0.0))
            labels[i] = result.get("label", "neutral")
        except Exception:
            scores[i] = 0.0
            labels[i] = "neutral"
    return scores, labels


def load_teacher_labels(n_samples: int, teacher_pkl: str = "xlmroberta_labeled.pkl") -> np.ndarray | None:
    path = ENCODED_DIR / teacher_pkl
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        y = np.asarray(payload.get("y"))
        if len(y) != n_samples:
            logger.warning(
                "Teacher %s có %d mẫu, cần %d — bỏ qua teacher labeling.",
                teacher_pkl,
                len(y),
                n_samples,
            )
            return None
        return y
    except Exception as e:
        logger.warning("Không đọc được teacher labels: %s", e)
        return None


def map_clusters_to_labels(
    cluster_ids: np.ndarray,
    texts: list[str],
    langs: list[str],
    sample_per_cluster: int | None = None,
) -> tuple[np.ndarray, dict[int, float], dict[int, str]]:
    """Map 3 cluster → criticism / neutral / praise theo điểm sentiment trung bình cluster."""
    sample_per_cluster = sample_per_cluster or int(PSEUDO_LABEL_CONFIG.get("cluster_sample_per_cluster", 500))
    cluster_scores: dict[int, float] = {}
    for cid in sorted(set(cluster_ids.tolist())):
        idxs = np.where(cluster_ids == cid)[0]
        if len(idxs) > sample_per_cluster:
            rng = np.random.default_rng(42)
            idxs = rng.choice(idxs, size=sample_per_cluster, replace=False)
        scores = []
        for i in idxs:
            lang = langs[i] if i < len(langs) else "vi"
            try:
                scores.append(analyze_sentiment(texts[i], language=lang).get("score", 0.0))
            except Exception:
                scores.append(0.0)
        cluster_scores[int(cid)] = float(np.median(scores)) if scores else 0.0

    sorted_cids = sorted(cluster_scores, key=lambda c: cluster_scores[c])
    if len(sorted_cids) < 3:
        label_map = {sorted_cids[0]: "neutral"} if sorted_cids else {}
    else:
        label_map = {
            sorted_cids[0]: "criticism",
            sorted_cids[1]: "neutral",
            sorted_cids[2]: "praise",
        }
    y = np.array([label_map.get(int(c), "neutral") for c in cluster_ids], dtype=object)
    return y, cluster_scores, label_map


def assign_labels_lexicon(texts: list[str], langs: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    scores, labels = _lexicon_scores_and_labels(texts, langs)
    meta = {
        "labeling_mode": "lexicon",
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    }
    return labels, meta


def assign_labels_percentile(
    texts: list[str],
    langs: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    scores, _ = _lexicon_scores_and_labels(texts, langs)
    p_low = float(PSEUDO_LABEL_CONFIG.get("percentile_low", 33.0))
    p_high = float(PSEUDO_LABEL_CONFIG.get("percentile_high", 67.0))
    t_low, t_high = np.percentile(scores, [p_low, p_high])
    y = np.where(
        scores <= t_low,
        "criticism",
        np.where(scores >= t_high, "praise", "neutral"),
    )
    meta = {
        "labeling_mode": "percentile",
        "percentile_low": p_low,
        "percentile_high": p_high,
        "threshold_low": float(t_low),
        "threshold_high": float(t_high),
    }
    return y.astype(object), meta


def assign_labels_cluster_scaled(
    texts: list[str],
    langs: list[str],
    X: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    X_scaled = StandardScaler().fit_transform(X)
    cluster_ids = KMeans(
        n_clusters=3,
        random_state=int(PSEUDO_LABEL_CONFIG.get("kmeans_random_state", 42)),
        n_init=int(PSEUDO_LABEL_CONFIG.get("kmeans_n_init", 20)),
        max_iter=500,
    ).fit_predict(X_scaled)
    y, cluster_scores, label_map = map_clusters_to_labels(cluster_ids, texts, langs)
    meta = {
        "labeling_mode": "cluster_scaled",
        "cluster_scores": cluster_scores,
        "label_map": label_map,
    }
    return y, meta


def assign_labels_hybrid(
    texts: list[str],
    langs: list[str],
    X: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Lexicon cho mẫu rõ ràng; KMeans (trên X chuẩn hóa) chỉ cho vùng biên |score| <= borderline.
    """
    borderline = float(PSEUDO_LABEL_CONFIG.get("borderline_score", 0.5))
    scores, lex_labels = _lexicon_scores_and_labels(texts, langs)
    borderline_mask = np.abs(scores) <= borderline

    X_scaled = StandardScaler().fit_transform(X)
    cluster_ids = KMeans(
        n_clusters=3,
        random_state=int(PSEUDO_LABEL_CONFIG.get("kmeans_random_state", 42)),
        n_init=int(PSEUDO_LABEL_CONFIG.get("kmeans_n_init", 20)),
        max_iter=500,
    ).fit_predict(X_scaled)
    cluster_labels, cluster_scores, label_map = map_clusters_to_labels(cluster_ids, texts, langs)

    y = lex_labels.copy()
    y[borderline_mask] = cluster_labels[borderline_mask]

    meta = {
        "labeling_mode": "hybrid",
        "borderline_score": borderline,
        "n_borderline": int(borderline_mask.sum()),
        "n_lexicon_fixed": int((~borderline_mask).sum()),
        "cluster_scores": cluster_scores,
        "label_map": label_map,
    }
    return y, meta


def assign_pseudo_labels(
    texts: list[str],
    langs: list[str],
    X: np.ndarray,
    mode: str | None = None,
    method_name: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    mode: auto | teacher | lexicon | percentile | hybrid | cluster
    auto: teacher (XLM) nếu có file → else hybrid
    """
    mode = (mode or PSEUDO_LABEL_CONFIG.get("mode", "auto")).lower().strip()
    n = len(texts)

    if mode == "auto":
        teacher = load_teacher_labels(n)
        if teacher is not None:
            mode = "teacher"
        else:
            mode = str(PSEUDO_LABEL_CONFIG.get("auto_fallback", "hybrid"))

    if mode == "teacher":
        teacher = load_teacher_labels(n)
        if teacher is None:
            logger.warning("[%s] Không có teacher — fallback hybrid.", method_name)
            y, meta = assign_labels_hybrid(texts, langs, X)
        else:
            y = np.asarray(teacher, dtype=object)
            meta = {"labeling_mode": "teacher", "teacher_file": "xlmroberta_labeled.pkl"}
        return y, meta

    if mode == "lexicon":
        return assign_labels_lexicon(texts, langs)
    if mode == "percentile":
        return assign_labels_percentile(texts, langs)
    if mode == "cluster":
        return assign_labels_cluster_scaled(texts, langs, X)
    if mode == "hybrid":
        return assign_labels_hybrid(texts, langs, X)

    logger.warning("Mode '%s' không hợp lệ — dùng hybrid.", mode)
    return assign_labels_hybrid(texts, langs, X)


def label_distribution(y: np.ndarray) -> dict[str, int]:
    return {k: int((y == k).sum()) for k in CLASS_LABELS}
