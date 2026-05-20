# -*- coding: utf-8 -*-
"""
Từ điển: crawl/clean CSV, chunk JSON, vector search (RAG).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CACHE_DIR, DICT_CHUNKS_JSON, DICT_DIR, LOGGING_CONFIG, RAG_CONFIG  # noqa: F401
from embeddings_manager import get_embeddings_manager
from utils import chunk_text

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = DICT_CHUNKS_JSON
DEFAULT_VI_OUTPUT = DICT_DIR / "dictionary_chunks_vi.json"
DEFAULT_EN_OUTPUT = DICT_DIR / "dictionary_chunks_en.json"
# --- CSV crawl/clean ---
def _ensure_dict_dirs():
    os.makedirs(DICT_DIR, exist_ok=True)
    os.makedirs("raw_dict", exist_ok=True)


def _definition_column(df: pd.DataFrame) -> str | None:
    for col in ("definition", "meaning", "definition_en", "semantics"):
        if col in df.columns:
            return col
    return None


def _word_column(df: pd.DataFrame) -> str | None:
    for col in ("word", "word_en", "term"):
        if col in df.columns:
            return col
    return None


def _pos_column(df: pd.DataFrame) -> str | None:
    for col in ("pos", "part_of_speech", "type"):
        if col in df.columns:
            return col
    return None


def crawl_vi_dict(output_path=None):
    _ensure_dict_dirs()
    output_path = output_path or "raw_dict/vi_dict.csv"
    df = pd.read_csv("hf://datasets/tsdocode/vietnamese-dictionary/vi_dictionary.csv")
    df.to_csv(output_path, index=False)
    return output_path


def _normalize_dict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột → word, definition, pos (schema HF có thể đổi)."""
    df = df.copy()
    rename: dict[str, str] = {}
    wcol = _word_column(df)
    dcol = _definition_column(df)
    pcol = _pos_column(df)
    if wcol and wcol != "word":
        rename[wcol] = "word"
    if dcol and dcol != "definition":
        rename[dcol] = "definition"
    if pcol and pcol != "pos":
        rename[pcol] = "pos"
    if rename:
        df = df.rename(columns=rename)
    return df


def crawl_en_dict(output_path=None):
    _ensure_dict_dirs()
    output_path = output_path or "raw_dict/en_dict.csv"
    df = pd.read_csv("hf://datasets/mrrtmob/english-khmer-dictionary/dictionary.csv")
    df = _normalize_dict_df(df)
    keep_cols = [c for c in ("word", "definition", "pos") if c in df.columns]
    if keep_cols:
        df = df[keep_cols]
    df.to_csv(output_path, index=False)
    return output_path


def clean_vi_dict(input_path="raw_dict/vi_dict.csv", output_path=None):
    output_path = output_path or str(DICT_DIR / "vi_dict.csv")
    df = _normalize_dict_df(pd.read_csv(input_path))
    if "word" not in df.columns or "definition" not in df.columns:
        raise KeyError(f"CSV VI thiếu word/definition. Cột hiện có: {list(df.columns)}")
    df = df.dropna(subset=["word", "definition"])
    for col in ("word", "definition"):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["word"] != "") & (df["definition"] != "")]
    df.drop_duplicates(subset=["word", "definition"], keep="first").to_csv(output_path, index=False)
    return output_path


def clean_en_dict(input_path="raw_dict/en_dict.csv", output_path=None):
    output_path = output_path or str(DICT_DIR / "en_dict.csv")
    df = _normalize_dict_df(pd.read_csv(input_path))
    if "word" not in df.columns or "definition" not in df.columns:
        raise KeyError(f"CSV EN thiếu word/definition. Cột hiện có: {list(df.columns)}")
    df = df.dropna(subset=["word", "definition"])
    for col in ("word", "definition"):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["word"] != "") & (df["definition"] != "")]
    df.drop_duplicates(subset=["word", "definition"], keep="first").to_csv(output_path, index=False)
    return output_path


def _safe_crawl(crawl_fn, label: str, raw_path: str) -> str:
    """Crawl HF; nếu lỗi mạng thì dùng file raw local (nếu có)."""
    try:
        return crawl_fn()
    except (ConnectionError, OSError) as e:
        if Path(raw_path).exists():
            logger.warning(
                "%s: không kết nối HuggingFace (%s) — dùng file local %s",
                label,
                e,
                raw_path,
            )
            return raw_path
        raise ConnectionError(
            f"{label}: không tải được từ HuggingFace (kiểm tra mạng/DNS). "
            f"Đặt CSV vào {raw_path} hoặc chạy: python -m sentence_list.cli dict build --skip-crawl"
        ) from e


def build_dictionary(export_chunks: bool = True, prefer_local: bool = True) -> dict[str, Any]:
    """Crawl HF → clean CSV → (tùy chọn) chunk JSON. Ưu tiên file clean_dict/ nếu đã có."""
    vi_clean_path = DICT_DIR / "vi_dict.csv"
    en_clean_path = DICT_DIR / "en_dict.csv"
    info: dict[str, Any] = {}

    if prefer_local and vi_clean_path.exists():
        logger.info("Dùng từ điển VI có sẵn: %s", vi_clean_path)
        info["vi"] = str(vi_clean_path)
    else:
        vi_raw = _safe_crawl(crawl_vi_dict, "VI", "raw_dict/vi_dict.csv")
        info["vi"] = clean_vi_dict(vi_raw)

    if prefer_local and en_clean_path.exists():
        logger.info("Dùng từ điển EN có sẵn: %s", en_clean_path)
        info["en"] = str(en_clean_path)
    else:
        try:
            en_raw = _safe_crawl(crawl_en_dict, "EN", "raw_dict/en_dict.csv")
            info["en"] = clean_en_dict(en_raw)
        except ConnectionError as e:
            logger.warning("Bỏ qua từ điển EN: %s", e)
            info["en"] = None

    if export_chunks:
        info.update(export_dictionary_chunks())
    return info


# --- Chunk CSV → JSON ---


def load_dictionary_csv(path: Path, language: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Không tìm thấy từ điển: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    wcol = _word_column(df)
    dcol = _definition_column(df)
    if not wcol or not dcol:
        raise ValueError(f"File {path} thiếu cột word/definition (có: {list(df.columns)})")

    rename = {wcol: "word", dcol: "definition"}
    pcol = _pos_column(df)
    if pcol and pcol not in rename:
        rename[pcol] = "part_of_speech"
    df = df.rename(columns=rename)
    df["word"] = df["word"].astype(str).str.strip()
    df["definition"] = df["definition"].astype(str).str.strip()
    df = df[(df["word"] != "") & (df["definition"] != "")]
    df = df.drop_duplicates(subset=["word", "definition"], keep="first")
    df["language"] = language
    if "part_of_speech" not in df.columns:
        df["part_of_speech"] = None
    return df


def entry_to_chunks(
    word: str,
    semantics: str,
    language: str,
    part_of_speech: str | None,
    source_file: str,
    entry_index: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Một mục từ điển -> một hoặc nhiều chunk JSON."""
    text = (semantics or "").strip()
    if not text:
        text = word

    parts = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not parts:
        parts = [text]

    n_parts = len(parts)
    lang_prefix = language[:2] if language else "xx"
    chunks: list[dict[str, Any]] = []
    for ci, sem in enumerate(parts):
        suffix = f"_{ci}" if n_parts > 1 else ""
        chunks.append(
            {
                "chunk_id": f"{lang_prefix}_{entry_index:06d}{suffix}",
                "word": word,
                "semantics": sem,
                "language": language,
                "part_of_speech": part_of_speech,
                "source_file": source_file,
                "chunk_index": ci,
                "total_chunks_for_entry": n_parts,
            }
        )
    return chunks


def build_chunks_from_dataframe(
    df: pd.DataFrame,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
    id_offset: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        entry_idx = id_offset + idx
        pos = row.get("part_of_speech")
        pos_str = str(pos).strip() if pos is not None and str(pos) != "nan" else None
        out.extend(
            entry_to_chunks(
                word=str(row["word"]),
                semantics=str(row["definition"]),
                language=str(row.get("language", "unknown")),
                part_of_speech=pos_str,
                source_file=source_file,
                entry_index=entry_idx,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return out


def export_dictionary_chunks(
    vi_path: Path | None = None,
    en_path: Path | None = None,
    output_path: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """
    Xuất chunk từ điển ra JSON (gộp + theo ngôn ngữ).

    Returns:
        Metadata dict (đường dẫn file, số lượng chunk, …)
    """
    chunk_size = chunk_size or int(RAG_CONFIG["chunk_size"])
    chunk_overlap = chunk_overlap or int(RAG_CONFIG["chunk_overlap"])
    vi_path = vi_path or DICT_DIR / "vi_dict.csv"
    en_path = en_path or DICT_DIR / "en_dict.csv"
    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict[str, Any]] = []
    per_lang: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {"languages": {}, "files_read": []}

    vi_df = load_dictionary_csv(vi_path, "vi")
    if not vi_df.empty:
        vi_chunks = build_chunks_from_dataframe(
            vi_df, vi_path.name, chunk_size, chunk_overlap, id_offset=0
        )
        per_lang["vi"] = vi_chunks
        all_chunks.extend(vi_chunks)
        stats["languages"]["vi"] = {
            "entries": len(vi_df),
            "chunks": len(vi_chunks),
            "source": str(vi_path),
        }
        stats["files_read"].append(str(vi_path))

    en_df = load_dictionary_csv(en_path, "en")
    if not en_df.empty:
        en_chunks = build_chunks_from_dataframe(
            en_df, en_path.name, chunk_size, chunk_overlap, id_offset=len(vi_df)
        )
        per_lang["en"] = en_chunks
        all_chunks.extend(en_chunks)
        stats["languages"]["en"] = {
            "entries": len(en_df),
            "chunks": len(en_chunks),
            "source": str(en_path),
        }
        stats["files_read"].append(str(en_path))

    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "metadata": {
            "created_at": created_at,
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": chunk_overlap,
            "total_chunks": len(all_chunks),
            "total_entries": sum(s.get("entries", 0) for s in stats["languages"].values()),
            "languages": list(stats["languages"].keys()),
            "stats": stats,
        },
        "chunks": all_chunks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if "vi" in per_lang:
        with open(DEFAULT_VI_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": {**payload["metadata"], "language": "vi"}, "chunks": per_lang["vi"]},
                f,
                ensure_ascii=False,
                indent=2,
            )
    if "en" in per_lang:
        with open(DEFAULT_EN_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": {**payload["metadata"], "language": "en"}, "chunks": per_lang["en"]},
                f,
                ensure_ascii=False,
                indent=2,
            )

    logger.info("Đã xuất %d chunk -> %s", len(all_chunks), output_path)
    return {
        "output": str(output_path),
        "vi_output": str(DEFAULT_VI_OUTPUT) if "vi" in per_lang else None,
        "en_output": str(DEFAULT_EN_OUTPUT) if "en" in per_lang else None,
        "total_chunks": len(all_chunks),
        "metadata": payload["metadata"],
    }


# --- Vector store (dictionary_store) ---
TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def chunk_embed_text(chunk: dict[str, Any]) -> str:
    word = str(chunk.get("word", "")).strip()
    sem = str(chunk.get("semantics", "")).strip()
    return f"{word}: {sem}" if word and sem else (word or sem)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class DictionaryChunkStore:
    def __init__(self, chunks_path: Path | None = None):
        self.chunks_path = chunks_path or DICT_CHUNKS_JSON
        self._chunks: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self._matrix_norm: np.ndarray | None = None
        self._embed_model: str | None = None
        self._loaded = False
        self._embeddings_manager = None

    def load_chunks(self, force: bool = False) -> int:
        if self._loaded and not force:
            return len(self._chunks)
        if not self.chunks_path.exists():
            self._chunks, self._matrix, self._loaded = [], None, True
            return 0
        with open(self.chunks_path, encoding="utf-8") as f:
            self._chunks = list(json.load(f).get("chunks") or [])
        self._matrix = None
        self._matrix_norm = None
        self._embed_model = None
        self._loaded = True
        return len(self._chunks)

    def _normalize_matrix(self) -> None:
        if self._matrix is None or len(self._matrix) == 0:
            self._matrix_norm = None
            return
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._matrix_norm = (self._matrix / norms).astype(np.float32)

    def _ensure_embeddings(self, model: str = "xlmroberta") -> None:
        """Nạp/build embedding từ điển một lần — tránh np.load lại mỗi query."""
        if self._matrix is not None and self._embed_model == model:
            return
        self._build_embeddings(model=model)

    def _build_embeddings(self, model: str = "xlmroberta", batch_size: int = 64) -> None:
        if not self._chunks:
            self.load_chunks()
        if not self._chunks:
            self._matrix = np.zeros((0, 1), dtype=np.float32)
            self._normalize_matrix()
            self._embed_model = model
            return
        npy = CACHE_DIR / f"dict_chunks_{model.replace('/', '_')}.npy"
        if npy.exists():
            try:
                self._matrix = np.load(npy)
                self._normalize_matrix()
                self._embed_model = model
                logger.info(
                    "Dictionary embedding cache loaded (1 lần): %s %s",
                    npy.name,
                    self._matrix.shape,
                )
                return
            except Exception as e:
                logger.warning("Không đọc được cache %s: %s — sẽ build lại.", npy, e)
        if self._embeddings_manager is None:
            self._embeddings_manager = get_embeddings_manager()
        texts = [chunk_embed_text(c) for c in self._chunks]
        vectors = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="DICT-EMB/build",
            unit="batch",
        ):
            batch = texts[i : i + batch_size]
            vectors.extend(self._embeddings_manager.embed_batch_xlmroberta(batch))
        self._matrix = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
        self._normalize_matrix()
        self._embed_model = model
        try:
            np.save(npy, self._matrix)
            logger.info("Dictionary embedding cache saved: %s", npy)
        except Exception as e:
            logger.warning("Không lưu được cache %s: %s", npy, e)

    def search(self, query: str, top_k: int = 3, model: str = "xlmroberta", threshold: float | None = None) -> list:
        threshold = threshold or float(RAG_CONFIG.get("dictionary_similarity_threshold", 0.25))
        self.load_chunks()
        if not self._chunks:
            return []
        self._ensure_embeddings(model=model)
        if self._embeddings_manager is None:
            self._embeddings_manager = get_embeddings_manager()
        q = np.asarray(self._embeddings_manager.embed_text_xlmroberta(query), dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-10)
        mat = self._matrix_norm
        if mat is None or len(mat) == 0:
            return []
        sims = mat @ qn
        # Lấy ứng viên top rồi lọc ngưỡng (nhanh hơn quét Python 200k+ dòng/query)
        cand_k = min(max(top_k * 50, top_k), len(sims))
        idxs = np.argpartition(-sims, cand_k - 1)[:cand_k]
        scored = [(float(sims[i]), int(i)) for i in idxs if sims[i] >= threshold]
        scored.sort(reverse=True)
        out = []
        for sim, idx in scored[:top_k]:
            ch = self._chunks[idx]
            out.append({
                "source": "dictionary",
                "chunk_id": ch.get("chunk_id"),
                "word": ch.get("word"),
                "semantics": ch.get("semantics"),
                "text_content": chunk_embed_text(ch),
                "similarity": float(sim),
            })
        return out


_store: DictionaryChunkStore | None = None


def get_dictionary_chunk_store() -> DictionaryChunkStore:
    global _store
    if _store is None:
        _store = DictionaryChunkStore()
    return _store
