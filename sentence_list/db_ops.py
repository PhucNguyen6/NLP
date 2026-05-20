# -*- coding: utf-8 -*-
"""PostgreSQL: schema, nạp embedding bình luận & từ điển, đánh giá RAG."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from tqdm import tqdm

from config import DB_CONFIG, ENCODED_DIR, LOGGING_CONFIG
from database import get_db_manager
from dictionary import chunk_embed_text, get_dictionary_chunk_store
from embeddings_manager import get_embeddings_manager

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
)
logger = logging.getLogger(__name__)


def run_setup(skip_confirm: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("PostgreSQL — tạo schema")
    print("=" * 60)
    print(f"  {DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['database']}")
    if not skip_confirm:
        if input("Tiếp tục? (yes/no): ").strip().lower() not in ("yes", "y"):
            print("Đã hủy.")
            return False
    db = get_db_manager()
    db.setup_schema()
    stats = db.get_statistics()
    print(f"✓ Xong. embeddings={stats['total_embeddings']}, words={stats['total_words']}")
    return True


def _load_encoded_data(model_name: str):
    model_path = ENCODED_DIR / f"{model_name}_labeled.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def _populate_embeddings_from_file(model_name: str = "xlmroberta") -> int:
    data = _load_encoded_data(model_name)
    if data is None:
        logger.warning("Không có file encoded: %s_labeled.pkl", model_name)
        return 0
    db = get_db_manager()
    X, y = data["X"], data["y"]
    total = 0
    for i in range(0, len(X), 100):
        records = []
        for j, (embedding, label) in enumerate(zip(X[i : i + 100], y[i : i + 100])):
            ex, doc, xlm, tf = None, None, None, None
            if model_name == "xlmroberta":
                xlm = embedding.tolist()
            elif model_name == "doc2vec":
                doc = embedding.tolist()
            elif model_name == "tfidf":
                tf = embedding.tolist()
            records.append(
                (
                    f"{model_name}_{i + j}",
                    f"Encoded {model_name} #{i + j}",
                    xlm,
                    doc,
                    tf,
                    label,
                    "vi",
                    '{"source": "encoded_data"}',
                )
            )
        total += db.batch_insert_embeddings(records)
    return total


def run_populate_embeddings(force: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("Nạp embedding bình luận vào DB")
    print("=" * 60)
    db = get_db_manager()
    n = db.get_statistics().get("total_embeddings", 0)
    if n > 0 and not force:
        print(f"DB đã có {n} embedding — bỏ qua (dùng --force để nạp lại).")
        return True
    total = 0
    for model in ("tfidf", "doc2vec", "xlmroberta"):
        total += _populate_embeddings_from_file(model)
    print(f"✓ Đã nạp {total} bản ghi.")
    return True


def run_populate_dictionary(batch_size: int = 64) -> bool:
    from config import DICT_CHUNKS_JSON

    store = get_dictionary_chunk_store()
    if store.load_chunks() == 0:
        print(f"Thiếu chunk. Chạy: python -m sentence_list.cli dict build")
        return False
    db = get_db_manager()
    emb = get_embeddings_manager()
    chunks = store._chunks
    inserted = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Dictionary DB"):
        batch = chunks[i : i + batch_size]
        texts = [chunk_embed_text(c) for c in batch]
        vectors = emb.embed_batch_xlmroberta(texts)
        for chunk, vec in zip(batch, vectors):
            word = str(chunk.get("word", "")).strip()
            if not word:
                continue
            try:
                db.insert_dictionary_entry(
                    word=word,
                    definition=str(chunk.get("semantics", "")),
                    language=str(chunk.get("language", "vi")),
                    embedding=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                    source="dictionary_chunks.json",
                )
                inserted += 1
            except Exception:
                pass
    print(f"✓ Dictionary: {inserted} mục (file chunk: {DICT_CHUNKS_JSON})")
    return inserted > 0


def run_rag_eval(**kwargs) -> bool:
    from rag_metrics import run_evaluation

    run_evaluation(**kwargs)
    return True
