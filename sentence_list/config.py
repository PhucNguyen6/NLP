# -*- coding: utf-8 -*-
"""
Configuration module for RAG + LLM Sentiment Analysis System
Centralized settings for database, embeddings, LLM, and performance
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ============ PROJECT PATHS ============
BASE_DIR = Path(__file__).parent

# Load .env từ sentence_list/ (không phụ thuộc cwd khi chạy từ NLP/)
load_dotenv(BASE_DIR / ".env")
load_dotenv()
DATA_DIR = BASE_DIR / "clean_data"
ENCODED_DIR = BASE_DIR / "encoded_data"
MODELS_DIR = BASE_DIR / "results" / "svm_models"
CACHE_DIR = BASE_DIR / "cache"
DICT_DIR = BASE_DIR / "clean_dict"
DICT_CHUNKS_JSON = DICT_DIR / "dictionary_chunks.json"
RAG_RESULTS_DIR = BASE_DIR / "results" / "rag"
RAG_PLOTS_DIR = BASE_DIR / "plots" / "rag"

# Create cache dir if not exists
CACHE_DIR.mkdir(exist_ok=True)

# ============ DATABASE CONFIGURATION ============
# PostgreSQL connection settings
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'nlp_sentiment_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
}

# ============ EMBEDDING CONFIGURATION ============
# Model choices: 'xlmroberta', 'doc2vec', 'tfidf'
EMBEDDING_CONFIG = {
    'primary_model': 'xlmroberta',  # Main model for new queries
    'models': ['xlmroberta', 'doc2vec', 'tfidf'],
    'xlmroberta': {
        'model_name': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'hidden_size': 768,
        'max_length': 128,
        'batch_size': 32,
    },
    'doc2vec': {
        'vector_size': 200,
        'window': 8,
        'min_count': 3,
        'epochs': 50,
        'dm': 1,
        'dbow_words': 1,
        'infer_epochs': 30,
        'negative': 8,
        'workers': 4,
    },
    'tfidf': {
        'max_features': 1500,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'min_df': 3,
        'max_df': 0.92,
    },
}

# Gán nhãn cho TF-IDF / Doc2Vec (pseudo-label)
# auto = dùng nhãn XLM nếu đã encode xlmroberta; không thì hybrid (lexicon + cluster vùng biên)
PSEUDO_LABEL_CONFIG = {
    'mode': os.getenv('PSEUDO_LABEL_MODE', 'auto'),
    'auto_fallback': 'hybrid',
    'borderline_score': 0.5,
    'percentile_low': 33.0,
    'percentile_high': 67.0,
    'cluster_sample_per_cluster': 500,
    'kmeans_random_state': 42,
    'kmeans_n_init': 20,
}

# ============ RAG CONFIGURATION ============
RAG_CONFIG = {
    'top_k': 5,  # Number of similar comment documents to retrieve
    'dict_top_k': 3,  # Number of dictionary chunks to retrieve
    'similarity_threshold': 0.3,  # Min similarity for comment retrieval
    'dictionary_similarity_threshold': 0.25,  # Min similarity for dictionary chunks
    'hybrid_retrieval': True,  # Combine comment RAG + dictionary chunks
    'chunk_size': 100,  # Words per chunk
    'chunk_overlap': 20,  # Overlap between chunks
    'max_context_tokens': 2000,  # Max tokens in context for LLM
}

# ============ LLM CONFIGURATION ============
LLM_CONFIG = {
    'lm_studio_url': os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1'),
    'model_name': os.getenv('LLM_MODEL', 'Vistral-7B-ChatML-GGUF'),
    'temperature': 0.7,
    'max_tokens': 500,
    'top_p': 0.95,
    'top_k': 50,
    'timeout': 30,
}

# ============ SENTIMENT CONFIGURATION ============
SENTIMENT_CONFIG = {
    'labels': ['praise', 'neutral', 'criticism'],
    'label_to_id': {'praise': 0, 'neutral': 1, 'criticism': 2},
    'id_to_label': {0: 'praise', 1: 'neutral', 2: 'criticism'},
}

# ============ DICTIONARY EXPANSION CONFIG ============
DICT_EXPANSION_CONFIG = {
    'auto_crawl_enabled': True,
    'crawl_sources': [
        'https://dictionary.cambridge.org',
        'https://www.dictionary.com',
        'https://www.merriam-webster.com',
    ],
    'vietnamese_sources': [
        'https://www.hoasen.edu.vn',
        'https://www.dichvuduconline.com',
    ],
    'batch_crawl_size': 10,
    'crawl_timeout': 10,
    'cache_ttl': 86400,  # 24 hours
}

# ============ PERFORMANCE OPTIMIZATION ============
OPTIMIZATION_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hour
    'batch_processing': True,
    'batch_size': 32,
    # USE_GPU=0 hoặc FORCE_CPU=1 trong .env để ép CPU
    'use_gpu': os.getenv('USE_GPU', '1').lower() not in ('0', 'false', 'no'),
    'num_workers': 4,
    'embedding_cache_size': 10000,  # Max cached embeddings
    # Huấn luyện SVM: learning curve 5-fold rất tốn CPU (có thể tắt bằng --skip-learning-curve)
    'svm_learning_curve': os.getenv('SVM_LEARNING_CURVE', '1').lower() not in ('0', 'false', 'no'),
    'learning_curve_cv': 5,
    'learning_curve_n_jobs': int(os.getenv('LEARNING_CURVE_JOBS', '-1')),
}


def torch_cuda_diagnostics() -> dict:
    """Thông tin GPU/CUDA — dùng khi debug USE_GPU không hoạt động."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "use_gpu_env": os.getenv("USE_GPU", "1"),
        "train_device_env": os.getenv("TRAIN_DEVICE", "auto"),
    }
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
    else:
        info["device_count"] = 0
        if "+cpu" in torch.__version__.lower():
            info["hint"] = (
                "PyTorch đang là bản CPU (+cpu). Cài bản CUDA, ví dụ:\n"
                "  pip uninstall torch -y\n"
                "  pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121"
            )
        else:
            info["hint"] = "torch.cuda.is_available()=False — kiểm tra driver NVIDIA (nvidia-smi)."
    return info


def log_torch_cuda_status(logger=None) -> dict:
    import logging

    log = logger or logging.getLogger(__name__)
    info = torch_cuda_diagnostics()
    log.info("PyTorch %s | CUDA available: %s", info["torch_version"], info["cuda_available"])
    if info["cuda_available"]:
        log.info("GPU: %s (CUDA %s)", info.get("device_name"), info.get("cuda_version"))
    elif info.get("hint"):
        log.warning("%s", info["hint"])
    return info


def get_torch_device(prefer: str | None = None):
    """
    Chọn thiết bị PyTorch cho encode / embedding.
    prefer: 'auto' | 'cuda' | 'cpu' — hoặc đặt TRAIN_DEVICE trong .env
    """
    import logging
    import torch

    log = logging.getLogger(__name__)
    choice = (prefer or os.getenv("TRAIN_DEVICE", "auto")).strip().lower()
    force_cpu = os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")
    want_gpu = choice == "cuda" or (
        choice == "auto" and OPTIMIZATION_CONFIG.get("use_gpu", True)
    )

    if force_cpu or choice == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available() and want_gpu:
        dev = torch.device("cuda")
        log.info("Dùng GPU: %s", torch.cuda.get_device_name(0))
        return dev

    if want_gpu:
        log.warning(
            "Yêu cầu GPU (USE_GPU=%s, --device=%s) nhưng CUDA không khả dụng → CPU. %s",
            os.getenv("USE_GPU", "1"),
            prefer or choice,
            torch_cuda_diagnostics().get("hint", ""),
        )
    return torch.device("cpu")

# ============ CRAWLING CONFIGURATION ============
CRAWLING_CONFIG = {
    'youtube_api_key': os.getenv('YOUTUBE_API_KEY'),
    'crawl_batch_size': 50,
    'max_retries': 3,
    'retry_delay': 5,
    'timeout': 30,
}

# ============ LOGGING ============
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'log_file': BASE_DIR / 'logs' / 'system.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# Create log directory
LOGGING_CONFIG['log_file'].parent.mkdir(exist_ok=True)


def setup_project_logging() -> None:
    """UTF-8 cho console + file log (tránh lỗi cp1252 trên Windows)."""
    import logging
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    handlers = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOGGING_CONFIG["log_file"], encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=handlers,
        force=True,
    )
