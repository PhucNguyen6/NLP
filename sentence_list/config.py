# -*- coding: utf-8 -*-
"""
Configuration module for RAG + LLM Sentiment Analysis System
Centralized settings for database, embeddings, LLM, and performance
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ============ PROJECT PATHS ============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "clean_data"
ENCODED_DIR = BASE_DIR / "encoded_data"
MODELS_DIR = BASE_DIR / "results" / "svm_models"
CACHE_DIR = BASE_DIR / "cache"
DICT_DIR = BASE_DIR / "clean_dict"

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
        'vector_size': 100,
        'window': 5,
        'epochs': 40,
        'dm': 1,  # Distributed Memory
    },
    'tfidf': {
        'max_features': 500,
        'ngram_range': (1, 2),
    }
}

# ============ RAG CONFIGURATION ============
RAG_CONFIG = {
    'top_k': 5,  # Number of similar documents to retrieve
    'similarity_threshold': 0.3,  # Min similarity score
    'chunk_size': 100,  # Words per chunk
    'chunk_overlap': 20,  # Overlap between chunks
    'max_context_tokens': 2000,  # Max tokens in context for LLM
}

# ============ LLM CONFIGURATION ============
LLM_CONFIG = {
    'lm_studio_url': os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1'),
    'model_name': os.getenv('LLM_MODEL', 'mistral'),
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
    'use_gpu': True,  # Use GPU if available
    'num_workers': 4,
    'embedding_cache_size': 10000,  # Max cached embeddings
}

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
