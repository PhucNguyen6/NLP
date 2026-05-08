# -*- coding: utf-8 -*-
"""
Embeddings Manager - Handles embedding generation and caching
Supports multiple embedding models: XLM-RoBERTa, Doc2Vec, TF-IDF
"""

import logging
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from config import EMBEDDING_CONFIG, OPTIMIZATION_CONFIG, LOGGING_CONFIG, ENCODED_DIR
from database import get_db_manager

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                   format=LOGGING_CONFIG['format'])
logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages text embeddings using multiple models"""
    
    def __init__(self):
        """Initialize embedding models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                   and OPTIMIZATION_CONFIG['use_gpu'] else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load XLM-RoBERTa model
        self.xlmroberta_model = None
        self.xlmroberta_tokenizer = None
        self._load_xlmroberta()
        
        # Load Doc2Vec model
        self.doc2vec_model = None
        self._load_doc2vec()

        # TF-IDF vectorizer (lazy-load + fallback rebuild)
        self.tfidf_vectorizer = None
        self._load_tfidf_vectorizer()
        
        # Embedding cache
        self.embedding_cache = {}
        self.cache_max_size = OPTIMIZATION_CONFIG['embedding_cache_size']
    
    def _load_xlmroberta(self):
        """Load XLM-RoBERTa model and tokenizer"""
        try:
            model_name = EMBEDDING_CONFIG['xlmroberta']['model_name']
            logger.info(f"Loading XLM-RoBERTa model: {model_name}")
            
            self.xlmroberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.xlmroberta_model = AutoModel.from_pretrained(model_name)
            self.xlmroberta_model.to(self.device)
            self.xlmroberta_model.eval()
            
            logger.info("XLM-RoBERTa model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XLM-RoBERTa: {e}")
            raise
    
    def _load_doc2vec(self):
        """Load Doc2Vec model from saved pickle"""
        try:
            model_path = ENCODED_DIR / 'doc2vec_model.model'
            if model_path.exists():
                from gensim.models import Doc2Vec
                self.doc2vec_model = Doc2Vec.load(str(model_path))
                logger.info("Doc2Vec model loaded successfully")
            else:
                logger.warning(f"Doc2Vec model not found at {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Doc2Vec: {e}")

    def _build_doc2vec_from_labeled(self) -> None:
        """
        Rebuild Doc2Vec model from `doc2vec_labeled.pkl` when `doc2vec_model.model` is missing.
        The rebuild result is cached to avoid repeated heavy training.
        """
        try:
            from gensim.models import Doc2Vec, TaggedDocument

            payload_path = ENCODED_DIR / "doc2vec_labeled.pkl"
            if not payload_path.exists():
                logger.error("doc2vec_labeled.pkl missing; cannot rebuild Doc2Vec model.")
                return

            payload = joblib.load(payload_path)
            texts = payload.get("texts") or []
            d2v_config = payload.get("metadata", {}).get("d2v_config", {}) or {}

            vector_size = int(d2v_config.get("vector_size", 100))
            window = int(d2v_config.get("window", 5))
            min_count = int(d2v_config.get("min_count", 1))
            epochs = int(d2v_config.get("epochs", 40))
            dm = int(d2v_config.get("dm", 1))

            if not texts:
                logger.error("doc2vec_labeled.pkl has no texts; cannot rebuild Doc2Vec.")
                return

            logger.info(
                "Rebuilding Doc2Vec model from %s (this can take time)...",
                payload_path,
            )

            tagged_data = [
                TaggedDocument(words=str(text).split(), tags=[str(i)])
                for i, text in enumerate(texts)
            ]

            # Keep gensim workers small-ish by default; user can adjust via their env.
            workers = int(OPTIMIZATION_CONFIG.get("num_workers", 4)) if isinstance(
                OPTIMIZATION_CONFIG.get("num_workers", 4), int
            ) else 4

            model = Doc2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=epochs,
                dm=dm,
                seed=42,
            )
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            # Cache built model
            model_path = ENCODED_DIR / "doc2vec_model.model"
            model.save(str(model_path))
            self.doc2vec_model = model
            logger.info("Doc2Vec model rebuilt and cached at %s", model_path)
        except Exception as e:
            logger.error("Failed to rebuild Doc2Vec model: %s", e)

    def _ensure_doc2vec_model(self) -> None:
        if self.doc2vec_model is not None:
            return
        # Try rebuild if missing
        self._build_doc2vec_from_labeled()
    
    def embed_text_xlmroberta(self, text: str) -> np.ndarray:
        """
        Generate XLM-RoBERTa embedding for text
        
        Args:
            text: Input text
            
        Returns:
            768-dimensional embedding vector
        """
        if not self.xlmroberta_model:
            raise RuntimeError("XLM-RoBERTa model not loaded")
        
        # Check cache
        cache_key = f"xlmroberta:{text[:100]}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Tokenize and encode
            max_length = EMBEDDING_CONFIG['xlmroberta']['max_length']
            inputs = self.xlmroberta_tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.xlmroberta_model(**inputs, output_hidden_states=True)
                # Use CLS token embedding (first token of last hidden state)
                embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
            
            # Cache result
            if len(self.embedding_cache) < self.cache_max_size:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating XLM-RoBERTa embedding: {e}")
            raise
    
    def embed_text_doc2vec(self, tokens: List[str]) -> np.ndarray:
        """
        Generate Doc2Vec embedding for tokenized text
        
        Args:
            tokens: List of tokens
            
        Returns:
            100-dimensional embedding vector
        """
        self._ensure_doc2vec_model()
        if not self.doc2vec_model:
            logger.warning("Doc2Vec model not loaded, returning zero vector")
            return np.zeros(EMBEDDING_CONFIG['doc2vec']['vector_size'])
        
        try:
            embedding = self.doc2vec_model.infer_vector(tokens)
            return embedding
        except Exception as e:
            logger.error(f"Error generating Doc2Vec embedding: {e}")
            return np.zeros(EMBEDDING_CONFIG['doc2vec']['vector_size'])
    
    def embed_batch_xlmroberta(self, texts: List[str], 
                              batch_size: int = None) -> List[np.ndarray]:
        """
        Generate embeddings for batch of texts efficiently
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or EMBEDDING_CONFIG['xlmroberta']['batch_size']
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            try:
                max_length = EMBEDDING_CONFIG['xlmroberta']['max_length']
                inputs = self.xlmroberta_tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                
                with torch.no_grad():
                    outputs = self.xlmroberta_model(**inputs, output_hidden_states=True)
                    batch_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                
                embeddings.extend(batch_embeddings)
                logger.debug(f"Processed batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Add zero vectors for failed texts
                embeddings.extend([np.zeros(768) for _ in batch_texts])
        
        return embeddings
    
    def embed_all_encoded_data(self) -> Dict[str, Dict]:
        """
        Generate embeddings for all encoded data in the project
        Returns mapping of dataset to embeddings and metadata
        
        Args:
            Returns:
            Dict with keys: 'xlmroberta', 'doc2vec', 'tfidf'
        """
        db = get_db_manager()
        results = {}
        
        # Load pre-computed embeddings
        for model_name in ['tfidf', 'doc2vec', 'xlmroberta']:
            model_path = ENCODED_DIR / f'{model_name}_labeled.pkl'
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    logger.info(f"Loaded {model_name} embeddings: {data['X'].shape}")
                    results[model_name] = data
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
        
        return results
    
    def embed_sentence(self, text: str, models: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a sentence using specified models
        
        Args:
            text: Input text
            models: List of models to use (default: all)
            
        Returns:
            Dict mapping model names to embedding vectors
        """
        models = models or EMBEDDING_CONFIG['models']
        embeddings = {}
        
        if 'xlmroberta' in models:
            embeddings['xlmroberta'] = self.embed_text_xlmroberta(text)
        
        if 'doc2vec' in models:
            self._ensure_doc2vec_model()
        if 'doc2vec' in models and self.doc2vec_model:
            # Tokenize text for doc2vec
            tokens = text.lower().split()
            embeddings['doc2vec'] = self.embed_text_doc2vec(tokens)
        
        if 'tfidf' in models:
            self._ensure_tfidf_vectorizer()
            if self.tfidf_vectorizer is not None:
                try:
                    tfidf_vec = self.tfidf_vectorizer.transform([text]).toarray().squeeze()
                    embeddings['tfidf'] = tfidf_vec
                except Exception as e:
                    logger.error(f"Error generating TF-IDF embedding: {e}")
        
        return embeddings

    def _load_tfidf_vectorizer(self) -> None:
        """Load TF-IDF vectorizer if cached; otherwise keep as None for rebuild."""
        try:
            vectorizer_path = ENCODED_DIR / "tfidf_vectorizer.pkl"
            if vectorizer_path.exists():
                with open(vectorizer_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("TF-IDF vectorizer loaded successfully")
            else:
                logger.warning("TF-IDF vectorizer not found; will rebuild on demand.")
        except Exception as e:
            logger.error("Failed to load TF-IDF vectorizer: %s", e)

    def _build_tfidf_from_labeled(self) -> None:
        """Rebuild TF-IDF vectorizer from `tfidf_labeled.pkl` when cache is missing."""
        try:
            payload_path = ENCODED_DIR / "tfidf_labeled.pkl"
            if not payload_path.exists():
                logger.error("tfidf_labeled.pkl missing; cannot rebuild TF-IDF vectorizer.")
                return

            payload = joblib.load(payload_path)
            texts = payload.get("texts") or []
            tfidf_config = payload.get("metadata", {}).get("tfidf_config", {}) or {}
            if not texts:
                logger.error("tfidf_labeled.pkl has no texts; cannot rebuild TF-IDF vectorizer.")
                return

            logger.info("Rebuilding TF-IDF vectorizer from %s...", payload_path)

            vectorizer = TfidfVectorizer(
                max_features=int(tfidf_config.get("max_features", 500)),
                ngram_range=tuple(tfidf_config.get("ngram_range", (1, 2))),
                sublinear_tf=bool(tfidf_config.get("sublinear_tf", True)),
                min_df=int(tfidf_config.get("min_df", 2)),
            )
            vectorizer.fit(texts)

            vectorizer_path = ENCODED_DIR / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)
            self.tfidf_vectorizer = vectorizer
            logger.info("TF-IDF vectorizer rebuilt and cached at %s", vectorizer_path)
        except Exception as e:
            logger.error("Failed to rebuild TF-IDF vectorizer: %s", e)

    def _ensure_tfidf_vectorizer(self) -> None:
        if self.tfidf_vectorizer is not None:
            return
        self._build_tfidf_from_labeled()
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit vector (L2 normalization)
        
        Args:
            embedding: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize embeddings
        emb1_norm = self.normalize_embedding(emb1)
        emb2_norm = self.normalize_embedding(emb2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return max(0, min(1, similarity))  # Clamp to [0, 1]
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")


# Global embeddings manager instance
_embeddings_manager = None


def get_embeddings_manager() -> EmbeddingsManager:
    """
    Get or create embeddings manager singleton
    
    Returns:
        EmbeddingsManager instance
    """
    global _embeddings_manager
    if _embeddings_manager is None:
        _embeddings_manager = EmbeddingsManager()
    return _embeddings_manager
