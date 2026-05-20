# -*- coding: utf-8 -*-
"""
RAG Retriever - Retrieval-Augmented Generation system
Retrieves top K similar vectors from database for context
"""

import logging
from typing import List, Dict, Tuple, Optional
import hashlib
import json

import numpy as np

from config import RAG_CONFIG, LOGGING_CONFIG, EMBEDDING_CONFIG
from database import get_db_manager
from dictionary import get_dictionary_chunk_store
from embeddings_manager import get_embeddings_manager

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                   format=LOGGING_CONFIG['format'])
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant context for queries using vector similarity"""
    
    def __init__(self, top_k: int = None, similarity_threshold: float = None):
        """
        Initialize RAG retriever
        
        Args:
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.top_k = top_k or RAG_CONFIG['top_k']
        self.dict_top_k = int(RAG_CONFIG.get('dict_top_k', 3))
        self.similarity_threshold = similarity_threshold or RAG_CONFIG['similarity_threshold']
        self.dict_threshold = float(RAG_CONFIG.get('dictionary_similarity_threshold', 0.25))
        self.hybrid_retrieval = bool(RAG_CONFIG.get('hybrid_retrieval', True))
        self.db = get_db_manager()
        self.embeddings = get_embeddings_manager()
        self.dict_store = get_dictionary_chunk_store()
        self.cache = {}
    
    def _query_embedding(self, query: str, model: str) -> np.ndarray:
        """Vector truy vấn theo đúng cột embedding trong DB (xlmroberta / doc2vec / tfidf)."""
        m = (model or "xlmroberta").lower().replace("-", "_")
        if m == "tf_idf":
            m = "tfidf"
        if m == "xlm_roberta":
            m = "xlmroberta"
        if m == "xlmroberta":
            return np.asarray(self.embeddings.embed_text_xlmroberta(query), dtype=np.float32)
        if m == "doc2vec":
            return np.asarray(self.embeddings.embed_text_doc2vec(query.lower().split()), dtype=np.float32)
        if m == "tfidf":
            vec = self.embeddings.embed_sentence(query, models=["tfidf"]).get("tfidf")
            if vec is None:
                dim = int(EMBEDDING_CONFIG.get("tfidf", {}).get("max_features", 500))
                return np.zeros(dim, dtype=np.float32)
            return np.asarray(vec, dtype=np.float32).reshape(-1)
        return np.asarray(self.embeddings.embed_text_xlmroberta(query), dtype=np.float32)

    def retrieve_context(self, query: str, top_k: int = None,
                        model: str = 'xlmroberta') -> List[Dict]:
        """
        Retrieve similar documents for a query
        
        Args:
            query: Input query text
            top_k: Number of results to return
            model: Embedding model to use
            
        Returns:
            List of similar documents with metadata
        """
        top_k = top_k or self.top_k
        
        try:
            query_embedding = self._query_embedding(query, model)
            
            # Search similar embeddings in database
            similar_docs = self.db.search_similar_embeddings(
                embedding=query_embedding.tolist(),
                top_k=top_k,
                threshold=self.similarity_threshold,
                model=model
            )
            
            for doc in similar_docs:
                doc['source'] = 'comments'
            logger.debug(f"Retrieved {len(similar_docs)} similar comment documents")
            return similar_docs
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def retrieve_dictionary_context(
        self,
        query: str,
        top_k: int = None,
        model: str = 'xlmroberta',
    ) -> List[Dict]:
        """
        Retrieve dictionary chunks (JSON embedding index) + bổ sung từ DB nếu có.
        """
        top_k = top_k or self.dict_top_k
        results: List[Dict] = []

        try:
            json_hits = self.dict_store.search(
                query,
                top_k=top_k,
                model=model,
                threshold=self.dict_threshold,
            )
            results.extend(json_hits)
        except Exception as e:
            logger.error(f"Dictionary JSON retrieval error: {e}")

        try:
            query_embedding = self.embeddings.embed_text_xlmroberta(query)
            db_hits = self.db.search_similar_dictionary(
                embedding=query_embedding.tolist(),
                top_k=top_k,
            )
            seen_words = {str(r.get('word', '')).lower() for r in results}
            for row in db_hits:
                w = str(row.get('word', '')).strip()
                if not w or w.lower() in seen_words:
                    continue
                sim = float(row.get('similarity', 0.0))
                if sim < self.dict_threshold:
                    continue
                results.append({
                    'source': 'dictionary_db',
                    'chunk_id': f"db_{row.get('id')}",
                    'word': w,
                    'semantics': row.get('definition') or '',
                    'language': row.get('language'),
                    'text_content': f"{w}: {row.get('definition', '')}",
                    'similarity': sim,
                })
                seen_words.add(w.lower())
        except Exception as e:
            logger.debug(f"Dictionary DB retrieval skipped: {e}")

        results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        return results[:top_k]

    def retrieve_hybrid_context(
        self,
        query: str,
        top_k: int = None,
        dict_top_k: int = None,
        model: str = 'xlmroberta',
    ) -> Dict:
        """
        Kết hợp retrieve bình luận (vector DB) + chunk từ điển (JSON/DB).
        """
        top_k = top_k or self.top_k
        dict_top_k = dict_top_k or self.dict_top_k
        comment_docs = self.retrieve_context(query=query, top_k=top_k, model=model)
        dictionary_docs: List[Dict] = []
        if self.hybrid_retrieval:
            dictionary_docs = self.retrieve_dictionary_context(
                query=query, top_k=dict_top_k, model=model
            )
        return {
            'comment_documents': comment_docs,
            'dictionary_documents': dictionary_docs,
            'documents': comment_docs,
        }

    def retrieve_semantic_context(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve semantic context (sentiment, emotion, keywords)
        
        Args:
            query: Input query
            top_k: Number of results
            
        Returns:
            List of documents with semantic context
        """
        top_k = top_k or self.top_k
        
        try:
            # First retrieve similar documents
            similar_docs = self.retrieve_context(query, top_k)
            
            # For each document, retrieve its semantic metadata
            context_results = []
            for doc in similar_docs:
                doc_with_context = dict(doc)
                
                # Get semantic metadata from database
                db_conn = self.db.get_connection()
                with db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT sentiment_context, semantic_metadata 
                        FROM semantic_data 
                        WHERE embedding_id = %s;
                    """, (doc['id'],))
                    
                    semantic_row = cur.fetchone()
                    if semantic_row:
                        doc_with_context['sentiment_context'] = json.loads(semantic_row[0]) if semantic_row[0] else {}
                        doc_with_context['semantic_metadata'] = json.loads(semantic_row[1]) if semantic_row[1] else {}
                
                self.db.return_connection(db_conn)
                context_results.append(doc_with_context)
            
            return context_results
        except Exception as e:
            logger.error(f"Error retrieving semantic context: {e}")
            return []
    
    def retrieve_related_words(self, query: str, top_k: int = 5) -> List[Dict]:
        """Alias — dùng retrieve_dictionary_context (JSON + DB)."""
        return self.retrieve_dictionary_context(query=query, top_k=top_k, model="xlmroberta")
    
    def augment_prompt_with_context(self, query: str, context_docs: List[Dict],
                                   max_tokens: int = None) -> str:
        """
        Create augmented prompt with retrieved context
        
        Args:
            query: Original query
            context_docs: Retrieved context documents
            max_tokens: Maximum tokens in augmented prompt
            
        Returns:
            Augmented prompt string for LLM
        """
        max_tokens = max_tokens or RAG_CONFIG['max_context_tokens']
        
        if not context_docs:
            return query

        comment_docs = [d for d in context_docs if d.get('source') in (None, 'comments')]
        dict_docs = [d for d in context_docs if str(d.get('source', '')).startswith('dictionary')]

        context_text = "### Retrieved Context:\n\n"
        token_count = 0

        if comment_docs:
            context_text += "#### Bình luận tương tự\n\n"
            for i, doc in enumerate(comment_docs, 1):
                doc_text = f"[Comment {i}] (sim={doc.get('similarity', 0):.2f})\n"
                doc_text += f"Text: {doc.get('text_content', '')}\n"
                if doc.get('sentiment_label'):
                    doc_text += f"Sentiment: {doc.get('sentiment_label')}\n"
                if doc.get('sentiment_context'):
                    doc_text += f"Context: {json.dumps(doc.get('sentiment_context'), ensure_ascii=False)}\n"
                doc_text += "\n"
                token_count += len(doc_text.split())
                if token_count > max_tokens:
                    break
                context_text += doc_text

        if dict_docs and token_count <= max_tokens:
            context_text += "#### Từ điển liên quan\n\n"
            for i, doc in enumerate(dict_docs, 1):
                doc_text = f"[Dict {i}] (sim={doc.get('similarity', 0):.2f}) "
                doc_text += f"{doc.get('word', '')}: {doc.get('semantics', doc.get('text_content', ''))}\n\n"
                token_count += len(doc_text.split())
                if token_count > max_tokens:
                    break
                context_text += doc_text

        if not comment_docs and not dict_docs:
            for i, doc in enumerate(context_docs, 1):
                doc_text = f"[Document {i}] (Similarity: {doc.get('similarity', 0):.2f})\n"
                doc_text += f"Text: {doc.get('text_content', '')}\n"
                if doc.get('sentiment_label'):
                    doc_text += f"Sentiment: {doc.get('sentiment_label')}\n"
                doc_text += "\n"
                token_count += len(doc_text.split())
                if token_count > max_tokens:
                    break
                context_text += doc_text
        
        # Combine query with context
        augmented_prompt = f"""Query: {query}

{context_text}

Please analyze the query considering the retrieved context above."""
        
        return augmented_prompt
    
    def build_sentiment_context(self, similar_docs: List[Dict]) -> Dict:
        """
        Build sentiment analysis context from similar documents
        
        Args:
            similar_docs: Retrieved similar documents
            
        Returns:
            Dict with sentiment statistics and examples
        """
        sentiment_context = {
            'sentiment_distribution': {},
            'positive_examples': [],
            'negative_examples': [],
            'neutral_examples': [],
            'related_emotions': [],
            'common_themes': []
        }
        
        # Analyze sentiment distribution
        for doc in similar_docs:
            sentiment = doc.get('sentiment_label', 'unknown')
            sentiment_context['sentiment_distribution'][sentiment] = \
                sentiment_context['sentiment_distribution'].get(sentiment, 0) + 1
            
            # Collect examples by sentiment
            text = doc.get('text_content', '')
            if sentiment == 'praise':
                sentiment_context['positive_examples'].append(text)
            elif sentiment == 'criticism':
                sentiment_context['negative_examples'].append(text)
            else:
                sentiment_context['neutral_examples'].append(text)
        
        return sentiment_context
    
    def cache_query_results(self, query: str, results: Dict) -> None:
        """
        Cache query results for quick retrieval
        
        Args:
            query: Query string
            results: Query results to cache
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        try:
            self.db.execute_query("""
                INSERT INTO query_cache (query_hash, query_text, results)
                VALUES (%s, %s, %s)
                ON CONFLICT (query_hash) DO UPDATE 
                SET results = EXCLUDED.results
            """, (query_hash, query, json.dumps(results, ensure_ascii=False)))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def get_cached_results(self, query: str) -> Optional[Dict]:
        """
        Retrieve cached query results
        
        Args:
            query: Query string
            
        Returns:
            Cached results if available, None otherwise
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        try:
            results = self.db.execute_query(
                "SELECT results FROM query_cache WHERE query_hash = %s;",
                (query_hash,),
                fetch=True
            )
            
            if results:
                return json.loads(results[0]['results'])
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        try:
            self.db.execute_query("TRUNCATE TABLE query_cache;")
            logger.info("Query cache cleared")
        except Exception as e:
            logger.warning(f"Cache clearing failed: {e}")


# Global RAG retriever instance
_rag_retriever = None


def get_rag_retriever(top_k: int = None, 
                     similarity_threshold: float = None) -> RAGRetriever:
    """
    Get or create RAG retriever singleton
    
    Args:
        top_k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score
        
    Returns:
        RAGRetriever instance
    """
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever(top_k, similarity_threshold)
    return _rag_retriever
