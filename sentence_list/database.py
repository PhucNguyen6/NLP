# -*- coding: utf-8 -*-
"""
Database management module for PostgreSQL
Handles connection, schema creation, and CRUD operations
Supports pgvector for vector similarity search
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool

from config import DB_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGGING_CONFIG['log_file'])
    ]
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connection and operations"""
    
    def __init__(self, config: Dict = None, pool_size: int = 5):
        """
        Initialize database manager
        
        Args:
            config: Database configuration dict
            pool_size: Connection pool size
        """
        self.config = config or DB_CONFIG
        self.pool = None
        self.pool_size = pool_size
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = SimpleConnectionPool(
                1, self.pool_size,
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                connect_timeout=10
            )
            logger.info("Database connection pool initialized successfully")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        return self.pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        if self.pool:
            self.pool.putconn(conn)
    
    def close_all(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("All database connections closed")
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """
        Execute a query
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, else None
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch:
                    results = cur.fetchall()
                    conn.commit()
                    return results
                else:
                    conn.commit()
                    return None
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Query execution error: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def setup_schema(self):
        """Create database schema without pgvector"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Skip pgvector - not available on this system
                logger.info("Skipping pgvector (not available)")
                
                # Create embeddings table WITHOUT vector type (using FLOAT8[] array instead)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        text_id VARCHAR(255) NOT NULL UNIQUE,
                        text_content TEXT NOT NULL,
                        embedding_xlmroberta FLOAT8[],
                        embedding_doc2vec FLOAT8[],
                        embedding_tfidf FLOAT8[],
                        sentiment_label VARCHAR(50),
                        sentiment_score FLOAT,
                        language VARCHAR(20),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON embeddings(sentiment_label);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_language ON embeddings(language);")
                logger.info("Created embeddings table")
                
                # Note: pgvector not available, using BYTEA for embeddings instead
                
                # Create dictionary table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dictionary (
                        id SERIAL PRIMARY KEY,
                        word VARCHAR(255) NOT NULL UNIQUE,
                        definition TEXT,
                        language VARCHAR(20),
                        part_of_speech VARCHAR(50),
                        embedding FLOAT8[],
                        sentiment_score FLOAT,
                        source VARCHAR(255),
                        examples JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for dictionary table
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_language ON dictionary(language);")
                
                # Create semantic data table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_data (
                        id SERIAL PRIMARY KEY,
                        chunk_id VARCHAR(255) NOT NULL UNIQUE,
                        embedding_id INT REFERENCES embeddings(id),
                        chunk_text TEXT NOT NULL,
                        embedding FLOAT8[],
                        sentiment_context JSONB,
                        semantic_metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for semantic_data table
                cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_embedding_id ON semantic_data(embedding_id);")
                
                # Create sentiment metadata table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_metadata (
                        id SERIAL PRIMARY KEY,
                        embedding_id INT REFERENCES embeddings(id),
                        emotion_keywords JSONB,
                        emotion_scores JSONB,
                        context_info JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for sentiment_metadata table
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_embedding_id ON sentiment_metadata(embedding_id);")
                
                # Create query cache table for performance
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        id SERIAL PRIMARY KEY,
                        query_hash VARCHAR(255) NOT NULL UNIQUE,
                        query_text TEXT,
                        results JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for query_cache table
                cur.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON query_cache(query_hash);")
                
            conn.commit()
            logger.info("Database schema created successfully")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Schema creation error: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def insert_embedding(self, text_id: str, text: str, embeddings: Dict[str, List[float]],
                        sentiment_label: str = None, language: str = None, 
                        metadata: Dict = None) -> int:
        """
        Insert embedding record
        
        Args:
            text_id: Unique text identifier
            text: Text content
            embeddings: Dict with keys 'xlmroberta', 'doc2vec', 'tfidf'
            sentiment_label: Sentiment classification
            language: Language code
            metadata: Additional metadata
            
        Returns:
            Record ID
        """
        query = """
            INSERT INTO embeddings 
            (text_id, text_content, embedding_xlmroberta, embedding_doc2vec, 
             embedding_tfidf, sentiment_label, language, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        params = (
            text_id,
            text,
            embeddings.get('xlmroberta'),
            embeddings.get('doc2vec'),
            embeddings.get('tfidf'),
            sentiment_label,
            language,
            json.dumps(metadata) if metadata else None
        )
        
        result = self.execute_query(query, params, fetch=True)
        return result[0]['id'] if result else None
    
    def insert_dictionary_entry(self, word: str, definition: str = None,
                               language: str = None, embedding: List[float] = None,
                               sentiment_score: float = None, source: str = None) -> int:
        """
        Insert dictionary entry
        
        Args:
            word: Word to add
            definition: Word definition
            language: Language code
            embedding: Word embedding vector
            sentiment_score: Sentiment score
            source: Source of definition
            
        Returns:
            Record ID
        """
        query = """
            INSERT INTO dictionary 
            (word, definition, language, embedding, sentiment_score, source)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (word) DO UPDATE SET 
                definition = EXCLUDED.definition,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """
        
        params = (word, definition, language, embedding, sentiment_score, source)
        result = self.execute_query(query, params, fetch=True)
        return result[0]['id'] if result else None
    
    def search_similar_embeddings(self, embedding: List[float], 
                                 top_k: int = 5, threshold: float = 0.3,
                                 model: str = 'xlmroberta') -> List[Dict]:
        """
        Search for similar embeddings using array similarity
        Note: pgvector not available, computing similarity in Python
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            model: Embedding model to use
            
        Returns:
            List of similar records with similarity scores
        """
        if model == 'xlmroberta':
            embedding_col = 'embedding_xlmroberta'
        elif model == 'doc2vec':
            embedding_col = 'embedding_doc2vec'
        else:
            embedding_col = 'embedding_tfidf'
        
        # Get all embeddings (pgvector not available, compute in Python)
        query = f"""
            SELECT 
                id, text_id, text_content, sentiment_label, language,
                {embedding_col} as embedding_array
            FROM embeddings
            WHERE {embedding_col} IS NOT NULL
            LIMIT 10000;
        """
        
        results = self.execute_query(query, fetch=True)
        
        if not results:
            return []
        
        # Compute cosine similarity in Python
        import numpy as np
        query_vec = np.array(embedding, dtype=np.float32)
        
        similarities = []
        for row in results:
            try:
                emb_array = np.array(row['embedding_array'], dtype=np.float32)
                # Cosine similarity
                sim = np.dot(query_vec, emb_array) / (np.linalg.norm(query_vec) * np.linalg.norm(emb_array) + 1e-10)
                
                if sim >= threshold:
                    similarities.append({
                        'id': row['id'],
                        'text_id': row['text_id'],
                        'text_content': row['text_content'],
                        'sentiment_label': row['sentiment_label'],
                        'language': row['language'],
                        'similarity': float(sim)
                    })
            except:
                continue
        
        # Sort by similarity and limit
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def search_similar_dictionary(self, embedding: List[float], 
                                 top_k: int = 5) -> List[Dict]:
        """
        Search similar words in dictionary by embedding
        Compute in Python since pgvector not available
        
        Args:
            embedding: Query embedding
            top_k: Number of results
            
        Returns:
            Similar dictionary entries
        """
        query = """
            SELECT 
                id, word, definition, language, sentiment_score,
                embedding as embedding_array
            FROM dictionary
            WHERE embedding IS NOT NULL
            LIMIT 1000;
        """
        
        results = self.execute_query(query, fetch=True)
        
        if not results:
            return []
        
        # Compute similarity in Python
        import numpy as np
        query_vec = np.array(embedding, dtype=np.float32)
        
        similarities = []
        for row in results:
            try:
                emb_array = np.array(row['embedding_array'], dtype=np.float32)
                # Cosine similarity
                sim = np.dot(query_vec, emb_array) / (np.linalg.norm(query_vec) * np.linalg.norm(emb_array) + 1e-10)
                
                similarities.append({
                    'id': row['id'],
                    'word': row['word'],
                    'definition': row['definition'],
                    'language': row['language'],
                    'sentiment_score': row['sentiment_score'],
                    'similarity': float(sim)
                })
            except:
                continue
        
        # Sort by similarity and limit
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_word_definition(self, word: str) -> Optional[Dict]:
        """Get definition for a word"""
        query = "SELECT * FROM dictionary WHERE word = %s;"
        results = self.execute_query(query, (word,), fetch=True)
        return results[0] if results else None
    
    def batch_insert_embeddings(self, records: List[Tuple]) -> int:
        """
        Batch insert embeddings for performance
        
        Args:
            records: List of tuples with embedding data
            
        Returns:
            Number of records inserted
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO embeddings 
                    (text_id, text_content, embedding_xlmroberta, 
                     embedding_doc2vec, embedding_tfidf, sentiment_label, 
                     language, metadata)
                    VALUES %s
                    ON CONFLICT (text_id) DO NOTHING;
                """
                result = execute_values(cur, query, records, page_size=1000)
                conn.commit()
                logger.info(f"Batch inserted {len(records)} embeddings")
                return len(records)
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Batch insert error: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        query = "SELECT COUNT(*) as count FROM embeddings;"
        stats['total_embeddings'] = self.execute_query(query, fetch=True)[0]['count']
        
        query = "SELECT COUNT(*) as count FROM dictionary;"
        stats['total_words'] = self.execute_query(query, fetch=True)[0]['count']
        
        query = "SELECT COUNT(DISTINCT sentiment_label) as count FROM embeddings WHERE sentiment_label IS NOT NULL;"
        stats['sentiment_labels'] = self.execute_query(query, fetch=True)[0]['count']
        
        query = """
            SELECT sentiment_label, COUNT(*) as count 
            FROM embeddings 
            WHERE sentiment_label IS NOT NULL 
            GROUP BY sentiment_label;
        """
        stats['label_distribution'] = {
            row['sentiment_label']: row['count'] 
            for row in self.execute_query(query, fetch=True)
        }
        
        return stats


# Global database manager instance
_db_manager = None


def get_db_manager(config: Dict = None, pool_size: int = 5) -> DatabaseManager:
    """
    Get or create database manager singleton
    
    Args:
        config: Database configuration
        pool_size: Connection pool size
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config, pool_size)
    return _db_manager
