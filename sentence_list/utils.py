# -*- coding: utf-8 -*-
"""
Utility functions for RAG + LLM Sentiment Analysis System
Includes helpers for text processing, chunking, and data loading
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

import pandas as pd
import numpy as np

from config import LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                   format=LOGGING_CONFIG['format'])
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Words per chunk
        overlap: Word overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def clean_text(text: str, language: str = 'vi') -> str:
    """
    Basic text cleaning
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        Cleaned text
    """
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (keep space and alphanumeric)
    if language == 'en':
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-]', '', text)
    else:
        # For Vietnamese, keep more characters
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text, flags=re.UNICODE)
    
    return text


def tokenize_text(text: str, language: str = 'vi') -> List[str]:
    """
    Basic tokenization
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        List of tokens
    """
    
    # Simple whitespace tokenization
    tokens = text.lower().split()
    
    # Remove short tokens
    tokens = [t for t in tokens if len(t) > 1]
    
    return tokens


def calculate_embedding_statistics(embeddings: np.ndarray) -> Dict:
    """
    Calculate statistics for embeddings
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        Dictionary with statistics
    """
    
    return {
        'shape': embeddings.shape,
        'dtype': str(embeddings.dtype),
        'mean': float(np.mean(embeddings)),
        'std': float(np.std(embeddings)),
        'min': float(np.min(embeddings)),
        'max': float(np.max(embeddings)),
        'norm_mean': float(np.linalg.norm(embeddings, axis=1).mean())
    }


def load_csv_data(filepath: Path) -> pd.DataFrame:
    """
    Load CSV data with error handling
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Pandas DataFrame
    """
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


def save_json(data: Dict, filepath: Path) -> bool:
    """
    Save data as JSON with error handling
    
    Args:
        data: Data to save
        filepath: Output path
        
    Returns:
        True if successful
    """
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False


def load_json(filepath: Path) -> Optional[Dict]:
    """
    Load JSON data with error handling
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary or None
    """
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return None


def batch_process(items: List, batch_size: int = 32) -> List[List]:
    """
    Split items into batches
    
    Args:
        items: List of items
        batch_size: Batch size
        
    Returns:
        List of batches
    """
    
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches


def format_similarity_results(results: List[Dict], max_items: int = 5) -> str:
    """
    Format similarity search results for display
    
    Args:
        results: List of results with similarity scores
        max_items: Maximum items to display
        
    Returns:
        Formatted string
    """
    
    if not results:
        return "No results found."
    
    lines = ["Top Similar Documents:"]
    for i, result in enumerate(results[:max_items], 1):
        similarity = result.get('similarity', 0)
        text = result.get('text_content', '')[:100]
        sentiment = result.get('sentiment_label', 'unknown')
        
        lines.append(f"{i}. [{sentiment}] ({similarity:.2%} similar)")
        lines.append(f"   {text}...")
    
    return "\n".join(lines)


def merge_sentiment_results(results_list: List[Dict]) -> Dict:
    """
    Merge multiple sentiment analysis results
    
    Args:
        results_list: List of analysis results
        
    Returns:
        Merged result
    """
    
    if not results_list:
        return {}
    
    # Count sentiment labels
    label_counts = {}
    all_similarities = []
    
    for result in results_list:
        if 'sentiment_analysis' in result:
            label = result['sentiment_analysis'].get('predicted_label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if 'similar_documents' in result:
                for doc in result['similar_documents']:
                    all_similarities.append(doc.get('similarity', 0))
    
    # Calculate merged statistics
    merged = {
        'total_results': len(results_list),
        'label_distribution': label_counts,
        'avg_similarity': float(np.mean(all_similarities)) if all_similarities else 0,
        'max_similarity': float(np.max(all_similarities)) if all_similarities else 0,
        'min_similarity': float(np.min(all_similarities)) if all_similarities else 0,
    }
    
    return merged


def create_sentiment_report(results: Dict) -> str:
    """
    Create a human-readable sentiment analysis report
    
    Args:
        results: Analysis results
        
    Returns:
        Formatted report string
    """
    
    lines = []
    lines.append("\n" + "="*50)
    lines.append("SENTIMENT ANALYSIS REPORT")
    lines.append("="*50)
    
    if 'input' in results:
        lines.append(f"\nComment: {results['input'][:200]}")
    
    if 'sentiment_analysis' in results:
        sa = results['sentiment_analysis']
        lines.append(f"\nPredicted Sentiment: {sa.get('predicted_label', 'unknown').upper()}")
        lines.append(f"Confidence: {sa.get('confidence', 0):.1%}")
        lines.append(f"Similar Documents: {sa.get('num_similar_docs', 0)}")
        
        if 'label_distribution' in sa:
            lines.append("\nLabel Distribution:")
            for label, count in sa['label_distribution'].items():
                lines.append(f"  - {label}: {count}")
    
    if 'similar_documents' in results and results['similar_documents']:
        lines.append("\nTop Similar Examples:")
        for i, doc in enumerate(results['similar_documents'][:3], 1):
            lines.append(f"  {i}. [{doc['sentiment']}] {doc['text'][:80]}...")
    
    if 'llm_explanation' in results and results['llm_explanation']:
        lines.append(f"\nExplanation:\n{results['llm_explanation']}")
    
    lines.append("\n" + "="*50 + "\n")
    
    return "\n".join(lines)


def validate_embedding_shape(embedding: np.ndarray, expected_dim: int) -> bool:
    """
    Validate embedding dimension
    
    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension
        
    Returns:
        True if shape matches
    """
    
    if isinstance(embedding, (list, tuple)):
        return len(embedding) == expected_dim
    elif isinstance(embedding, np.ndarray):
        return embedding.shape[0] == expected_dim or embedding.shape == (expected_dim,)
    
    return False


def compute_batch_embeddings(texts: List[str], embedding_fn) -> np.ndarray:
    """
    Compute embeddings for multiple texts efficiently
    
    Args:
        texts: List of texts
        embedding_fn: Function that generates embedding for a single text
        
    Returns:
        Array of embeddings
    """
    
    embeddings = []
    
    for i, text in enumerate(texts):
        try:
            embedding = embedding_fn(text)
            embeddings.append(embedding)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
        except Exception as e:
            logger.warning(f"Error processing text {i}: {e}")
            embeddings.append(np.zeros(768))  # Default zero vector
    
    return np.array(embeddings)


def print_summary_statistics(results: List[Dict]) -> None:
    """
    Print summary statistics from batch results
    
    Args:
        results: List of analysis results
    """
    
    print("\n" + "="*50)
    print("BATCH ANALYSIS SUMMARY")
    print("="*50)
    
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    successful = total - errors
    
    print(f"Total processed: {total}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    
    # Sentiment distribution
    label_dist = {}
    for result in results:
        if 'sentiment_analysis' in result:
            label = result['sentiment_analysis'].get('predicted_label', 'unknown')
            label_dist[label] = label_dist.get(label, 0) + 1
    
    if label_dist:
        print("\nSentiment Distribution:")
        for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / successful * 100) if successful > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Average confidence
    confidences = [
        r['sentiment_analysis'].get('confidence', 0)
        for r in results if 'sentiment_analysis' in r
    ]
    if confidences:
        print(f"\nAverage Confidence: {np.mean(confidences):.1%}")
        print(f"Min Confidence: {np.min(confidences):.1%}")
        print(f"Max Confidence: {np.max(confidences):.1%}")
    
    print("="*50 + "\n")
