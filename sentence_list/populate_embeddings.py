# -*- coding: utf-8 -*-
"""
Populate Embeddings - Load pre-computed embeddings into PostgreSQL database
Transfers existing encoded data to the database for RAG retrieval
"""

import sys
import logging
import pickle
import zlib
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import ENCODED_DIR, LOGGING_CONFIG
from database import get_db_manager
from embeddings_manager import get_embeddings_manager

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


def load_encoded_data(model_name: str):
    """
    Load pre-computed encoded data (handles zlib-compressed pickle files)
    
    Args:
        model_name: Name of model ('tfidf', 'doc2vec', 'xlmroberta')
        
    Returns:
        Dictionary with 'X' and 'y' arrays
    """
    model_path = ENCODED_DIR / f'{model_name}_labeled.pkl'
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            compressed_data = f.read()
        
        # Try to decompress with zlib first
        try:
            decompressed_data = zlib.decompress(compressed_data)
            data = pickle.loads(decompressed_data)
        except zlib.error:
            # If not compressed, try loading directly as pickle
            data = pickle.loads(compressed_data)
        
        logger.info(f"Loaded {model_name}: shape={data['X'].shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return None


def populate_embeddings_from_file(model_name: str = 'xlmroberta'):
    """
    Populate embeddings from saved pickle file
    
    Args:
        model_name: Model to load
        
    Returns:
        Number of records inserted
    """
    
    print(f"\nLoading {model_name} embeddings...")
    data = load_encoded_data(model_name)
    
    if data is None:
        return 0
    
    db = get_db_manager()
    X = data['X']  # Embeddings array
    y = data['y']  # Labels array
    
    total_inserted = 0
    batch_size = 100
    
    print(f"Preparing to insert {len(X)} records...")
    
    # Batch processing
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        
        records = []
        for j, (embedding, label) in enumerate(zip(batch_X, batch_y)):
            text_id = f"{model_name}_{i + j}"
            
            # Create embeddings dict based on model
            embeddings_dict = {}
            if model_name == 'xlmroberta':
                embeddings_dict['xlmroberta'] = embedding.tolist()
            elif model_name == 'doc2vec':
                embeddings_dict['doc2vec'] = embedding.tolist()
            elif model_name == 'tfidf':
                embeddings_dict['tfidf'] = embedding.tolist()
            
            record = (
                text_id,
                f"Encoded {model_name} #{i + j}",  # Placeholder text
                embeddings_dict.get('xlmroberta'),
                embeddings_dict.get('doc2vec'),
                embeddings_dict.get('tfidf'),
                label,
                'vi',  # Language
                '{"source": "encoded_data"}'  # Metadata
            )
            records.append(record)
        
        try:
            inserted = db.batch_insert_embeddings(records)
            total_inserted += inserted
        except Exception as e:
            logger.error(f"Error inserting batch {i//batch_size}: {e}")
    
    logger.info(f"Inserted {total_inserted} records for {model_name}")
    return total_inserted


def populate_embeddings_from_clean_data():
    """
    Generate and populate embeddings from clean comment data
    
    Returns:
        Number of records inserted
    """
    
    from config import DATA_DIR
    import pandas as pd
    
    data_file = DATA_DIR / 'clean_comment.csv'
    if not data_file.exists():
        logger.warning(f"Data file not found: {data_file}")
        return 0
    
    print(f"\nLoading clean comments from {data_file}...")
    df = pd.read_csv(data_file, encoding='utf-8')
    logger.info(f"Loaded {len(df)} comments")
    
    db = get_db_manager()
    embeddings_manager = get_embeddings_manager()
    
    total_inserted = 0
    batch_size = 32
    
    print(f"Generating embeddings for {len(df)} comments...")
    
    # Generate embeddings in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        texts = batch_df['text'].tolist() if 'text' in batch_df.columns else batch_df.iloc[:, 0].tolist()
        
        # Generate embeddings
        try:
            embeddings_list = embeddings_manager.embed_batch_xlmroberta(texts)
            
            records = []
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                text_id = f"comment_{i + j}"
                text_content = row.get('text', '') if isinstance(row, dict) else row.iloc[0]
                
                embedding_xlmroberta = embeddings_list[j].tolist() if j < len(embeddings_list) else None
                sentiment_label = row.get('sentiment', 'neutral') if isinstance(row, dict) else 'neutral'
                
                record = (
                    text_id,
                    str(text_content)[:1000],
                    embedding_xlmroberta,
                    None,  # doc2vec
                    None,  # tfidf
                    sentiment_label,
                    'vi',
                    '{"source": "clean_comments"}'
                )
                records.append(record)
            
            inserted = db.batch_insert_embeddings(records)
            total_inserted += inserted
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + batch_size}/{len(df)} comments")
        
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {e}")
    
    logger.info(f"Inserted {total_inserted} records from clean comments")
    return total_inserted


def main():
    """Main population function"""
    
    print("\n" + "="*60)
    print("Populate Embeddings - RAG + LLM System")
    print("="*60)
    
    try:
        db = get_db_manager()
        
        # Check if embeddings already exist
        stats = db.get_statistics()
        existing_count = stats.get('total_embeddings', 0)
        
        if existing_count > 0:
            print(f"\n✓ Database already contains {existing_count} embeddings")
            print("Skipping generation. Using existing embeddings for RAG.")
            total_inserted = 0
        else:
            total_inserted = 0
            
            # Populate from pre-computed models
            print("\n1. Loading pre-computed embeddings...")
            for model in ['tfidf', 'doc2vec', 'xlmroberta']:
                count = populate_embeddings_from_file(model)
                total_inserted += count
            
            # Populate from clean comments (generate embeddings)
            if total_inserted == 0:
                print("\n2. Generating embeddings from clean comments...")
                count = populate_embeddings_from_clean_data()
                total_inserted += count
        
        # Show statistics
        print("\n" + "-"*60)
        print("Database Statistics:")
        stats = db.get_statistics()
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Sentiment distribution: {stats['label_distribution']}")
        
        print("\n" + "="*60)
        print(f"✓ Populated {total_inserted} records successfully!")
        print("="*60 + "\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Population failed: {e}")
        print(f"\n✗ Populate failed: {e}")
        return False
    
    finally:
        try:
            db.close_all()
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
