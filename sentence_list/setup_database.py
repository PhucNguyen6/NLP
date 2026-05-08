# -*- coding: utf-8 -*-
"""
Database Setup Script - Initialize PostgreSQL schema and tables
Must be run before using the RAG + LLM system
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_CONFIG, LOGGING_CONFIG
from database import get_db_manager

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


def main():
    """Main setup function"""
    
    print("\n" + "="*60)
    print("PostgreSQL Database Setup for RAG + LLM System")
    print("="*60)
    
    # Display database configuration
    print("\nDatabase Configuration:")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Port: {DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  User: {DB_CONFIG['user']}")
    
    # Confirm setup
    confirm = input("\nProceed with database setup? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Setup cancelled.")
        return False
    
    try:
        # Initialize database manager
        print("\nInitializing database manager...")
        db = get_db_manager()
        
        # Setup schema
        print("Creating database schema...")
        db.setup_schema()
        
        # Get statistics
        print("Database statistics:")
        stats = db.get_statistics()
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Sentiment labels: {stats['sentiment_labels']}")
        
        print("\n" + "="*60)
        print("Database setup completed successfully!")
        print("="*60 + "\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        print(f"\nSetup failed: {e}")
        print("Please check the error log for details.")
        return False
    
    finally:
        # Close connections
        try:
            db.close_all()
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
