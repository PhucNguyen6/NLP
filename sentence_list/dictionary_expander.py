# -*- coding: utf-8 -*-
"""
Dictionary Expander - Auto-crawl and expand dictionary with new words
Handles web scraping of word definitions and sentiment analysis
"""

import logging
import re
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio

import requests
from bs4 import BeautifulSoup

from config import DICT_EXPANSION_CONFIG, LOGGING_CONFIG, DICT_DIR
from database import get_db_manager
from embeddings_manager import get_embeddings_manager
from sentiment_define import get_vi_sentiment_score, get_en_sentiment_score

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                   format=LOGGING_CONFIG['format'])
logger = logging.getLogger(__name__)


class DictionaryExpander:
    """Expands dictionary by crawling web sources and analyzing sentiment"""
    
    def __init__(self):
        """Initialize dictionary expander"""
        self.db = get_db_manager()
        self.embeddings = get_embeddings_manager()
        self.crawl_sources = DICT_EXPANSION_CONFIG['crawl_sources']
        self.vietnamese_sources = DICT_EXPANSION_CONFIG['vietnamese_sources']
        self.timeout = DICT_EXPANSION_CONFIG['crawl_timeout']
        DICT_DIR.mkdir(parents=True, exist_ok=True)
        self.local_dict_path = DICT_DIR / 'vi_dict.csv'
        self.en_dict_path = DICT_DIR / 'en_dict.csv'
        
    def check_word_exists(self, word: str) -> bool:
        """
        Check if word already exists in database
        
        Args:
            word: Word to check
            
        Returns:
            True if word exists
        """
        try:
            result = self.db.get_word_definition(word.lower())
            return result is not None
        except Exception as e:
            logger.warning(f"Error checking word: {e}")
            return False
    
    def crawl_english_definition(self, word: str) -> Optional[Dict]:
        """
        Crawl English definition from web sources
        
        Args:
            word: Word to search
            
        Returns:
            Dictionary entry with definition or None
        """
        
        for source in self.crawl_sources:
            try:
                definition = self._crawl_source(word, source)
                if definition:
                    return {
                        'word': word,
                        'definition': definition,
                        'language': 'en',
                        'source': source,
                        'crawled_at': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Error crawling {source} for '{word}': {e}")
        
        return None
    
    def _crawl_source(self, word: str, source: str) -> Optional[str]:
        """
        Crawl definition from specific source
        
        Args:
            word: Word to search
            source: Source URL
            
        Returns:
            Definition text or None
        """
        try:
            if 'cambridge' in source:
                return self._crawl_cambridge(word)
            elif 'merriam' in source:
                return self._crawl_merriam(word)
            elif 'dictionary.com' in source:
                return self._crawl_dictionary_com(word)
        except Exception as e:
            logger.warning(f"Error crawling {source}: {e}")
        
        return None
    
    def _crawl_cambridge(self, word: str) -> Optional[str]:
        """
        Crawl definition from Cambridge Dictionary
        
        Args:
            word: Word to search
            
        Returns:
            Definition or None
        """
        try:
            url = f"https://dictionary.cambridge.org/us/dictionary/english/{word}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find definition
                def_block = soup.find('div', class_='def-block')
                if def_block:
                    definition = def_block.find('div', class_='ddef_d')
                    if definition:
                        return definition.get_text(strip=True)
        except Exception as e:
            logger.debug(f"Cambridge crawl error: {e}")
        
        return None
    
    def _crawl_merriam(self, word: str) -> Optional[str]:
        """
        Crawl definition from Merriam-Webster
        
        Args:
            word: Word to search
            
        Returns:
            Definition or None
        """
        try:
            url = f"https://www.merriam-webster.com/dictionary/{word}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find definition
                def_text = soup.find('span', class_='dtText')
                if def_text:
                    return def_text.get_text(strip=True)
        except Exception as e:
            logger.debug(f"Merriam crawl error: {e}")
        
        return None
    
    def _crawl_dictionary_com(self, word: str) -> Optional[str]:
        """
        Crawl definition from Dictionary.com
        
        Args:
            word: Word to search
            
        Returns:
            Definition or None
        """
        try:
            url = f"https://www.dictionary.com/browse/{word}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find definition section
                def_section = soup.find('div', class_='css-1v2eamc')
                if def_section:
                    def_text = def_section.find('p')
                    if def_text:
                        return def_text.get_text(strip=True)
        except Exception as e:
            logger.debug(f"Dictionary.com crawl error: {e}")
        
        return None
    
    def crawl_vietnamese_definition(self, word: str) -> Optional[Dict]:
        """
        Crawl Vietnamese definition from local sources or web
        
        Args:
            word: Vietnamese word
            
        Returns:
            Dictionary entry with definition
        """
        
        # First check local dictionary
        definition = self._check_local_vi_dict(word)
        if definition:
            return {
                'word': word,
                'definition': definition,
                'language': 'vi',
                'source': 'local_dict',
                'crawled_at': datetime.now().isoformat()
            }
        
        # Try web sources
        for source in self.vietnamese_sources:
            try:
                definition = self._crawl_vi_source(word, source)
                if definition:
                    return {
                        'word': word,
                        'definition': definition,
                        'language': 'vi',
                        'source': source,
                        'crawled_at': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.debug(f"Vietnamese crawl error: {e}")
        
        return None
    
    def _check_local_vi_dict(self, word: str) -> Optional[str]:
        """
        Check local Vietnamese dictionary file
        
        Args:
            word: Word to search
            
        Returns:
            Definition or None
        """
        try:
            if self.local_dict_path.exists():
                import csv
                with open(self.local_dict_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if str(row.get('word', '')).strip().lower() == word.lower():
                            return str(row.get('definition') or row.get('meaning') or '').strip() or None
        except Exception as e:
            logger.debug(f"Local dict search error: {e}")
        
        return None
    
    def _crawl_vi_source(self, word: str, source: str) -> Optional[str]:
        """
        Crawl Vietnamese definition from web source
        
        Args:
            word: Word to search
            source: Source URL
            
        Returns:
            Definition or None
        """
        # This is a placeholder - Vietnamese sources have different structures
        try:
            response = requests.get(f"{source}?q={word}", timeout=self.timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Parse based on source structure
                # (Implementation depends on specific source HTML structure)
                pass
        except Exception as e:
            logger.debug(f"Vietnamese source crawl error: {e}")
        
        return None
    
    def analyze_word_sentiment(self, word: str, language: str = 'en') -> float:
        """
        Analyze sentiment score of a word
        
        Args:
            word: Word to analyze
            language: Language ('en' or 'vi')
            
        Returns:
            Sentiment score
        """
        try:
            if language == 'vi':
                score = get_vi_sentiment_score([word])
            else:
                score = get_en_sentiment_score([word])
            
            return float(score)
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for '{word}': {e}")
            return 0.0
    
    def add_word_to_database(self, word: str, definition: str = None,
                            language: str = 'en', source: str = None,
                            auto_sentiment: bool = True) -> bool:
        """
        Add a new word to database with definition and embedding
        
        Args:
            word: Word to add
            definition: Definition
            language: Language code
            source: Source of definition
            auto_sentiment: Whether to analyze sentiment automatically
            
        Returns:
            True if successful
        """
        try:
            # Generate embedding
            embedding = self.embeddings.embed_text_xlmroberta(word)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            # Analyze sentiment
            sentiment_score = None
            if auto_sentiment:
                sentiment_score = self.analyze_word_sentiment(word, language)
            
            # Insert into database
            word_id = self.db.insert_dictionary_entry(
                word=word.lower(),
                definition=definition,
                language=language,
                embedding=embedding_list,
                sentiment_score=sentiment_score,
                source=source
            )
            
            logger.info(f"Added word '{word}' to database (ID: {word_id})")
            return True
        
        except Exception as e:
            logger.error(f"Error adding word to database: {e}")
            return False
    
    def expand_from_text(self, text: str, language: str = 'en',
                        batch_crawl: bool = True) -> int:
        """
        Extract words from text and add unknown words to dictionary
        
        Args:
            text: Input text
            language: Language of text
            batch_crawl: Whether to batch process unknown words
            
        Returns:
            Number of new words added
        """
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower()) if language == 'en' else text.split()
        unique_words = set(words)
        
        new_words_count = 0
        unknown_words = []
        
        # Check which words are unknown
        for word in unique_words:
            if len(word) > 2 and not self.check_word_exists(word):
                unknown_words.append(word)
        
        logger.info(f"Found {len(unknown_words)} unknown words")
        
        # Batch crawl and add
        if batch_crawl and unknown_words:
            batch_size = DICT_EXPANSION_CONFIG['batch_crawl_size']
            for i in range(0, len(unknown_words), batch_size):
                batch = unknown_words[i:i + batch_size]
                
                for word in batch:
                    try:
                        if language == 'en':
                            entry = self.crawl_english_definition(word)
                        else:
                            entry = self.crawl_vietnamese_definition(word)
                        
                        if entry:
                            if self.add_word_to_database(
                                word=entry['word'],
                                definition=entry['definition'],
                                language=entry['language'],
                                source=entry.get('source')
                            ):
                                new_words_count += 1
                    except Exception as e:
                        logger.warning(f"Error processing word '{word}': {e}")
        
        logger.info(f"Added {new_words_count} new words to dictionary")
        return new_words_count
    
    def get_dictionary_stats(self) -> Dict:
        """
        Get dictionary statistics
        
        Returns:
            Dictionary stats
        """
        stats = self.db.get_statistics()
        return {
            'total_words': stats.get('total_words', 0),
            'total_embeddings': stats.get('total_embeddings', 0),
            'sentiment_labels': stats.get('sentiment_labels', 0),
            'label_distribution': stats.get('label_distribution', {})
        }


# Global dictionary expander instance
_dict_expander = None


def get_dictionary_expander() -> DictionaryExpander:
    """
    Get or create dictionary expander singleton
    
    Returns:
        DictionaryExpander instance
    """
    global _dict_expander
    if _dict_expander is None:
        _dict_expander = DictionaryExpander()
    return _dict_expander
