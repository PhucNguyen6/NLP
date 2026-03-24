# -*- coding: utf-8 -*-
"""
Sentiment Processing Module
Analyzes and assigns sentiment scores to comments
Imports sentiment words and modifiers from sentiment_word.py
"""

import csv
import os
import re
from langdetect import detect, DetectorFactory
from sentiment_word import (
    VI_POSITIVE_WORDS, VI_NEGATIVE_WORDS, VI_SENTIMENT_WORDS, VI_MODIFIERS,
    EN_POSITIVE_WORDS, EN_NEGATIVE_WORDS, EN_SENTIMENT_WORDS, EN_MODIFIERS
)

DetectorFactory.seed = 0

def get_vi_sentiment_score(words_list):
    score = 0
    modifier_multiplier = 1.0
    
    for word in words_list:
        word_lower = word.lower().strip()
        
        # Check if current word is a modifier
        if word_lower in VI_MODIFIERS:
            modifier_multiplier = VI_MODIFIERS[word_lower]
        
        # Check if current word is a sentiment word
        if word_lower in VI_SENTIMENT_WORDS:
            score += VI_SENTIMENT_WORDS[word_lower] * modifier_multiplier
            modifier_multiplier = 1.0  # Reset modifier after use
    return score

def get_en_sentiment_score(words_list):
    score = 0
    modifier_multiplier = 1.0
    
    for word in words_list:
        word_lower = word.lower().strip()
        
        # Check if current word is a modifier
        if word_lower in EN_MODIFIERS:
            modifier_multiplier = EN_MODIFIERS[word_lower]
        
        # Check if current word is a sentiment word
        if word_lower in EN_SENTIMENT_WORDS:
            score += EN_SENTIMENT_WORDS[word_lower] * modifier_multiplier
            modifier_multiplier = 1.0  # Reset modifier after use
    return score

def get_sentiment_score(words_list, language='vi'):
    if language.lower() == 'vi':
        return get_vi_sentiment_score(words_list)
    elif language.lower() == 'en':
        return get_en_sentiment_score(words_list)
    else:
        return 0.0

def get_sentiment_label_3class(score, vi_labels=True):
    if score > 0.5:
        return 'khen' if vi_labels else 'praise'
    elif score < -0.5:
        return 'chê' if vi_labels else 'criticism'
    else:
        return 'trung lập' if vi_labels else 'neutral'

def analyze_sentiment(text, language='en'):
    """
    Analyze sentiment of text
    Text is assumed to be already cleaned by preprocessing_comment.py
    
    Args:
        text: Input text (already cleaned)
        language: Language of text ('vi', 'en', or 'mix'). Default: 'en'
    
    Returns:
        dict: Analysis result with score and label
    """
    words = text.split()
    
    if language == 'vi':
        score = get_vi_sentiment_score(words)
    elif language == 'mix':
        vi_score = get_vi_sentiment_score(words)
        en_score = get_en_sentiment_score(words)
        score = vi_score + en_score  # Combine sentiment from both languages
    else:
        score = get_en_sentiment_score(words)
    
    # Get label (use English labels for consistency)
    label = get_sentiment_label_3class(score, vi_labels=False)
    
    return {
        'text': text,
        'score': score,
        'label': label,
        'language': language
    }


def process_csv_sentiment(input_csv, output_csv, comment_column='comment', lang_column='lang'):
    """
    Process CSV file and add sentiment labels
    Reads from clean_comment.csv and writes to labeled_comment.csv
    
    Data is already cleaned by preprocessing_comment.py
    Language is obtained from 'lang' column (en, vi, or mix)
    
    Args:
        input_csv: Path to input CSV file (clean_comment.csv)
        output_csv: Path to output CSV file (labeled_comment.csv)
        comment_column: Name of the column containing comments (default: 'comment')
        lang_column: Name of the column containing language (default: 'lang')
    
    Returns:
        dict: Statistics about processing
    """
    # Read CSV file
    comments_data = []
    stats = {
        'total': 0,
        'processed': 0,
        'praise': 0,
        'criticism': 0,
        'neutral': 0,
        'errors': 0
    }
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            field_names = reader.fieldnames
            
            if comment_column not in field_names:
                print(f"Error: Column '{comment_column}' not found in CSV")
                print(f"Available columns: {field_names}")
                return stats
            
            if lang_column not in field_names:
                print(f"Error: Column '{lang_column}' not found in CSV")
                print(f"Available columns: {field_names}")
                return stats
            
            for row in reader:
                stats['total'] += 1
                comment = row.get(comment_column, '').strip()
                lang = row.get(lang_column, 'en').strip()
                
                if comment:
                    try:
                        # Get language from data (already labeled by preprocessing)
                        # Validate language value
                        if lang not in ['vi', 'en', 'mix']:
                            lang = 'en'  # Default to English if invalid
                        
                        result = analyze_sentiment(comment, language=lang)
                        
                        # Add sentiment fields to row
                        row['sentiment_score'] = f"{result['score']:.2f}"
                        row['sentiment_label'] = result['label']
                        
                        # Count by label
                        if result['label'] == 'praise':
                            stats['praise'] += 1
                        elif result['label'] == 'criticism':
                            stats['criticism'] += 1
                        else:  # neutral
                            stats['neutral'] += 1
                        
                        stats['processed'] += 1
                        comments_data.append(row)
                    except Exception as e:
                        print(f"⚠️  Error processing row {stats['total']}: {e}")
                        stats['errors'] += 1
                else:
                    stats['errors'] += 1
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return stats
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write to output CSV
    if comments_data:
        try:
            # Prepare output field names (add sentiment columns)
            updated_field_names = list(field_names) + ['sentiment_score', 'sentiment_label']
            
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=updated_field_names, restval='')
                writer.writeheader()
                writer.writerows(comments_data)
            
            print("\n" + "="*70)
            print("SENTIMENT ANALYSIS PROCESSING COMPLETED")
            print("="*70)
            print(f"\nStatistics:")
            print(f"  Total records: {stats['total']}")
            print(f"  Successfully processed: {stats['processed']}")
            print(f"  Output saved to: {output_csv}")
            print(f"\nSentiment Distribution:")
            if stats['processed'] > 0:
                print(f"  KHEN (Praise):       {stats['praise']:4d} ({stats['praise']/stats['processed']*100:5.1f}%)")
                print(f"  CHÊ (Criticism):     {stats['criticism']:4d} ({stats['criticism']/stats['processed']*100:5.1f}%)")
                print(f"  TRUNG LẬP (Neutral): {stats['neutral']:4d} ({stats['neutral']/stats['processed']*100:5.1f}%)")
            print(f"\n   Skipped/Errors: {stats['errors']}")
            print("="*70 + "\n")
        
        except Exception as e:
            print(f"  Error writing CSV: {e}")
    
    return stats


# ===========================
# DEFAULT PATHS
# ===========================

DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__), "clean_data", "clean_comment.csv"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "labeled_data", "labeled_comment.csv"
)


# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS - LABEL ASSIGNMENT SYSTEM")
    print("="*70)
    print(f"\n Input:  {DEFAULT_INPUT}")
    print(f" Output: {DEFAULT_OUTPUT}")
    print("\n Processing comments with sentiment labels...")
    
    # Check if input file exists
    if os.path.exists(DEFAULT_INPUT):
        # Process CSV file
        stats = process_csv_sentiment(DEFAULT_INPUT, DEFAULT_OUTPUT)
        
        # Print summary
        if stats['processed'] > 0:
            print(f"  Processing completed successfully!")
        else:
            print(f"  No records were processed")
    else:
        print(f"  Error: Input file not found: {DEFAULT_INPUT}")
        print(f"\n   Please ensure clean_comment.csv exists in the clean_data directory")


