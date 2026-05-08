# -*- coding: utf-8 -*-
"""
Main Entry Point - RAG + LLM Sentiment Analysis System
Interactive interface and batch processing
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LOGGING_CONFIG
from inference_pipeline import get_pipeline
from utils import create_sentiment_report, print_summary_statistics

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
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='RAG + LLM Vietnamese Sentiment Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive session
  python main.py --interactive
  
  # Analyze single comment
  python main.py "Bài hát rất hay!"
  
  # Batch analysis from file
  python main.py --batch data.txt --output results.json
  
  # Show system statistics
  python main.py --stats
        """
    )
    
    # Command arguments
    parser.add_argument(
        'comment',
        nargs='?',
        help='Comment to analyze'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run interactive session'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to input file with comments (one per line)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results.json',
        help='Output file for results (default: results.json)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of similar documents to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--llm',
        action='store_true',
        default=True,
        help='Use LLM for explanations (default: True)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_false',
        dest='llm',
        help='Disable LLM explanations'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics'
    )
    parser.add_argument(
        '--compare-encoders',
        type=str,
        help='Path to input text file for benchmarking tfidf/doc2vec/xlmroberta'
    )
    
    parser.add_argument(
        '--language', '-l',
        choices=['vi', 'en', 'auto'],
        default='auto',
        help='Language of input (default: auto)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("Initializing RAG + LLM pipeline...")
        pipeline = get_pipeline(use_llm=args.llm, auto_expand_dict=True)
        print("Pipeline ready\n")
        
        if args.compare_encoders:
            with open(args.compare_encoders, 'r', encoding='utf-8') as f:
                comments = [line.strip() for line in f if line.strip()]
            benchmark = pipeline.compare_encoders(comments)
            print(json.dumps(benchmark, ensure_ascii=True, indent=2))
            return True

        # Show statistics
        if args.stats:
            stats = pipeline.get_statistics()
            print("\n" + "="*60)
            print("SYSTEM STATISTICS")
            print("="*60)
            print(json.dumps(stats, indent=2, ensure_ascii=True))
            print("="*60 + "\n")
            return True
        
        # Interactive session
        if args.interactive:
            pipeline.interactive_session()
            return True
        
        # Batch processing
        if args.batch:
            print(f"Loading comments from {args.batch}...")
            try:
                with open(args.batch, 'r', encoding='utf-8') as f:
                    comments = [line.strip() for line in f if line.strip()]
                
                print(f"Processing {len(comments)} comments...\n")
                results = pipeline.batch_analyze(
                    comments,
                    return_explanations=args.llm,
                    top_k=args.top_k
                )
                
                # Save results
                output_data = pipeline.export_results(results, args.format)
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_data)
                
                print(f"\nResults saved to {output_path}")
                
                # Print summary
                print_summary_statistics(results)
                
                return True
            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                print(f"Error: {e}")
                return False
        
        # Single comment analysis
        if args.comment:
            result = pipeline.analyze_comment(
                args.comment,
                return_explanation=args.llm,
                top_k=args.top_k,
                language=args.language
            )
            
            print(create_sentiment_report(result))
            return True
        
        # If no specific action, show help and start interactive
        print("No specific action provided. Starting interactive session...")
        print("(Use --help for command options)\n")
        pipeline.interactive_session()
        return True
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return True
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nError: {e}")
        print("Check logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
