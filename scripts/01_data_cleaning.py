"""
Data Cleaning Pipeline for Trump Speech Transcripts

This script:
1. Loads raw scraped transcripts
2. Removes HTML tags, metadata, timestamps
3. Removes crowd reaction tags (applause, cheers, etc.)
4. Standardizes speaker tags
5. Normalizes whitespace and punctuation
6. Removes duplicates and noise
7. Validates UTF-8 encoding
8. Saves cleaned data
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json, save_csv,
    clean_whitespace, normalize_quotes, remove_html_tags,
    validate_utf8, create_speech_id, print_section, print_stats
)


class TranscriptCleaner:
    """Clean and normalize Trump speech transcripts"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize cleaner with configuration"""
        self.config = load_config(config_path)
        self.cleaning_config = self.config['cleaning']
        self.reaction_tags = self.cleaning_config['reaction_tags']
        self.noise_tokens = self.cleaning_config['noise_tokens']
        self.speaker_patterns = self.cleaning_config['speaker_patterns']
        
    def remove_reaction_tags(self, text: str) -> str:
        """Remove crowd reaction tags like (applause), [cheers], etc."""
        for tag in self.reaction_tags:
            # Case-insensitive removal
            text = re.sub(re.escape(tag), '', text, flags=re.IGNORECASE)
        
        # Also remove any remaining parenthetical expressions that are just reactions
        text = re.sub(r'\([^)]*(?:applause|cheer|laugh|inaudible|crosstalk)[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[[^\]]*(?:applause|cheer|laugh|inaudible|crosstalk)[^\]]*\]', '', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_timestamps(self, text: str) -> str:
        """Remove timestamp patterns like [00:12:34] or (12:34)"""
        # Remove [HH:MM:SS] or [MM:SS]
        text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text)
        # Remove (HH:MM:SS) or (MM:SS)
        text = re.sub(r'\(\d{1,2}:\d{2}(?::\d{2})?\)', '', text)
        return text
    
    def standardize_speakers(self, text: str) -> str:
        """Standardize speaker tags to SPEAKER_TRUMP"""
        for pattern, replacement in self.speaker_patterns.items():
            # Match pattern followed by colon (speaker tag format)
            text = re.sub(f"{pattern}:", f"{replacement}:", text, flags=re.IGNORECASE)
        
        return text
    
    def remove_noise_tokens(self, text: str) -> str:
        """Remove filler words like 'uh', 'um', 'you know'"""
        for token in self.noise_tokens:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(token) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation and quotes"""
        # Normalize quotes
        text = normalize_quotes(text)
        
        # Fix multiple punctuation marks
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamation marks
        
        # Ensure space after sentence terminators
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text
    
    def remove_duplicates(self, text: str) -> str:
        """Remove duplicate consecutive paragraphs"""
        paragraphs = text.split('\n\n')
        
        # Remove consecutive duplicates
        cleaned_paragraphs = []
        prev_para = None
        
        for para in paragraphs:
            para = para.strip()
            if para and para != prev_para:
                cleaned_paragraphs.append(para)
                prev_para = para
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def remove_metadata_headers(self, text: str) -> str:
        """Remove common metadata headers from transcripts"""
        # Remove "Transcript" header lines
        text = re.sub(r'^Transcript\s*\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove "Rev.com" references
        text = re.sub(r'Rev\.com\s*', '', text, flags=re.IGNORECASE)
        
        # Remove copyright notices
        text = re.sub(r'©.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Copyright.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        
        return text
    
    def clean_transcript(self, text: str) -> str:
        """Apply all cleaning steps to a transcript"""
        if not text:
            return ""
        
        # Step 1: Remove HTML tags
        text = remove_html_tags(text)
        
        # Step 2: Remove metadata headers
        text = self.remove_metadata_headers(text)
        
        # Step 3: Remove timestamps
        text = self.remove_timestamps(text)
        
        # Step 4: Remove reaction tags
        text = self.remove_reaction_tags(text)
        
        # Step 5: Standardize speaker tags
        text = self.standardize_speakers(text)
        
        # Step 6: Remove noise tokens
        text = self.remove_noise_tokens(text)
        
        # Step 7: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Step 8: Normalize whitespace
        text = clean_whitespace(text)
        
        # Step 9: Remove duplicates
        text = self.remove_duplicates(text)
        
        # Step 10: Final whitespace cleanup
        text = clean_whitespace(text)
        
        return text
    
    def process_speeches(self, input_file: str) -> List[Dict[str, Any]]:
        """Process all speeches from input file"""
        print_section("DATA CLEANING PIPELINE")
        
        # Load data
        print(f"\nLoading data from {input_file}...")
        
        if input_file.endswith('.json'):
            raw_data = load_json(input_file)
        elif input_file.endswith('.csv'):
            df = pd.read_csv(input_file, encoding='utf-8')
            raw_data = df.to_dict('records')
        else:
            print("Error: Unsupported file format. Use .json or .csv")
            return []
        
        if not raw_data:
            print("Error: No data loaded")
            return []
        
        print(f"✓ Loaded {len(raw_data)} speeches")
        
        # Clean each speech
        print("\nCleaning transcripts...")
        cleaned_speeches = []
        
        for i, speech in enumerate(raw_data, 1):
            print(f"  [{i}/{len(raw_data)}] {speech.get('title', 'Untitled')[:50]}...")
            
            # Extract fields
            raw_text = speech.get('transcript', '')
            title = speech.get('title', '')
            date = speech.get('date', '')
            url = speech.get('url', '')
            
            # Clean the transcript
            cleaned_text = self.clean_transcript(raw_text)
            
            # Validate UTF-8
            if not validate_utf8(cleaned_text):
                print(f"    Warning: UTF-8 validation failed for speech {i}")
                # Try to fix encoding issues
                cleaned_text = cleaned_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Create speech ID
            speech_id = create_speech_id(title, date)
            
            # Calculate basic stats
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            
            # Store cleaned speech
            cleaned_speech = {
                'speech_id': speech_id,
                'title': title,
                'date': date,
                'url': url,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'word_count': word_count,
                'char_count': char_count,
                'cleaned_at': datetime.now().isoformat()
            }
            
            cleaned_speeches.append(cleaned_speech)
            
            # Print stats
            reduction = ((len(raw_text) - len(cleaned_text)) / len(raw_text) * 100) if raw_text else 0
            print(f"    Words: {word_count:,} | Chars: {char_count:,} | Reduction: {reduction:.1f}%")
        
        return cleaned_speeches
    
    def save_cleaned_data(self, cleaned_speeches: List[Dict[str, Any]]):
        """Save cleaned speeches to JSON and CSV"""
        print_section("SAVING CLEANED DATA")
        
        output_dir = Path(self.config['paths']['cleaned_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON (full data with both raw and cleaned)
        json_file = output_dir / f"speeches_cleaned_{timestamp}.json"
        save_json(cleaned_speeches, str(json_file))
        
        # Save CSV (cleaned text only for easy viewing)
        csv_data = []
        for speech in cleaned_speeches:
            csv_data.append({
                'speech_id': speech['speech_id'],
                'title': speech['title'],
                'date': speech['date'],
                'url': speech['url'],
                'word_count': speech['word_count'],
                'cleaned_text': speech['cleaned_text']
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = output_dir / f"speeches_cleaned_{timestamp}.csv"
        save_csv(df, str(csv_file))
        
        # Save summary statistics
        self.print_summary(cleaned_speeches)
        
        return str(json_file), str(csv_file)
    
    def print_summary(self, cleaned_speeches: List[Dict[str, Any]]):
        """Print summary statistics"""
        print_section("SUMMARY STATISTICS")
        
        total_speeches = len(cleaned_speeches)
        total_words = sum(s['word_count'] for s in cleaned_speeches)
        avg_words = total_words / total_speeches if total_speeches > 0 else 0
        
        print_stats("Total speeches cleaned", total_speeches)
        print_stats("Total words", f"{total_words:,}")
        print_stats("Average words per speech", f"{avg_words:,.0f}")
        
        if cleaned_speeches:
            longest = max(cleaned_speeches, key=lambda x: x['word_count'])
            shortest = min(cleaned_speeches, key=lambda x: x['word_count'])
            
            print(f"\n  Longest speech: {longest['title'][:50]} ({longest['word_count']:,} words)")
            print(f"  Shortest speech: {shortest['title'][:50]} ({shortest['word_count']:,} words)")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Trump speech transcripts')
    parser.add_argument('input_file', help='Input JSON or CSV file with raw transcripts')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = TranscriptCleaner(args.config)
    
    # Process speeches
    cleaned_speeches = cleaner.process_speeches(args.input_file)
    
    if cleaned_speeches:
        # Save results
        json_file, csv_file = cleaner.save_cleaned_data(cleaned_speeches)
        print(f"\n✓ Cleaning complete!")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
    else:
        print("\n✗ No speeches were cleaned")
        sys.exit(1)


if __name__ == "__main__":
    main()

