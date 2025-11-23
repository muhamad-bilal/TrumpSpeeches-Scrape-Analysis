"""
Data Transformation Pipeline for Trump Speech Transcripts

This script applies advanced NLP transformations:
1. Sentence segmentation and tokenization
2. POS tagging and lemmatization
3. Named Entity Recognition
4. Sentiment analysis (VADER + transformers)
5. Emotion classification
6. Readability metrics
7. N-gram extraction
8. TF-IDF vectorization
9. Embedding generation
10. Temporal indexing
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import spacy
import nltk
from nltk import ngrams
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json,
    get_stopwords, print_section, print_stats
)


class TranscriptTransformer:
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.nlp_config = self.config['nlp']
        
        print_section("INITIALIZING NLP MODELS")
        
        # Load spaCy model
        print("Loading spaCy model...")
        try:
            spacy_model = self.nlp_config.get('spacy_model', 'en_core_web_sm')
            self.nlp = spacy.load(spacy_model)
            print(f"✓ Loaded {spacy_model}")
        except OSError:
            print(f"  Model not found. Downloading...")
            import subprocess
            try:
                subprocess.run(['python', '-m', 'spacy', 'download', spacy_model], check=True)
                self.nlp = spacy.load(spacy_model)
            except:
                print("  Falling back to en_core_web_sm...")
                subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
                self.nlp = spacy.load('en_core_web_sm')
        
        print("Loading VADER sentiment analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        print("✓ VADER loaded")
        
        print("Ensuring NLTK data is available...")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        print("✓ NLTK data ready")
        
        print("Loading sentence transformer...")
        try:
            embedding_model = self.nlp_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✓ Loaded {embedding_model}")
        except Exception as e:
            print(f"  Warning: Could not load sentence transformer: {e}")
            self.embedder = None
        
        # Get stopwords
        self.stopwords = get_stopwords(self.config)
        
        print(f"✓ Loaded {len(self.stopwords)} stopwords")
        print("\n✓ All models initialized successfully\n")
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def tokenize_and_pos_tag(self, text: str) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """
        Tokenize and POS tag text
        
        Returns:
            Tuple of (tokens, pos_tags, lemmas)
        """
        doc = self.nlp(text)
        
        tokens = [token.text for token in doc if not token.is_space]
        pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        
        return tokens, pos_tags, lemmas
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities
        
        Returns:
            Dictionary mapping entity type to list of entities
        """
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entity (countries, cities)
            'DATE': [],
            'MONEY': [],
            'NORP': [],  # Nationalities, religious/political groups
            'FAC': [],   # Facilities
            'LOC': [],   # Non-GPE locations
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        return scores
    
    def analyze_sentiment_per_sentence(self, sentences: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for each sentence"""
        sentiment_scores = []
        
        for sentence in sentences:
            scores = self.analyze_sentiment_vader(sentence)
            sentiment_scores.append(scores)
        
        return sentiment_scores
    
    def classify_emotions_lexicon(self, text: str) -> Dict[str, float]:
        """
        Classify emotions using NRCLex (lexicon-based)
        
        Returns 8 emotion scores
        """
        try:
            from nrclex import NRCLex
            
            emotion_obj = NRCLex(text)
            raw_scores = emotion_obj.affect_frequencies
            
            emotions = {
                'anger': raw_scores.get('anger', 0),
                'fear': raw_scores.get('fear', 0),
                'joy': raw_scores.get('joy', 0),
                'sadness': raw_scores.get('sadness', 0),
                'surprise': raw_scores.get('surprise', 0),
                'disgust': raw_scores.get('disgust', 0),
                'trust': raw_scores.get('trust', 0),
                'anticipation': raw_scores.get('anticipation', 0)
            }
            
            return emotions
        except Exception as e:
            return {
                'anger': 0, 'fear': 0, 'joy': 0, 'sadness': 0,
                'surprise': 0, 'disgust': 0, 'trust': 0, 'anticipation': 0
            }
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability metrics"""
        try:
            metrics = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
            }
        except Exception as e:
            metrics = {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'smog_index': 0,
                'coleman_liau_index': 0,
                'automated_readability_index': 0,
            }
        
        return metrics
    
    def extract_ngrams(self, tokens: List[str], n_range: Tuple[int, int] = (1, 3)) -> Dict[str, List[Tuple]]:
        """
        Extract n-grams from tokens
        
        Args:
            tokens: List of tokens
            n_range: Tuple of (min_n, max_n)
        
        Returns:
            Dictionary mapping n-gram size to list of n-grams with counts
        """
        # Filter out stopwords and punctuation
        filtered_tokens = [
            token.lower() for token in tokens
            if token.lower() not in self.stopwords and token.isalpha()
        ]
        
        ngram_dict = {}
        
        for n in range(n_range[0], n_range[1] + 1):
            ngram_list = list(ngrams(filtered_tokens, n))
            ngram_counts = Counter(ngram_list)
            
            # Get top 20 n-grams
            top_ngrams = ngram_counts.most_common(20)
            ngram_dict[f'{n}gram'] = [
                (' '.join(gram), count) for gram, count in top_ngrams
            ]
        
        return ngram_dict
    
    def generate_embeddings(self, text: str) -> List[float]:
        if self.embedder is None:
            return []
        
        try:
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"  Warning: Embedding generation failed: {e}")
            return []
    
    def transform_speech(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        text = speech.get('cleaned_text', '')
        
        if not text or len(text) < 10:
            print("  Warning: Text too short, skipping transformations")
            return speech
        
        transformed = speech.copy()
        
        # Sentence segmentation
        sentences = self.segment_sentences(text)
        transformed['sentences'] = sentences
        transformed['sentence_count'] = len(sentences)
        
        # Tokenization, POS tagging, lemmatization
        tokens, pos_tags, lemmas = self.tokenize_and_pos_tag(text)
        transformed['tokens'] = tokens
        transformed['token_count'] = len(tokens)
        transformed['lemmas'] = lemmas
        
        # POS tag distribution
        pos_counts = Counter([tag for _, tag in pos_tags])
        transformed['pos_distribution'] = dict(pos_counts)
        
        # Named Entity Recognition
        entities = self.extract_entities(text)
        transformed['entities'] = entities
        
        # Entity counts
        entity_counts = {k: len(v) for k, v in entities.items()}
        transformed['entity_counts'] = entity_counts
        
        for ent_type, ent_list in entities.items():
            if ent_list:
                top_entities = Counter(ent_list).most_common(10)
                transformed[f'top_{ent_type.lower()}'] = [
                    {'entity': ent, 'count': count} for ent, count in top_entities
                ]
        
        sentiment_overall = self.analyze_sentiment_vader(text)
        transformed['sentiment_overall'] = sentiment_overall
        
        sentiment_sentences = self.analyze_sentiment_per_sentence(sentences)
        transformed['sentiment_sentences'] = sentiment_sentences
        
        avg_sentiment = {
            'neg': np.mean([s['neg'] for s in sentiment_sentences]),
            'neu': np.mean([s['neu'] for s in sentiment_sentences]),
            'pos': np.mean([s['pos'] for s in sentiment_sentences]),
            'compound': np.mean([s['compound'] for s in sentiment_sentences])
        }
        transformed['sentiment_average'] = avg_sentiment
        
        emotions = self.classify_emotions_lexicon(text)
        transformed['emotions'] = emotions
        
        readability = self.calculate_readability(text)
        transformed['readability'] = readability
        
        ngram_range = (
            self.nlp_config.get('ngram_range', {}).get('min', 1),
            self.nlp_config.get('ngram_range', {}).get('max', 3)
        )
        ngrams_data = self.extract_ngrams(tokens, ngram_range)
        transformed['ngrams'] = ngrams_data
        
        # Generate embddings
        embeddings = self.generate_embeddings(text)
        if embeddings:
            transformed['embeddings'] = embeddings
            transformed['embedding_dimension'] = len(embeddings)
        
        # Add processing timestamp
        transformed['transformed_at'] = datetime.now().isoformat()
        
        return transformed
    
    # def transform_all_speeches(self, input_file: str) -> List[Dict[str, Any]]:
    #     """Transform all speeches from cleaned data"""
    #     print_section("DATA TRANSFORMATION PIPELINE")
        
    #     # Load cleaned data
    #     print(f"\nLoading cleaned data from {input_file}...")
    #     speeches = load_json(input_file)
        
    #     if not speeches:
    #         print("Error: No data loaded")
    #         return []
        
    #     print(f"✓ Loaded {len(speeches)} speeches")
        
    #     # Transform each speech
    #     print("\nApplying NLP transformations...")
    #     transformed_speeches = []
        
    #     for i, speech in enumerate(speeches, 1):
    #         title = speech.get('title', 'Untitled')[:50]
    #         print(f"\n  [{i}/{len(speeches)}] {title}...")
            
    #         try:
    #             transformed = self.transform_speech(speech)
    #             transformed_speeches.append(transformed)
                
    #             # Print quick stats
    #             print(f"    Sentences: {transformed.get('sentence_count', 0)}")
    #             print(f"    Tokens: {transformed.get('token_count', 0)}")
    #             print(f"    Entities: {sum(transformed.get('entity_counts', {}).values())}")
    #             print(f"    Sentiment: {transformed.get('sentiment_overall', {}).get('compound', 0):.3f}")
                
    #         except Exception as e:
    #             print(f"    Error transforming speech: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             # Still add the speech with basic data
    #             transformed_speeches.append(speech)
        
    #     return transformed_speeches
    
    def transform_all_speeches(self, input_file: str) -> List[Dict[str, Any]]:
        print_section("DATA TRANSFORMATION PIPELINE")
        
        # Load cleaned data - support both JSON and CSV
        print(f"\nLoading cleaned data from {input_file}...")
        
        input_path = Path(input_file)
        speeches = []
        
        if input_path.suffix.lower() == '.csv':
            # Load CSV and convert to list of dicts
            import pandas as pd
            df = pd.read_csv(input_file, encoding='utf-8')
            speeches = df.to_dict('records')
            print(f"✓ Loaded {len(speeches)} speeches from CSV")
        else:
            # Load JSON
            speeches = load_json(input_file)
            if isinstance(speeches, dict) and 'speeches' in speeches:
                speeches = speeches['speeches']
            elif not isinstance(speeches, list):
                speeches = [speeches] if speeches else []
            print(f"✓ Loaded {len(speeches)} speeches from JSON")
        
        if not speeches:
            print("Error: No data loaded")
            return []
        
        print("\nApplying NLP transformations...")
        transformed_speeches = []
        
        for i, speech in enumerate(speeches, 1):
            title = speech.get('title', 'Untitled')[:50]
            print(f"\n  [{i}/{len(speeches)}] {title}...")
            
            try:
                transformed = self.transform_speech(speech)
                transformed_speeches.append(transformed)
                
                # Print quick stats
                print(f"    Sentences: {transformed.get('sentence_count', 0)}")
                print(f"    Tokens: {transformed.get('token_count', 0)}")
                print(f"    Entities: {sum(transformed.get('entity_counts', {}).values())}")
                print(f"    Sentiment: {transformed.get('sentiment_overall', {}).get('compound', 0):.3f}")
                
            except Exception as e:
                print(f"    Error transforming speech: {e}")
                import traceback
                traceback.print_exc()
                # Still add the speech with basic data
                transformed_speeches.append(speech)
        
        return transformed_speeches
    
    def save_transformed_data(self, transformed_speeches: List[Dict[str, Any]]):
        print_section("SAVING TRANSFORMED DATA")
        
        output_dir = Path(self.config['paths']['transformed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_file = output_dir / f"speeches_nlp_features_{timestamp}.json"
        save_json(transformed_speeches, str(json_file))
        
        print(f"\n✓ Transformation complete!")
        print(f"  Saved to: {json_file}")
        
        return str(json_file)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform Trump speech transcripts with NLP')
    parser.add_argument('input_file', help='Input JSON file with cleaned transcripts')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    transformer = TranscriptTransformer(args.config)
    
    # Transform speeches
    transformed_speeches = transformer.transform_all_speeches(args.input_file)
    
    if transformed_speeches:
        # Save results
        output_file = transformer.save_transformed_data(transformed_speeches)
        print(f"\n✓ Output: {output_file}")
    else:
        print("\n✗ No speeches were transformed")
        sys.exit(1)


if __name__ == "__main__":
    main()

