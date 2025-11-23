"""
Feature Engineering for Trump Speech Analysis

This script computes derived attributes and features:
1. Linguistic features (complexity, diversity, readability)
2. Rhetorical features (anaphora, repetition, contrast)
3. Political/thematic features (keyword clusters, topics)
4. Emotional features (aggregated emotions, sentiment stats)
5. Psychological features (power/affiliation, pronouns, modals)
6. Temporal features (trends over time)
7. NER metrics (top entities, frequencies)
8. Stylistic features (adj/adv ratio, questions, exclamations)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import re

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json, save_csv,
    extract_anaphora_patterns, detect_contrast_markers,
    calculate_repetition_density, extract_modal_verbs,
    calculate_type_token_ratio, extract_pronouns,
    create_speech_id, print_section, print_stats
)


class FeatureEngineer:
    """Engineer features from transformed speech data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature engineer"""
        self.config = load_config(config_path)
        self.feature_config = self.config.get('features', {})
        self.anaphora_patterns = self.feature_config.get('anaphora_patterns', [])
        self.keyword_clusters = self.feature_config.get('keyword_clusters', {})
        self.wpm = self.feature_config.get('words_per_minute', 150)
    
    def extract_basic_attributes(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic speech attributes"""
        text = speech.get('cleaned_text', '')
        
        attributes = {
            'speech_id': speech.get('speech_id', ''),
            'title': speech.get('title', ''),
            'date': speech.get('date', ''),
            'url': speech.get('url', ''),
            'speaker': 'Donald Trump',
            'word_count': speech.get('word_count', len(text.split())),
            'sentence_count': speech.get('sentence_count', 0),
            'char_count': speech.get('char_count', len(text)),
        }
        
        # Duration estimate (words per minute)
        if attributes['word_count'] > 0:
            attributes['duration_estimate_minutes'] = attributes['word_count'] / self.wpm
        else:
            attributes['duration_estimate_minutes'] = 0
        
        return attributes
    
    def compute_linguistic_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute linguistic complexity and diversity features"""
        text = speech.get('cleaned_text', '')
        tokens = speech.get('tokens', text.split())
        sentences = speech.get('sentences', [])
        
        features = {}
        
        # Average sentence length
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['std_sentence_length'] = np.std(sentence_lengths)
            features['max_sentence_length'] = max(sentence_lengths)
            features['min_sentence_length'] = min(sentence_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['std_sentence_length'] = 0
            features['max_sentence_length'] = 0
            features['min_sentence_length'] = 0
        
        # Type-token ratio (lexical diversity)
        features['type_token_ratio'] = calculate_type_token_ratio(tokens)
        
        # Lexical diversity (unique words / total words)
        if tokens:
            features['lexical_diversity'] = len(set(tokens)) / len(tokens)
            features['unique_word_count'] = len(set(tokens))
        else:
            features['lexical_diversity'] = 0
            features['unique_word_count'] = 0
        
        # Readability scores
        readability = speech.get('readability', {})
        for metric, value in readability.items():
            features[f'readability_{metric}'] = value
        
        # Average word length
        if tokens:
            word_lengths = [len(token) for token in tokens if token.isalpha()]
            if word_lengths:
                features['avg_word_length'] = np.mean(word_lengths)
        else:
            features['avg_word_length'] = 0
        
        return features
    
    def compute_rhetorical_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute rhetorical device features"""
        text = speech.get('cleaned_text', '')
        
        features = {}
        
        # Anaphora patterns
        anaphora_counts = extract_anaphora_patterns(text, self.anaphora_patterns)
        features['anaphora_total'] = sum(anaphora_counts.values())
        for pattern, count in anaphora_counts.items():
            # Clean pattern name for feature name
            pattern_name = re.sub(r'[^\w]', '_', pattern).lower()
            features[f'anaphora_{pattern_name}'] = count
        
        # Contrast markers
        features['contrast_markers'] = detect_contrast_markers(text)
        
        # Repetition density
        features['repetition_density'] = calculate_repetition_density(text)
        
        # Alliteration detection (simple: count words starting with same letter)
        words = text.lower().split()
        alliteration_count = 0
        for i in range(len(words) - 1):
            if words[i] and words[i+1] and words[i][0] == words[i+1][0]:
                alliteration_count += 1
        features['alliteration_count'] = alliteration_count
        
        return features
    
    def compute_political_thematic_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute political and thematic keyword features"""
        text = speech.get('cleaned_text', '').lower()
        
        features = {}
        
        # Keyword cluster counts
        for cluster_name, keywords in self.keyword_clusters.items():
            count = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                count += len(re.findall(pattern, text))
            
            features[f'keywords_{cluster_name}'] = count
        
        # Total political keywords
        features['keywords_total'] = sum(
            v for k, v in features.items() if k.startswith('keywords_')
        )
        
        return features
    
    def compute_emotional_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregated emotional features"""
        features = {}
        
        # Overall sentiment
        sentiment = speech.get('sentiment_overall', {})
        features['sentiment_neg'] = sentiment.get('neg', 0)
        features['sentiment_neu'] = sentiment.get('neu', 0)
        features['sentiment_pos'] = sentiment.get('pos', 0)
        features['sentiment_compound'] = sentiment.get('compound', 0)
        
        # Sentiment statistics across sentences
        sentiment_sentences = speech.get('sentiment_sentences', [])
        if sentiment_sentences:
            compounds = [s.get('compound', 0) for s in sentiment_sentences]
            features['sentiment_mean'] = np.mean(compounds)
            features['sentiment_variance'] = np.var(compounds)
            features['sentiment_std'] = np.std(compounds)
            features['sentiment_max'] = max(compounds)
            features['sentiment_min'] = min(compounds)
            features['sentiment_range'] = max(compounds) - min(compounds)
        else:
            features['sentiment_mean'] = 0
            features['sentiment_variance'] = 0
            features['sentiment_std'] = 0
            features['sentiment_max'] = 0
            features['sentiment_min'] = 0
            features['sentiment_range'] = 0
        
        # Emotion scores
        emotions = speech.get('emotions', {})
        for emotion, score in emotions.items():
            features[f'emotion_{emotion}'] = score
        
        # Dominant emotion
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            features['dominant_emotion'] = dominant_emotion[0]
            features['dominant_emotion_score'] = dominant_emotion[1]
        else:
            features['dominant_emotion'] = 'none'
            features['dominant_emotion_score'] = 0
        
        return features
    
    def compute_psychological_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute psychological profiling features"""
        text = speech.get('cleaned_text', '')
        
        features = {}
        
        # Pronoun analysis
        pronouns = extract_pronouns(text)
        for pronoun_type, count in pronouns.items():
            features[f'pronoun_{pronoun_type}'] = count
        
        # Pronoun ratios
        first_singular = pronouns.get('first_singular', 0)
        first_plural = pronouns.get('first_plural', 0)
        
        # I vs We ratio (ego vs collective)
        if first_plural > 0:
            features['pronoun_i_we_ratio'] = first_singular / first_plural
        else:
            features['pronoun_i_we_ratio'] = first_singular  # If no "we", just count "I"
        
        # Total pronoun usage
        features['pronoun_total'] = sum(pronouns.values())
        
        # Modal verb analysis (will, should, must, etc.)
        modals = extract_modal_verbs(text)
        for modal, count in modals.items():
            features[f'modal_{modal}'] = count
        
        # Total modals (certainty/confidence indicator)
        features['modal_total'] = sum(modals.values())
        
        # Certainty markers
        certainty_words = ['absolutely', 'definitely', 'certainly', 'clearly', 
                          'obviously', 'undoubtedly', 'surely']
        certainty_count = 0
        text_lower = text.lower()
        for word in certainty_words:
            pattern = r'\b' + word + r'\b'
            certainty_count += len(re.findall(pattern, text_lower))
        features['certainty_markers'] = certainty_count
        
        # Power vs affiliation words (simplified)
        power_words = ['strong', 'power', 'control', 'authority', 'lead', 
                      'command', 'force', 'dominate', 'superior']
        affiliation_words = ['together', 'team', 'support', 'help', 'share',
                            'cooperate', 'unite', 'friend', 'ally']
        
        power_count = sum(len(re.findall(r'\b' + w + r'\b', text_lower)) for w in power_words)
        affiliation_count = sum(len(re.findall(r'\b' + w + r'\b', text_lower)) for w in affiliation_words)
        
        features['power_words'] = power_count
        features['affiliation_words'] = affiliation_count
        
        if affiliation_count > 0:
            features['power_affiliation_ratio'] = power_count / affiliation_count
        else:
            features['power_affiliation_ratio'] = power_count
        
        return features
    
    def compute_ner_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Named Entity Recognition features"""
        features = {}
        
        # Entity counts
        entity_counts = speech.get('entity_counts', {})
        for entity_type, count in entity_counts.items():
            features[f'ner_{entity_type.lower()}_count'] = count
        
        # Total entities
        features['ner_total_entities'] = sum(entity_counts.values())
        
        # Top entities (convert to strings for CSV compatibility)
        for entity_type in ['PERSON', 'ORG', 'GPE']:
            top_key = f'top_{entity_type.lower()}'
            if top_key in speech:
                top_entities = speech[top_key][:5]  # Top 5
                if top_entities:
                    # Create comma-separated string
                    entity_names = [e['entity'] for e in top_entities]
                    features[f'top_{entity_type.lower()}_names'] = ', '.join(entity_names)
                else:
                    features[f'top_{entity_type.lower()}_names'] = ''
        
        return features
    
    def compute_stylistic_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Compute stylistic features"""
        text = speech.get('cleaned_text', '')
        pos_distribution = speech.get('pos_distribution', {})
        
        features = {}
        
        # Adjective and adverb usage
        adj_count = pos_distribution.get('ADJ', 0)
        adv_count = pos_distribution.get('ADV', 0)
        
        features['adj_count'] = adj_count
        features['adv_count'] = adv_count
        
        if adv_count > 0:
            features['adj_adv_ratio'] = adj_count / adv_count
        else:
            features['adj_adv_ratio'] = adj_count
        
        # Question and exclamation counts
        features['question_count'] = text.count('?')
        features['exclamation_count'] = text.count('!')
        
        # All caps words (emphasis)
        words = text.split()
        all_caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['all_caps_words'] = all_caps_count
        
        # Superlatives (most, best, greatest, etc.)
        superlatives = ['most', 'best', 'greatest', 'worst', 'largest', 
                       'smallest', 'highest', 'lowest', 'biggest']
        superlative_count = 0
        text_lower = text.lower()
        for word in superlatives:
            pattern = r'\b' + word + r'\b'
            superlative_count += len(re.findall(pattern, text_lower))
        features['superlative_count'] = superlative_count
        
        return features
    
    def engineer_features(self, speech: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer all features for a single speech"""
        features = {}
        
        # Basic attributes
        features.update(self.extract_basic_attributes(speech))
        
        # Linguistic features
        features.update(self.compute_linguistic_features(speech))
        
        # Rhetorical features
        features.update(self.compute_rhetorical_features(speech))
        
        # Political/thematic features
        features.update(self.compute_political_thematic_features(speech))
        
        # Emotional features
        features.update(self.compute_emotional_features(speech))
        
        # Psychological features
        features.update(self.compute_psychological_features(speech))
        
        # NER features
        features.update(self.compute_ner_features(speech))
        
        # Stylistic features
        features.update(self.compute_stylistic_features(speech))
        
        # Add timestamp
        features['features_extracted_at'] = datetime.now().isoformat()
        
        return features
    
    def process_all_speeches(self, input_file: str) -> pd.DataFrame:
        """Process all speeches and create feature dataframe"""
        print_section("FEATURE ENGINEERING PIPELINE")
        
        # Load transformed data
        print(f"\nLoading transformed data from {input_file}...")
        speeches = load_json(input_file)
        
        if not speeches:
            print("Error: No data loaded")
            return pd.DataFrame()
        
        print(f"✓ Loaded {len(speeches)} speeches")
        
        # Engineer features for each speech
        print("\nEngineering features...")
        all_features = []
        
        for i, speech in enumerate(speeches, 1):
            title = speech.get('title', 'Untitled')[:50]
            print(f"  [{i}/{len(speeches)}] {title}...")
            
            try:
                features = self.engineer_features(speech)
                all_features.append(features)
                
                # Print sample features
                print(f"    Lexical diversity: {features.get('lexical_diversity', 0):.3f}")
                print(f"    Sentiment: {features.get('sentiment_compound', 0):.3f}")
                print(f"    Modal verbs: {features.get('modal_total', 0)}")
                
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        print(f"\n✓ Engineered {len(df.columns)} features for {len(df)} speeches")
        
        return df
    
    def save_features(self, df: pd.DataFrame):
        """Save feature dataframe"""
        print_section("SAVING FEATURES")
        
        output_dir = Path(self.config['paths']['transformed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_file = output_dir / f"speeches_features_complete_{timestamp}.csv"
        save_csv(df, str(csv_file))
        
        # Save JSON
        json_file = output_dir / f"speeches_features_complete_{timestamp}.json"
        df_dict = df.to_dict('records')
        save_json(df_dict, str(json_file))
        
        # Print summary
        self.print_feature_summary(df)
        
        return str(csv_file), str(json_file)
    
    def print_feature_summary(self, df: pd.DataFrame):
        """Print feature summary statistics"""
        print_section("FEATURE SUMMARY")
        
        print_stats("Total speeches", len(df))
        print_stats("Total features", len(df.columns))
        
        # Summary stats for key features
        if 'word_count' in df.columns:
            print(f"\n  Word count stats:")
            print(f"    Mean: {df['word_count'].mean():.0f}")
            print(f"    Min: {df['word_count'].min():.0f}")
            print(f"    Max: {df['word_count'].max():.0f}")
        
        if 'sentiment_compound' in df.columns:
            print(f"\n  Sentiment stats:")
            print(f"    Mean: {df['sentiment_compound'].mean():.3f}")
            print(f"    Min: {df['sentiment_compound'].min():.3f}")
            print(f"    Max: {df['sentiment_compound'].max():.3f}")
        
        if 'lexical_diversity' in df.columns:
            print(f"\n  Lexical diversity stats:")
            print(f"    Mean: {df['lexical_diversity'].mean():.3f}")
            print(f"    Min: {df['lexical_diversity'].min():.3f}")
            print(f"    Max: {df['lexical_diversity'].max():.3f}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Engineer features from transformed transcripts')
    parser.add_argument('input_file', help='Input JSON file with transformed transcripts')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize engineer
    engineer = FeatureEngineer(args.config)
    
    # Process speeches
    df = engineer.process_all_speeches(args.input_file)
    
    if not df.empty:
        # Save results
        csv_file, json_file = engineer.save_features(df)
        print(f"\n✓ Feature engineering complete!")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")
    else:
        print("\n✗ No features were engineered")
        sys.exit(1)


if __name__ == "__main__":
    main()

