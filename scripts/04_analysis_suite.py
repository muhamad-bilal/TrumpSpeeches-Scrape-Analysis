"""
Analysis Suite for Trump Speech Features

Modular analysis functions:
1. Linguistic Analysis - complexity metrics, readability trends
2. Rhetorical Analysis - anaphora, alliteration, contrast patterns
3. Political & Thematic Analysis - topic modeling, keyword clustering
4. Emotional Analysis - sentiment timelines, emotion distributions
5. Psychological Profiling - Big Five markers, power language, pronouns
6. Comparative Trend Analysis - temporal trends, change points
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

# Machine learning
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Topic modeling
try:
    from gensim import corpora
    from gensim.models import LdaModel
    HAS_GENSIM = True
except:
    HAS_GENSIM = False

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json, save_csv,
    print_section, print_stats
)


class SpeechAnalyzer:
    """Comprehensive speech analysis suite"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize analyzer"""
        self.config = load_config(config_path)
        self.analysis_config = self.config.get('analysis', {})
        self.results = {}
    
    def load_data(self, features_file: str, transformed_file: str = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """Load feature DataFrame and optionally transformed data"""
        print_section("LOADING DATA")
        
        # Load features CSV
        print(f"Loading features from {features_file}...")
        if features_file.endswith('.csv'):
            df = pd.read_csv(features_file, encoding='utf-8')
        else:
            data = load_json(features_file)
            df = pd.DataFrame(data)
        
        print(f"✓ Loaded {len(df)} speeches with {len(df.columns)} features")
        
        # Load transformed data if provided
        transformed_data = None
        if transformed_file:
            print(f"\nLoading transformed data from {transformed_file}...")
            transformed_data = load_json(transformed_file)
            print(f"✓ Loaded {len(transformed_data)} transformed speeches")
        
        return df, transformed_data
    
    # ========== 1. LINGUISTIC ANALYSIS ==========
    
    def linguistic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze linguistic complexity and diversity"""
        print_section("LINGUISTIC ANALYSIS")
        
        results = {}
        
        # Complexity metrics
        complexity_cols = [c for c in df.columns if 'readability' in c]
        if complexity_cols:
            complexity_stats = df[complexity_cols].describe()
            results['complexity_statistics'] = complexity_stats.to_dict()
            
            print("\nReadability Statistics:")
            for col in complexity_cols:
                mean_val = df[col].mean()
                print(f"  {col}: {mean_val:.2f}")
        
        # Sentence length distribution
        if 'avg_sentence_length' in df.columns:
            results['sentence_length'] = {
                'mean': df['avg_sentence_length'].mean(),
                'std': df['avg_sentence_length'].std(),
                'min': df['avg_sentence_length'].min(),
                'max': df['avg_sentence_length'].max(),
                'median': df['avg_sentence_length'].median()
            }
            print(f"\nAverage sentence length: {results['sentence_length']['mean']:.2f} words")
        
        # Lexical diversity trends
        if 'lexical_diversity' in df.columns:
            results['lexical_diversity'] = {
                'mean': df['lexical_diversity'].mean(),
                'std': df['lexical_diversity'].std(),
                'min': df['lexical_diversity'].min(),
                'max': df['lexical_diversity'].max()
            }
            print(f"Mean lexical diversity: {results['lexical_diversity']['mean']:.3f}")
        
        # Vocabulary richness
        if 'unique_word_count' in df.columns and 'word_count' in df.columns:
            results['vocabulary_richness'] = {
                'total_unique_words': df['unique_word_count'].sum(),
                'avg_unique_per_speech': df['unique_word_count'].mean(),
                'avg_words_per_speech': df['word_count'].mean()
            }
            print(f"\nTotal unique words: {results['vocabulary_richness']['total_unique_words']:,}")
        
        # Temporal trends (if date available)
        if 'date' in df.columns:
            df_sorted = df.sort_values('date')
            if 'lexical_diversity' in df.columns:
                # Calculate trend
                x = np.arange(len(df_sorted))
                y = df_sorted['lexical_diversity'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                results['lexical_diversity_trend'] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'trend': 'increasing' if slope > 0 else 'decreasing'
                }
                print(f"\nLexical diversity trend: {results['lexical_diversity_trend']['trend']} (R²={r_value**2:.3f})")
        
        self.results['linguistic'] = results
        return results
    
    # ========== 2. RHETORICAL ANALYSIS ==========
    
    def rhetorical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rhetorical devices"""
        print_section("RHETORICAL ANALYSIS")
        
        results = {}
        
        # Anaphora patterns
        anaphora_cols = [c for c in df.columns if c.startswith('anaphora_')]
        if anaphora_cols:
            anaphora_stats = {}
            print("\nAnaphora usage:")
            for col in anaphora_cols:
                total = df[col].sum()
                avg = df[col].mean()
                anaphora_stats[col] = {
                    'total': int(total),
                    'average_per_speech': avg,
                    'max': int(df[col].max())
                }
                print(f"  {col}: {total:.0f} total, {avg:.2f} per speech")
            
            results['anaphora'] = anaphora_stats
        
        # Contrast markers
        if 'contrast_markers' in df.columns:
            results['contrast_markers'] = {
                'total': int(df['contrast_markers'].sum()),
                'average_per_speech': df['contrast_markers'].mean(),
                'max': int(df['contrast_markers'].max())
            }
            print(f"\nContrast markers: {results['contrast_markers']['average_per_speech']:.2f} per speech")
        
        # Repetition density
        if 'repetition_density' in df.columns:
            results['repetition_density'] = {
                'mean': df['repetition_density'].mean(),
                'std': df['repetition_density'].std(),
                'max': df['repetition_density'].max()
            }
            print(f"Repetition density: {results['repetition_density']['mean']:.2f} per 1000 words")
        
        # Alliteration
        if 'alliteration_count' in df.columns:
            results['alliteration'] = {
                'total': int(df['alliteration_count'].sum()),
                'average_per_speech': df['alliteration_count'].mean()
            }
            print(f"Alliteration: {results['alliteration']['average_per_speech']:.2f} instances per speech")
        
        # Superlatives (stylistic emphasis)
        if 'superlative_count' in df.columns:
            results['superlatives'] = {
                'total': int(df['superlative_count'].sum()),
                'average_per_speech': df['superlative_count'].mean()
            }
            print(f"Superlatives: {results['superlatives']['average_per_speech']:.2f} per speech")
        
        self.results['rhetorical'] = results
        return results
    
    # ========== 3. POLITICAL & THEMATIC ANALYSIS ==========
    
    def political_thematic_analysis(self, df: pd.DataFrame, transformed_data: List[Dict] = None) -> Dict[str, Any]:
        """Analyze political themes and topics"""
        print_section("POLITICAL & THEMATIC ANALYSIS")
        
        results = {}
        
        # Keyword cluster analysis
        keyword_cols = [c for c in df.columns if c.startswith('keywords_') and c != 'keywords_total']
        if keyword_cols:
            print("\nKeyword cluster frequencies:")
            cluster_stats = {}
            for col in keyword_cols:
                cluster_name = col.replace('keywords_', '')
                total = df[col].sum()
                avg = df[col].mean()
                cluster_stats[cluster_name] = {
                    'total': int(total),
                    'average_per_speech': avg
                }
                print(f"  {cluster_name}: {total:.0f} total, {avg:.2f} per speech")
            
            results['keyword_clusters'] = cluster_stats
            
            # Find dominant themes
            cluster_totals = {k: v['total'] for k, v in cluster_stats.items()}
            sorted_clusters = sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True)
            results['dominant_themes'] = sorted_clusters[:5]
            
            print(f"\nTop 3 themes:")
            for theme, count in sorted_clusters[:3]:
                print(f"  {theme}: {count}")
        
        # Topic modeling with LDA (if we have text data)
        if transformed_data and HAS_GENSIM:
            print("\nPerforming LDA topic modeling...")
            try:
                results['lda_topics'] = self.perform_lda(transformed_data)
            except Exception as e:
                print(f"  LDA failed: {e}")
        
        self.results['political_thematic'] = results
        return results
    
    def perform_lda(self, transformed_data: List[Dict], n_topics: int = None) -> Dict[str, Any]:
        """Perform LDA topic modeling"""
        if n_topics is None:
            n_topics = self.analysis_config.get('lda', {}).get('n_topics', 10)
        
        # Extract cleaned text and tokenize
        texts = []
        for speech in transformed_data:
            lemmas = speech.get('lemmas', [])
            if lemmas:
                # Filter stopwords
                filtered = [l.lower() for l in lemmas if l.isalpha() and len(l) > 3]
                texts.append(filtered)
        
        if not texts:
            return {}
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=self.analysis_config.get('lda', {}).get('passes', 15),
            iterations=self.analysis_config.get('lda', {}).get('iterations', 400),
            random_state=42
        )
        
        # Extract topics
        topics = {}
        print(f"\n  Extracted {n_topics} topics:")
        for idx, topic in lda_model.print_topics(-1):
            topics[f'topic_{idx}'] = topic
            # Print top words
            words = [w.split('*')[1].strip('"') for w in topic.split('+')[:5]]
            print(f"    Topic {idx}: {', '.join(words)}")
        
        return {
            'n_topics': n_topics,
            'topics': topics
        }
    
    # ========== 4. EMOTIONAL ANALYSIS ==========
    
    def emotional_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze emotional content and sentiment"""
        print_section("EMOTIONAL ANALYSIS")
        
        results = {}
        
        # Sentiment statistics
        if 'sentiment_compound' in df.columns:
            results['sentiment_overall'] = {
                'mean': df['sentiment_compound'].mean(),
                'std': df['sentiment_compound'].std(),
                'min': df['sentiment_compound'].min(),
                'max': df['sentiment_compound'].max(),
                'median': df['sentiment_compound'].median()
            }
            
            print(f"\nSentiment (compound) statistics:")
            print(f"  Mean: {results['sentiment_overall']['mean']:.3f}")
            print(f"  Std: {results['sentiment_overall']['std']:.3f}")
            print(f"  Range: [{results['sentiment_overall']['min']:.3f}, {results['sentiment_overall']['max']:.3f}]")
            
            # Classify speeches
            positive = (df['sentiment_compound'] > 0.05).sum()
            neutral = ((df['sentiment_compound'] >= -0.05) & (df['sentiment_compound'] <= 0.05)).sum()
            negative = (df['sentiment_compound'] < -0.05).sum()
            
            results['sentiment_distribution'] = {
                'positive': int(positive),
                'neutral': int(neutral),
                'negative': int(negative)
            }
            
            print(f"\nSentiment distribution:")
            print(f"  Positive: {positive} ({positive/len(df)*100:.1f}%)")
            print(f"  Neutral: {neutral} ({neutral/len(df)*100:.1f}%)")
            print(f"  Negative: {negative} ({negative/len(df)*100:.1f}%)")
            
            # Most positive/negative speeches
            if 'title' in df.columns:
                most_positive = df.loc[df['sentiment_compound'].idxmax()]
                most_negative = df.loc[df['sentiment_compound'].idxmin()]
                
                results['extremes'] = {
                    'most_positive': {
                        'title': most_positive['title'],
                        'score': float(most_positive['sentiment_compound'])
                    },
                    'most_negative': {
                        'title': most_negative['title'],
                        'score': float(most_negative['sentiment_compound'])
                    }
                }
                
                print(f"\nMost positive: {most_positive['title'][:50]} ({most_positive['sentiment_compound']:.3f})")
                print(f"Most negative: {most_negative['title'][:50]} ({most_negative['sentiment_compound']:.3f})")
        
        # Emotion distribution
        emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
        if emotion_cols:
            emotion_means = df[emotion_cols].mean()
            results['emotion_averages'] = emotion_means.to_dict()
            
            print(f"\nAverage emotion scores:")
            for emotion, score in sorted(emotion_means.items(), key=lambda x: x[1], reverse=True):
                emotion_name = emotion.replace('emotion_', '')
                print(f"  {emotion_name}: {score:.3f}")
        
        # Emotional volatility
        if 'sentiment_std' in df.columns:
            results['emotional_volatility'] = {
                'mean_volatility': df['sentiment_std'].mean(),
                'speeches_with_high_volatility': int((df['sentiment_std'] > df['sentiment_std'].mean() + df['sentiment_std'].std()).sum())
            }
            print(f"\nMean emotional volatility: {results['emotional_volatility']['mean_volatility']:.3f}")
        
        self.results['emotional'] = results
        return results
    
    # ========== 5. PSYCHOLOGICAL PROFILING ==========
    
    def psychological_profiling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze psychological indicators"""
        print_section("PSYCHOLOGICAL PROFILING")
        
        results = {}
        
        # Pronoun analysis (ego vs collective)
        pronoun_cols = [c for c in df.columns if c.startswith('pronoun_')]
        if pronoun_cols:
            pronoun_stats = df[pronoun_cols].mean()
            results['pronoun_usage'] = pronoun_stats.to_dict()
            
            print("\nPronoun usage (average per speech):")
            for col in pronoun_cols:
                print(f"  {col}: {df[col].mean():.2f}")
            
            # I vs We ratio
            if 'pronoun_i_we_ratio' in df.columns:
                results['ego_vs_collective'] = {
                    'mean_ratio': df['pronoun_i_we_ratio'].mean(),
                    'interpretation': 'ego-focused' if df['pronoun_i_we_ratio'].mean() > 1 else 'collective-focused'
                }
                print(f"\nI/We ratio: {results['ego_vs_collective']['mean_ratio']:.2f} ({results['ego_vs_collective']['interpretation']})")
        
        # Modal verb analysis (certainty/confidence)
        modal_cols = [c for c in df.columns if c.startswith('modal_')]
        if modal_cols:
            modal_stats = df[modal_cols].mean()
            results['modal_verb_usage'] = modal_stats.to_dict()
            
            print("\nTop modal verbs:")
            top_modals = modal_stats.nlargest(5)
            for modal, count in top_modals.items():
                print(f"  {modal}: {count:.2f}")
        
        # Power vs affiliation
        if 'power_affiliation_ratio' in df.columns:
            results['power_affiliation'] = {
                'mean_ratio': df['power_affiliation_ratio'].mean(),
                'power_words_avg': df['power_words'].mean() if 'power_words' in df.columns else 0,
                'affiliation_words_avg': df['affiliation_words'].mean() if 'affiliation_words' in df.columns else 0,
                'orientation': 'power-oriented' if df['power_affiliation_ratio'].mean() > 1 else 'affiliation-oriented'
            }
            print(f"\nPower/Affiliation ratio: {results['power_affiliation']['mean_ratio']:.2f} ({results['power_affiliation']['orientation']})")
        
        # Certainty markers
        if 'certainty_markers' in df.columns:
            results['certainty'] = {
                'mean_per_speech': df['certainty_markers'].mean(),
                'total': int(df['certainty_markers'].sum())
            }
            print(f"\nCertainty markers: {results['certainty']['mean_per_speech']:.2f} per speech")
        
        self.results['psychological'] = results
        return results
    
    # ========== 6. COMPARATIVE TREND ANALYSIS ==========
    
    def comparative_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends over time"""
        print_section("COMPARATIVE TREND ANALYSIS")
        
        results = {}
        
        if 'date' not in df.columns:
            print("No date information available for trend analysis")
            return results
        
        # Convert date to datetime
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        df_sorted = df.sort_values('date_parsed').copy()
        
        # Sentiment trend
        if 'sentiment_compound' in df.columns:
            x = np.arange(len(df_sorted))
            y = df_sorted['sentiment_compound'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            results['sentiment_trend'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend': 'increasingly positive' if slope > 0 else 'increasingly negative',
                'significant': p_value < 0.05
            }
            
            print(f"\nSentiment trend: {results['sentiment_trend']['trend']}")
            print(f"  R²: {r_value**2:.3f}, p-value: {p_value:.4f}")
        
        # Complexity trend
        if 'readability_flesch_kincaid_grade' in df.columns:
            x = np.arange(len(df_sorted))
            y = df_sorted['readability_flesch_kincaid_grade'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            results['complexity_trend'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend': 'increasingly complex' if slope > 0 else 'increasingly simple',
                'significant': p_value < 0.05
            }
            
            print(f"\nComplexity trend: {results['complexity_trend']['trend']}")
            print(f"  R²: {r_value**2:.3f}, p-value: {p_value:.4f}")
        
        # Detect change points (simple method: split data and compare means)
        if len(df_sorted) >= 10:
            mid_point = len(df_sorted) // 2
            first_half = df_sorted.iloc[:mid_point]
            second_half = df_sorted.iloc[mid_point:]
            
            changes = {}
            
            if 'sentiment_compound' in df.columns:
                sentiment_change = second_half['sentiment_compound'].mean() - first_half['sentiment_compound'].mean()
                changes['sentiment'] = {
                    'first_half_mean': first_half['sentiment_compound'].mean(),
                    'second_half_mean': second_half['sentiment_compound'].mean(),
                    'change': sentiment_change,
                    'direction': 'more positive' if sentiment_change > 0 else 'more negative'
                }
            
            if 'lexical_diversity' in df.columns:
                diversity_change = second_half['lexical_diversity'].mean() - first_half['lexical_diversity'].mean()
                changes['lexical_diversity'] = {
                    'first_half_mean': first_half['lexical_diversity'].mean(),
                    'second_half_mean': second_half['lexical_diversity'].mean(),
                    'change': diversity_change,
                    'direction': 'more diverse' if diversity_change > 0 else 'less diverse'
                }
            
            results['temporal_changes'] = changes
            
            print(f"\nTemporal changes (first half vs second half):")
            for metric, data in changes.items():
                print(f"  {metric}: {data['direction']} (Δ={data['change']:.3f})")
        
        self.results['comparative_trend'] = results
        return results
    
    # ========== SAVE RESULTS ==========
    
    def save_all_results(self):
        """Save all analysis results"""
        print_section("SAVING ANALYSIS RESULTS")
        
        output_dir = Path(self.config['paths']['results'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results as JSON
        json_file = output_dir / f"analysis_results_{timestamp}.json"
        save_json(self.results, str(json_file))
        
        print(f"\n✓ Analysis complete!")
        print(f"  Results saved to: {json_file}")
        
        return str(json_file)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Trump speech features')
    parser.add_argument('features_file', help='Input CSV/JSON file with engineered features')
    parser.add_argument('--transformed', help='Optional: transformed data JSON for topic modeling')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SpeechAnalyzer(args.config)
    
    # Load data
    df, transformed_data = analyzer.load_data(args.features_file, args.transformed)
    
    # Run all analyses
    analyzer.linguistic_analysis(df)
    analyzer.rhetorical_analysis(df)
    analyzer.political_thematic_analysis(df, transformed_data)
    analyzer.emotional_analysis(df)
    analyzer.psychological_profiling(df)
    analyzer.comparative_trend_analysis(df)
    
    # Save results
    output_file = analyzer.save_all_results()
    print(f"\n✓ All analyses complete!")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()

