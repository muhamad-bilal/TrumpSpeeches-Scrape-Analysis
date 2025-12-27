"""
Trump Speech Analyzer - Predictive Analytics GUI
=================================================
Interactive interface for 7 predictive models:
- Model 0: MEGA MODEL - Unified Trump Predictor
- Model 1: Entity Reaction Profiler
- Model 2: Personality Compatibility Predictor
- Model 3: Negotiation Success Predictor
- Model 4: Response Classifier (ML)
- Model 5: Psychological Influence Model
- Model 6: Trigger Word Detector
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
from mega_trump_model import MegaTrumpPredictor

# Page config
st.set_page_config(
    page_title="Trump Speech Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #262730;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #505050;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #262730;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .input-label {
        font-weight: 500;
        color: #262730;
        margin-bottom: 0.5rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .positive { color: #27ae60; font-weight: 600; }
    .negative { color: #e74c3c; font-weight: 600; }
    .neutral { color: #f39c12; font-weight: 600; }
    .high { color: #e74c3c; font-weight: 600; }
    .medium { color: #f39c12; font-weight: 600; }
    .low { color: #27ae60; font-weight: 600; }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA LOADING FUNCTIONS
# ============================================

@st.cache_data
def load_features_data():
    """Load speech features data"""
    data_dir = Path('data/transformed')
    feature_files = list(data_dir.glob('speeches_features_complete_*.json'))
    if feature_files:
        latest = max(feature_files, key=lambda p: p.stat().st_mtime)
        with open(latest, 'r', encoding='utf-8') as f:
            return pd.DataFrame(json.load(f))
    return None

@st.cache_data
def load_entity_profiles():
    """Load entity reaction profiles for Model 1"""
    results_dir = Path('data/results')
    profile_files = list(results_dir.glob('entity_reaction_profiles_*.csv'))
    if profile_files:
        latest = max(profile_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest)
        baseline = df['avg_sentiment'].mean()
        return df, baseline
    return None, None

@st.cache_data
def load_trigger_words():
    """Load trigger word data for Model 6"""
    results_dir = Path('data/results')
    trigger_files = list(results_dir.glob('trigger_words_*.csv'))
    if trigger_files:
        latest = max(trigger_files, key=lambda p: p.stat().st_mtime)
        return pd.read_csv(latest)
    # Compute on-the-fly if needed
    speeches = load_cleaned_speeches()
    if speeches:
        return compute_trigger_words_from_speeches(speeches)
    return None

@st.cache_data
def load_cleaned_speeches():
    """Load cleaned speech texts"""
    cleaned_dir = Path('data/cleaned')
    speech_files = list(cleaned_dir.glob('speeches_cleaned_*.json'))
    if speech_files:
        latest = max(speech_files, key=lambda p: p.stat().st_mtime)
        with open(latest, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ============================================
# MODEL 1: ENTITY REACTION PROFILER
# ============================================

def predict_entity_reaction(entity_name, df_profiles, baseline):
    """Predict Trump's reaction to an entity"""
    if df_profiles is None:
        return {'status': 'NO_DATA', 'message': 'Entity profiles not loaded'}
    
    entity_lower = entity_name.lower().strip()
    matches = df_profiles[df_profiles['entity_name'].str.lower().str.contains(entity_lower, na=False)]
    
    if len(matches) == 0:
        return {
            'status': 'NOT_FOUND',
            'entity': entity_name,
            'message': f"No data found for '{entity_name}'. Try a different entity."
        }
    
    match = matches.iloc[0]
    raw_sentiment = float(match['avg_sentiment'])
    centered_sentiment = raw_sentiment - baseline
    
    # Calculate emotion ratio
    neg_emotions = float(match.get('avg_anger', 0)) + float(match.get('avg_fear', 0)) + float(match.get('avg_disgust', 0))
    pos_emotions = float(match.get('avg_joy', 0)) + float(match.get('avg_trust', 0))
    total_emotion = neg_emotions + pos_emotions
    emotion_ratio = neg_emotions / total_emotion if total_emotion > 0 else 0.5
    
    if emotion_ratio > 0.35 or centered_sentiment < -0.03:
        sentiment_label = 'NEGATIVE'
    elif emotion_ratio < 0.30 and centered_sentiment > 0.03:
        sentiment_label = 'POSITIVE'
    else:
        sentiment_label = 'NEUTRAL'
    
    volatility = float(match['volatility'])
    if sentiment_label == 'NEGATIVE':
        reaction_type = 'ATTACK_MODE' if volatility > 0.15 else 'CRITICISM_MODE'
    elif sentiment_label == 'POSITIVE':
        reaction_type = 'CELEBRATION_MODE'
    elif volatility > 0.15:
        reaction_type = 'UNPREDICTABLE'
    else:
        reaction_type = 'NEUTRAL_MODE'
    
    return {
        'status': 'FOUND',
        'entity': match['entity_name'],
        'entity_type': match['entity_type'],
        'speech_count': int(match['speech_count']),
        'sentiment_label': sentiment_label,
        'volatility_label': 'HIGH' if volatility > 0.15 else 'LOW' if volatility < 0.05 else 'MEDIUM',
        'reaction_type': reaction_type
    }


# ============================================
# MODEL 2: PERSONALITY COMPATIBILITY PREDICTOR
# ============================================

def compute_trump_big_five(df):
    """Compute Trump's Big Five personality from speech features"""
    features = {
        'i_we_ratio': df['pronoun_i_we_ratio'].mean(),
        'first_singular': df['pronoun_first_singular'].mean(),
        'lexical_diversity': df['lexical_diversity'].mean(),
        'readability': df['readability_flesch_reading_ease'].mean(),
        'avg_sentence_length': df['avg_sentence_length'].mean(),
        'modal_will': df['modal_will'].mean(),
        'certainty_markers': df['certainty_markers'].mean(),
        'superlatives': df['superlative_count'].mean(),
        'sentiment_pos': df['sentiment_pos'].mean(),
        'sentiment_neg': df['sentiment_neg'].mean(),
        'sentiment_variance': df['sentiment_variance'].mean(),
        'exclamations': df['exclamation_count'].mean(),
        'all_caps': df['all_caps_words'].mean(),
        'affiliation_words': df['affiliation_words'].mean(),
        'power_affiliation_ratio': df['power_affiliation_ratio'].mean(),
    }
    
    openness = min(100, max(0, (features['lexical_diversity'] * 100 + (100 - features['readability']) * 0.5 + features['avg_sentence_length'] * 2) / 3 * 2))
    conscientiousness = min(100, max(0, (features['modal_will'] * 3 + features['certainty_markers'] * 5 + features['superlatives'] * 2) / 10 * 4))
    extraversion = min(100, max(0, (features['first_singular'] / 10 + features['sentiment_pos'] * 100 + features['exclamations'] * 5 + features['all_caps'] * 2 + features['superlatives'] * 3) / 5 * 2))
    agreeableness = min(100, max(0, ((1 / (features['i_we_ratio'] + 0.1)) * 10 + features['affiliation_words'] * 3 + (1 - features['sentiment_neg']) * 50 + (100 - features['power_affiliation_ratio'] * 100) * 0.5) / 4))
    neuroticism = min(100, max(0, (features['sentiment_neg'] * 200 + features['sentiment_variance'] * 100 + features['all_caps'] * 1 + (100 - features['certainty_markers'] * 5)) / 4))
    
    return {
        'openness': round(openness, 1),
        'conscientiousness': round(conscientiousness, 1),
        'extraversion': round(extraversion, 1),
        'agreeableness': round(agreeableness, 1),
        'neuroticism': round(neuroticism, 1)
    }

def predict_compatibility(other_profile, trump_profile):
    """Predict compatibility between Trump and another personality"""
    trump_dominance = (trump_profile['extraversion'] + (100 - trump_profile['agreeableness'])) / 2
    other_dominance = (other_profile['extraversion'] + (100 - other_profile['agreeableness'])) / 2
    
    dominance_diff = trump_dominance - other_dominance
    if dominance_diff > 30:
        dominance_score, dominance_note = 25, "Deferential - Trump will appreciate respect"
    elif dominance_diff > 10:
        dominance_score, dominance_note = 20, "Respectful but not pushover"
    elif dominance_diff > -10:
        dominance_score, dominance_note = 10, "Power struggle likely"
    else:
        dominance_score, dominance_note = 5, "Will trigger competitive response"
    
    if other_profile['agreeableness'] > 70:
        agree_score, agree_note = 25, "Accommodating - will avoid conflict"
    elif other_profile['agreeableness'] > 50:
        agree_score, agree_note = 18, "Diplomatic - can manage tensions"
    elif other_profile['agreeableness'] > 30:
        agree_score, agree_note = 10, "Both competitive - friction expected"
    else:
        agree_score, agree_note = 5, "Conflict inevitable"
    
    if other_profile['conscientiousness'] > 60:
        consc_score = 20
    elif other_profile['conscientiousness'] > 40:
        consc_score = 15
    else:
        consc_score = 8
    
    if other_profile['neuroticism'] < 30:
        neuro_score = 15
    elif other_profile['neuroticism'] < 50:
        neuro_score = 12
    else:
        neuro_score = 7
    
    status_proxy = (other_profile['extraversion'] + other_profile['conscientiousness']) / 2
    status_score = 15 if status_proxy > 60 else 10 if status_proxy > 40 else 5
    
    total_score = dominance_score + agree_score + consc_score + neuro_score + status_score
    
    if total_score >= 75:
        response_type = "COOPERATIVE"
        description = "Trump will likely be friendly and receptive"
    elif total_score >= 55:
        response_type = "TRANSACTIONAL"
        description = "Trump will engage if there's something in it for him"
    elif total_score >= 40:
        response_type = "COMPETITIVE" if other_dominance > trump_dominance - 10 else "DISMISSIVE"
        description = "Trump will try to assert dominance" if response_type == "COMPETITIVE" else "Trump may ignore or belittle"
    else:
        response_type = "HOSTILE" if other_dominance > trump_dominance else "CONTEMPTUOUS"
        description = "Trump will likely attack" if response_type == "HOSTILE" else "Trump will show disdain"
    
    return {
        'compatibility_score': total_score,
        'response_type': response_type,
        'description': description,
        'factors': {
            'Dominance': {'score': dominance_score, 'note': dominance_note},
            'Agreeableness': {'score': agree_score, 'note': agree_note},
        }
    }


# ============================================
# MODEL 3: NEGOTIATION SUCCESS PREDICTOR
# ============================================

def predict_negotiation_success(topic, style, strategies):
    """Predict negotiation success probability"""
    
    topic_favorability = {
        'trade': 0.75, 'economy': 0.85, 'immigration': 0.65, 'foreign_policy': 0.60,
        'security': 0.80, 'healthcare': 0.45, 'environment': 0.30, 'taxes': 0.80,
        'china': 0.40, 'russia': 0.55, 'nato': 0.50, 'media': 0.20,
        'democrats': 0.15, 'military': 0.90
    }
    
    style_effectiveness = {
        'flattering': 0.85, 'transactional': 0.80, 'assertive': 0.55,
        'diplomatic': 0.50, 'confrontational': 0.25, 'intellectual': 0.35,
        'emotional': 0.40, 'humble': 0.60
    }
    
    strategy_multipliers = {
        'show_win': 1.3, 'media_angle': 1.2, 'business_frame': 1.25,
        'loyalty_appeal': 1.15, 'facts_only': 0.8, 'moral_argument': 0.6,
        'expert_consensus': 0.7, 'precedent': 0.75
    }
    
    topic_score = topic_favorability.get(topic.lower(), 0.5)
    style_score = style_effectiveness.get(style.lower(), 0.5)
    
    base_prob = (topic_score * 0.4 + style_score * 0.6)
    
    final_prob = base_prob
    applied_strategies = []
    for strategy in strategies:
        strategy_key = strategy.lower().replace(' ', '_')
        if strategy_key in strategy_multipliers:
            mult = strategy_multipliers[strategy_key]
            final_prob *= mult
            applied_strategies.append({'strategy': strategy, 'multiplier': mult})
    
    final_prob = min(0.95, max(0.05, final_prob))
    
    if final_prob >= 0.75:
        outcome = "HIGH SUCCESS LIKELY"
    elif final_prob >= 0.55:
        outcome = "MODERATE SUCCESS LIKELY"
    elif final_prob >= 0.35:
        outcome = "UNCERTAIN"
    else:
        outcome = "LOW SUCCESS"
    
    return {
        'success_probability': round(final_prob * 100, 1),
        'outcome': outcome,
        'topic_favorability': round(topic_score * 100),
        'style_effectiveness': round(style_score * 100),
        'strategies_applied': applied_strategies
    }


# ============================================
# MODEL 4: RESPONSE CLASSIFIER (ML-based)
# ============================================

def classify_response(sentiment, neg_ratio, pos_ratio, power_ratio, certainty):
    """Classify Trump's response type based on linguistic features"""
    
    if neg_ratio > 0.15 and power_ratio > 0.6:
        return 'ATTACK', 85
    elif neg_ratio > 0.12 and sentiment < 0.3:
        return 'ATTACK', 75
    elif pos_ratio > 0.2 and sentiment > 0.8:
        return 'PRAISE', 80
    elif pos_ratio > 0.15 and power_ratio < 0.4:
        return 'PRAISE', 70
    elif 0.4 <= power_ratio <= 0.6 and certainty > 0.3:
        return 'NEGOTIATE', 65
    elif sentiment > 0.5 and neg_ratio < 0.08:
        return 'NEGOTIATE', 60
    elif certainty < 0.2 and 0.3 < sentiment < 0.7:
        return 'DEFLECT', 55
    else:
        return 'NEUTRAL', 50


# ============================================
# MODEL 5: PSYCHOLOGICAL INFLUENCE MODEL
# ============================================

def predict_influence_strategy(context, goal, trump_profile):
    """Predict best influence strategy for Trump"""
    
    p_ego = min(100, trump_profile.get('extraversion', 50) + (100 - trump_profile.get('agreeableness', 50)))
    
    base_effectiveness = {
        'Liking (Flattery)': {'score': 90, 'tactic': 'Compliment his achievements, show admiration'},
        'Scarcity': {'score': 75, 'tactic': 'Offer exclusive opportunities, "only you can do this"'},
        'Social Proof': {'score': 70, 'tactic': 'Show poll numbers, crowd sizes, popularity'},
        'Reciprocity': {'score': 55, 'tactic': 'Do him a favor first, then ask'},
        'Commitment': {'score': 40, 'tactic': 'Reference his past statements and promises'},
        'Authority': {'score': 35, 'tactic': 'Position HIM as the expert, not outside experts'},
    }
    
    context_mods = {
        'business': {'Scarcity': 1.2, 'Reciprocity': 1.1},
        'political': {'Social Proof': 1.3, 'Liking (Flattery)': 1.1},
        'personal': {'Liking (Flattery)': 1.3, 'Reciprocity': 1.2},
        'media': {'Social Proof': 1.3, 'Scarcity': 1.1}
    }
    
    results = []
    for principle, data in base_effectiveness.items():
        score = data['score']
        if context in context_mods and principle in context_mods[context]:
            score *= context_mods[context][principle]
        results.append({
            'principle': principle,
            'effectiveness': min(100, round(score, 1)),
            'tactic': data['tactic']
        })
    
    results.sort(key=lambda x: x['effectiveness'], reverse=True)
    return results


# ============================================
# MODEL 6: TRIGGER WORD DETECTOR
# ============================================

def compute_trigger_words_from_speeches(speeches):
    """Compute trigger words from speech data"""
    EMOTION_LEXICON = {
        'anger': ['angry', 'furious', 'hate', 'terrible', 'horrible', 'disaster', 'corrupt', 'crooked', 'liar', 'enemy', 'destroy', 'attack', 'stupid', 'idiot', 'pathetic', 'disgrace', 'worst', 'nasty', 'fake', 'fraud'],
        'fear': ['afraid', 'fear', 'terror', 'threat', 'danger', 'scary', 'worried', 'nervous', 'crime', 'criminal', 'violence', 'terrorist'],
        'joy': ['happy', 'joy', 'love', 'wonderful', 'great', 'amazing', 'fantastic', 'beautiful', 'excellent', 'incredible', 'tremendous', 'success', 'win', 'winner', 'best', 'proud'],
        'sadness': ['sad', 'sorry', 'unfortunate', 'tragic', 'terrible', 'horrible', 'devastating', 'heartbreaking'],
        'trust': ['trust', 'believe', 'faith', 'honest', 'true', 'loyal', 'reliable', 'america', 'freedom', 'liberty'],
        'disgust': ['disgusting', 'disgrace', 'shameful', 'pathetic', 'loser', 'fake', 'fraud', 'corrupt', 'nasty', 'worst'],
    }
    
    word_stats = defaultdict(lambda: {'high_emotion': 0, 'low_emotion': 0, 'total_intensity': 0, 'emotions': defaultdict(int)})
    
    for speech in speeches:
        text = speech.get('cleaned_text', speech.get('text', ''))
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            intensity = sum(1 for w in words for emo, kw in EMOTION_LEXICON.items() if w in kw)
            is_high = intensity >= 3
            
            for word in set(words):
                if is_high:
                    word_stats[word]['high_emotion'] += 1
                else:
                    word_stats[word]['low_emotion'] += 1
                word_stats[word]['total_intensity'] += intensity
                
                for emo, keywords in EMOTION_LEXICON.items():
                    if word in keywords:
                        word_stats[word]['emotions'][emo] += 1
    
    rows = []
    for word, stats in word_stats.items():
        total = stats['high_emotion'] + stats['low_emotion']
        if total < 5:
            continue
        
        high_ratio = stats['high_emotion'] / total
        trigger_score = high_ratio * 100
        
        emotions = dict(stats['emotions'])
        neg_emotions = emotions.get('anger', 0) + emotions.get('fear', 0) + emotions.get('disgust', 0)
        pos_emotions = emotions.get('joy', 0) + emotions.get('trust', 0)
        total_emo = neg_emotions + pos_emotions
        neg_ratio = neg_emotions / total_emo if total_emo > 0 else 0.5
        
        dominant = max(emotions, key=emotions.get) if emotions else 'none'
        
        rows.append({
            'word': word,
            'trigger_score': trigger_score,
            'high_emotion_ratio': high_ratio,
            'negative_ratio': neg_ratio,
            'dominant_emotion': dominant,
            'occurrences': total
        })
    
    return pd.DataFrame(rows) if rows else None

def predict_trigger(word_or_phrase, df_triggers):
    """Predict if a word/phrase triggers emotional response"""
    if df_triggers is None:
        return {'status': 'NO_DATA', 'message': 'Trigger data not available'}
    
    trigger_lookup = df_triggers.set_index('word').to_dict('index')
    words = re.findall(r'\b[a-zA-Z]{3,}\b', word_or_phrase.lower())
    
    if not words:
        return {'status': 'INVALID', 'message': 'No valid words found'}
    
    found_words = [(w, trigger_lookup[w]) for w in words if w in trigger_lookup]
    
    if not found_words:
        return {'status': 'NOT_FOUND', 'words_checked': words, 'message': 'No data for these words'}
    
    max_score = max(data['trigger_score'] for _, data in found_words)
    avg_negative = sum(data['negative_ratio'] for _, data in found_words) / len(found_words)
    
    trigger_level = 'HIGH' if max_score >= 50 else 'MEDIUM' if max_score >= 35 else 'LOW'
    valence = 'NEGATIVE' if avg_negative > 0.45 else 'POSITIVE' if avg_negative < 0.35 else 'MIXED'
    
    return {
        'status': 'FOUND',
        'trigger_score': round(max_score, 1),
        'trigger_level': trigger_level,
        'negative_ratio': round(avg_negative, 3),
        'valence': valence,
        'words_analyzed': [w for w, _ in found_words]
    }


# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown('<h1 class="main-header">Trump Behavioral Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">7 Predictive Models: 1 Unified Mega Model + 6 Specialized Models</p>', unsafe_allow_html=True)
    
    # Load all data
    df_features = load_features_data()
    df_profiles, baseline = load_entity_profiles()
    df_triggers = load_trigger_words()
    
    # Compute Trump's profile if features available
    trump_profile = compute_trump_big_five(df_features) if df_features is not None else None
    
    # Sidebar - Model Selection
    st.sidebar.markdown("## Model Selection")
    
    model_options = [
        "Model 0: MEGA MODEL (Unified)",
        "Model 1: Entity Reaction Profiler",
        "Model 2: Personality Compatibility",
        "Model 3: Negotiation Predictor",
        "Model 4: Response Classifier (ML)",
        "Model 5: Influence Strategy",
        "Model 6: Trigger Word Detector"
    ]
    
    selected_model = st.sidebar.radio("Choose a model:", model_options)
    model_num = int(selected_model.split(':')[0].split()[-1])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Description")
    
    descriptions = {
        0: "UNIFIED MODEL: Combines all 6 sub-models for comprehensive predictions",
        1: "Predicts Trump's emotional reaction to entities (people, countries, organizations)",
        2: "Predicts how Trump will respond to different personality types (Big Five)",
        3: "Predicts negotiation success based on topic, style, and strategies",
        4: "ML classifier predicting response type (ATTACK, PRAISE, NEGOTIATE, etc.)",
        5: "Recommends psychological influence tactics (Cialdini's principles)",
        6: "Identifies words that trigger strong emotional responses"
    }
    st.sidebar.info(descriptions.get(model_num, ""))
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Input Parameters</h2>', unsafe_allow_html=True)
        
        # ============= MODEL 0: MEGA MODEL (UNIFIED) =============
        if model_num == 0:
            st.markdown("**Fill in any combination of inputs below:**")
            st.caption("The more inputs you provide, the more accurate the prediction.")
            
            # Initialize Mega Model
            model_functions = {
                'predict_entity_reaction': predict_entity_reaction,
                'compute_trump_big_five': compute_trump_big_five,
                'predict_compatibility': predict_compatibility,
                'predict_negotiation_success': predict_negotiation_success,
                'classify_response': classify_response,
                'predict_influence_strategy': predict_influence_strategy,
                'predict_trigger': predict_trigger
            }
            
            mega_predictor = MegaTrumpPredictor(
                df_features=df_features,
                df_profiles=df_profiles,
                df_triggers=df_triggers,
                baseline=baseline,
                model_functions=model_functions
            )
            
            # Input sections with expanders
            with st.expander("Entity Context", expanded=False):
                entity = st.text_input("Entity name:", placeholder="e.g., China, Biden, NATO...", key="mega_entity")
            
            with st.expander("Personality Analysis", expanded=False):
                use_personality = st.checkbox("Include personality compatibility", key="mega_personality")
                if use_personality:
                    openness = st.slider("Openness", 0, 100, 50, key="mega_o")
                    conscientiousness = st.slider("Conscientiousness", 0, 100, 50, key="mega_c")
                    extraversion = st.slider("Extraversion", 0, 100, 50, key="mega_e")
                    agreeableness = st.slider("Agreeableness", 0, 100, 50, key="mega_a")
                    neuroticism = st.slider("Neuroticism", 0, 100, 50, key="mega_n")
            
            with st.expander("Negotiation Context", expanded=False):
                use_negotiation = st.checkbox("Include negotiation analysis", key="mega_negotiation")
                if use_negotiation:
                    topic = st.selectbox("Topic:", ['trade', 'economy', 'immigration', 'security', 'military', 'china', 'russia', 'nato', 'healthcare', 'environment', 'media', 'democrats'], key="mega_topic")
                    style = st.selectbox("Communication Style:", ['flattering', 'transactional', 'assertive', 'diplomatic', 'confrontational', 'intellectual', 'humble'], key="mega_style")
                    strategies = st.multiselect("Strategies:", ['show_win', 'media_angle', 'business_frame', 'loyalty_appeal', 'facts_only', 'moral_argument', 'expert_consensus'], key="mega_strategies")
            
            with st.expander("Linguistic Features", expanded=False):
                use_linguistic = st.checkbox("Include linguistic analysis", key="mega_linguistic")
                if use_linguistic:
                    sentiment = st.slider("Overall Sentiment", -1.0, 1.0, 0.5, key="mega_sent")
                    neg_ratio = st.slider("Negative Ratio", 0.0, 1.0, 0.1, key="mega_neg")
                    pos_ratio = st.slider("Positive Ratio", 0.0, 1.0, 0.15, key="mega_pos")
                    power_ratio = st.slider("Power/Affiliation Ratio", 0.0, 1.0, 0.5, key="mega_pow")
                    certainty = st.slider("Certainty Level", 0.0, 1.0, 0.3, key="mega_cert")
            
            with st.expander("Influence Strategy", expanded=False):
                use_influence = st.checkbox("Include influence strategy", key="mega_influence")
                if use_influence:
                    context = st.selectbox("Context:", ['business', 'political', 'personal', 'media'], key="mega_context")
                    goal = st.selectbox("Your Goal:", ['agreement', 'favor', 'information', 'relationship'], key="mega_goal")
            
            with st.expander("Trigger Words", expanded=False):
                words = st.text_input("Words/Phrases:", placeholder="e.g., fake news, winning...", key="mega_words")
            
            if st.button("Run Mega Prediction", type="primary", key="mega_btn"):
                # Build input dictionary
                inputs = {}
                if entity: inputs['entity'] = entity
                if use_personality: 
                    inputs['personality_traits'] = {
                        'openness': openness,
                        'conscientiousness': conscientiousness,
                        'extraversion': extraversion,
                        'agreeableness': agreeableness,
                        'neuroticism': neuroticism
                    }
                if use_negotiation:
                    inputs['topic'] = topic
                    inputs['style'] = style
                    inputs['strategies'] = strategies
                if use_linguistic:
                    inputs['linguistic_features'] = {
                        'sentiment': sentiment,
                        'neg_ratio': neg_ratio,
                        'pos_ratio': pos_ratio,
                        'power_ratio': power_ratio,
                        'certainty': certainty
                    }
                if use_influence:
                    inputs['context'] = context
                    inputs['goal'] = goal
                if words: inputs['words'] = words
                
                # Run unified prediction
                result = mega_predictor.predict_comprehensive(**inputs)
                
                with col2:
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    
                    if result.get('overall_prediction') == 'INSUFFICIENT_DATA':
                        st.error(result.get('message', 'Insufficient data'))
                    else:
                        # Overall prediction
                        st.success(f"**Overall Response:** {result['overall_prediction']}")
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Confidence", f"{result['confidence']}%")
                        with col_metric2:
                            st.metric("Models Used", result.get('num_models_used', 0))
                        
                        # Category scores
                        st.markdown("**Category Scores:**")
                        for cat, score in result['category_scores'].items():
                            st.progress(score/100, text=f"{cat}: {score}%")
                        
                        # Sub-model contributions
                        with st.expander("Sub-Model Contributions", expanded=False):
                            for model, pred in result['sub_model_contributions'].items():
                                st.markdown(f"**{model.replace('_', ' ').title()}:**")
                                if isinstance(pred, dict):
                                    for key, val in list(pred.items())[:3]:
                                        st.caption(f"  {key}: {val}")
                                elif isinstance(pred, list):
                                    st.caption(f"  {pred[0] if pred else 'N/A'}")
                                else:
                                    st.caption(f"  {pred}")
                                st.markdown("---")
                        
                        # Recommendations
                        st.markdown("**Unified Recommendations:**")
                        for rec in result.get('recommendations', []):
                            st.info(rec)
        
        # ============= MODEL 1: ENTITY REACTION =============
        elif model_num == 1:
            user_input = st.text_input("Enter an entity (person, country, organization):", placeholder="e.g., China, Biden, NATO...")
            examples = ["China", "Biden", "America", "Democrats", "Russia", "NATO"]
            st.markdown("**Quick examples:**")
            cols = st.columns(3)
            for i, ex in enumerate(examples):
                if cols[i % 3].button(ex, key=f"ex1_{ex}"):
                    user_input = ex
            
            if user_input:
                result = predict_entity_reaction(user_input, df_profiles, baseline)
                with col2:
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    if result['status'] == 'FOUND':
                        st.success(f"**Entity:** {result['entity']} ({result['entity_type']})")
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Reaction Type", result['reaction_type'])
                            st.metric("Sentiment", result['sentiment_label'])
                        with col_m2:
                            st.metric("Volatility", result['volatility_label'])
                            st.metric("Speech Count", result.get('speech_count', 'N/A'))
                    else:
                        st.warning(result.get('message', 'Entity not found'))
        
        # ============= MODEL 2: PERSONALITY COMPATIBILITY =============
        elif model_num == 2:
            st.markdown("**Enter personality traits (0-100):**")
            openness = st.slider("Openness", 0, 100, 50, key="o")
            conscientiousness = st.slider("Conscientiousness", 0, 100, 50, key="c")
            extraversion = st.slider("Extraversion", 0, 100, 50, key="e")
            agreeableness = st.slider("Agreeableness", 0, 100, 50, key="a")
            neuroticism = st.slider("Neuroticism", 0, 100, 50, key="n")
            
            # Quick presets
            st.markdown("**Quick Presets:**")
            cols = st.columns(3)
            presets = {
                "Alpha Leader": {'openness': 50, 'conscientiousness': 70, 'extraversion': 90, 'agreeableness': 20, 'neuroticism': 30},
                "Diplomat": {'openness': 65, 'conscientiousness': 80, 'extraversion': 55, 'agreeableness': 70, 'neuroticism': 25},
                "Yes-Man": {'openness': 40, 'conscientiousness': 50, 'extraversion': 30, 'agreeableness': 85, 'neuroticism': 50}
            }
            for i, (name, vals) in enumerate(presets.items()):
                if cols[i % 3].button(name, key=f"preset_{name}"):
                    openness, conscientiousness, extraversion, agreeableness, neuroticism = vals.values()
            
            if st.button("Predict Compatibility", type="primary", key="compat_btn"):
                other_profile = {'openness': openness, 'conscientiousness': conscientiousness, 'extraversion': extraversion, 'agreeableness': agreeableness, 'neuroticism': neuroticism}
                result = predict_compatibility(other_profile, trump_profile) if trump_profile else {'response_type': 'NO_DATA', 'description': 'Feature data not loaded'}
                with col2:
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Compatibility Score", f"{result['compatibility_score']}/100")
                    with col_m2:
                        st.metric("Predicted Response", result['response_type'])
                    st.info(result['description'])
                    if 'factors' in result:
                        with st.expander("Factor Breakdown", expanded=False):
                            for factor, data in result['factors'].items():
                                st.markdown(f"**{factor}:** {data.get('score', 'N/A')} pts")
                                st.caption(data.get('note', ''))
        
        # ============= MODEL 3: NEGOTIATION SUCCESS =============
        elif model_num == 3:
            topic = st.selectbox("Select Topic:", ['trade', 'economy', 'immigration', 'security', 'military', 'china', 'russia', 'nato', 'healthcare', 'environment', 'media', 'democrats'])
            style = st.selectbox("Communication Style:", ['flattering', 'transactional', 'assertive', 'diplomatic', 'confrontational', 'intellectual', 'humble'])
            strategies = st.multiselect("Strategies:", ['show_win', 'media_angle', 'business_frame', 'loyalty_appeal', 'facts_only', 'moral_argument', 'expert_consensus'])
            
            if st.button("Predict Success", type="primary", key="neg_btn"):
                result = predict_negotiation_success(topic, style, strategies)
                with col2:
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Success Probability", f"{result['success_probability']}%")
                    with col_m2:
                        st.metric("Outcome", result['outcome'])
                    st.info(f"Topic Favorability: {result['topic_favorability']}% | Style Effectiveness: {result['style_effectiveness']}%")
                    if result.get('strategies_applied'):
                        with st.expander("Applied Strategies", expanded=False):
                            for strat in result['strategies_applied']:
                                effect = "BOOST" if strat.get('multiplier', 1) > 1 else "PENALTY"
                                st.caption(f"{strat.get('strategy', 'Unknown')}: {strat.get('multiplier', 1)}x ({effect})")
        
        # ============= MODEL 4: RESPONSE CLASSIFIER =============
        elif model_num == 4:
            st.markdown("**Enter linguistic features:**")
            sentiment = st.slider("Overall Sentiment", -1.0, 1.0, 0.5, key="sent")
            neg_ratio = st.slider("Negative Ratio", 0.0, 1.0, 0.1, key="neg")
            pos_ratio = st.slider("Positive Ratio", 0.0, 1.0, 0.15, key="pos")
            power_ratio = st.slider("Power/Affiliation Ratio", 0.0, 1.0, 0.5, key="pow")
            certainty = st.slider("Certainty Level", 0.0, 1.0, 0.3, key="cert")
            
            if st.button("Classify Response", type="primary", key="ml_btn"):
                response_type, confidence = classify_response(sentiment, neg_ratio, pos_ratio, power_ratio, certainty)
                with col2:
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Predicted Response", response_type)
                    with col_m2:
                        st.metric("Confidence", f"{confidence}%")
        
        # ============= MODEL 5: INFLUENCE STRATEGY =============
        elif model_num == 5:
            context = st.selectbox("Context:", ['business', 'political', 'personal', 'media'])
            goal = st.selectbox("Your Goal:", ['agreement', 'favor', 'information', 'relationship'])
            
            if st.button("Get Strategy", type="primary", key="inf_btn"):
                results = predict_influence_strategy(context, goal, trump_profile) if trump_profile else []
                with col2:
                    st.markdown('<h2 class="section-header">Ranked Strategies</h2>', unsafe_allow_html=True)
                    if results:
                        for i, r in enumerate(results, 1):
                            rank_label = f"[{i}]" if i <= 3 else f"[{i}]"
                            st.markdown(f"**{rank_label} {r['principle']}**: {r['effectiveness']}%")
                            st.caption(r['tactic'])
                            if i < len(results):
                                st.markdown("---")
                    else:
                        st.warning("Feature data not loaded")
        
        # ============= MODEL 6: TRIGGER WORD DETECTOR =============
        elif model_num == 6:
            user_input = st.text_input("Enter a word or phrase:", placeholder="e.g., fake news, winning, disaster...")
            examples = ["fake news", "winning", "disaster", "beautiful", "corrupt", "freedom"]
            st.markdown("**Quick examples:**")
            cols = st.columns(3)
            for i, ex in enumerate(examples):
                if cols[i % 3].button(ex, key=f"ex6_{ex}"):
                    user_input = ex
            
            if user_input:
                result = predict_trigger(user_input, df_triggers)
                with col2:
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    if result['status'] == 'FOUND':
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Trigger Level", result['trigger_level'])
                        with col_m2:
                            st.metric("Trigger Score", f"{result['trigger_score']}/100")
                        with col_m3:
                            st.metric("Emotional Valence", result['valence'])
                        if result.get('words_analyzed'):
                            st.caption(f"Words analyzed: {', '.join(result['words_analyzed'])}")
                    else:
                        st.warning(result.get('message', 'Word not found'))


if __name__ == "__main__":
    main()
