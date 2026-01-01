"""
Personality Compatibility Predictor (Model 2)
=============================================
Predicts how Trump will respond to different personality types using Big Five traits.
"""

from typing import Dict, Any, Optional
from .data_loader import get_data_loader


class PersonalityCompatibilityPredictor:
    """Predict compatibility between Trump and other personalities."""

    def __init__(self):
        """Initialize the personality predictor."""
        self.data_loader = get_data_loader()
        self._trump_profile: Optional[Dict[str, float]] = None

    @property
    def trump_profile(self) -> Dict[str, float]:
        """Get Trump's Big Five personality profile derived from speech features."""
        if self._trump_profile is None:
            self._trump_profile = self._compute_trump_profile()
        return self._trump_profile

    def _compute_trump_profile(self) -> Dict[str, float]:
        """Compute Trump's Big Five personality from speech features."""
        df = self.data_loader.features_df

        if df is None:
            # Return default profile if data not available
            return {
                'openness': 35.0,
                'conscientiousness': 72.0,
                'extraversion': 85.0,
                'agreeableness': 28.0,
                'neuroticism': 45.0
            }

        # Extract relevant features (with defaults)
        features = {
            'i_we_ratio': df['pronoun_i_we_ratio'].mean() if 'pronoun_i_we_ratio' in df.columns else 0.7,
            'first_singular': df['pronoun_first_singular'].mean() if 'pronoun_first_singular' in df.columns else 150,
            'lexical_diversity': df['lexical_diversity'].mean() if 'lexical_diversity' in df.columns else 0.16,
            'readability': df['readability_flesch_reading_ease'].mean() if 'readability_flesch_reading_ease' in df.columns else 75,
            'avg_sentence_length': df['avg_sentence_length'].mean() if 'avg_sentence_length' in df.columns else 11,
            'modal_will': df['modal_will'].mean() if 'modal_will' in df.columns else 15,
            'certainty_markers': df['certainty_markers'].mean() if 'certainty_markers' in df.columns else 10,
            'superlatives': df['superlative_count'].mean() if 'superlative_count' in df.columns else 20,
            'sentiment_pos': df['sentiment_pos'].mean() if 'sentiment_pos' in df.columns else 0.12,
            'sentiment_neg': df['sentiment_neg'].mean() if 'sentiment_neg' in df.columns else 0.08,
            'sentiment_variance': df['sentiment_variance'].mean() if 'sentiment_variance' in df.columns else 0.1,
            'exclamations': df['exclamation_count'].mean() if 'exclamation_count' in df.columns else 1,
            'all_caps': df['all_caps_words'].mean() if 'all_caps_words' in df.columns else 5,
            'affiliation_words': df['affiliation_words'].mean() if 'affiliation_words' in df.columns else 15,
            'power_affiliation_ratio': df['power_affiliation_ratio'].mean() if 'power_affiliation_ratio' in df.columns else 0.5,
        }

        # Compute Big Five scores
        openness = min(100, max(0, (
            features['lexical_diversity'] * 100 +
            (100 - features['readability']) * 0.5 +
            features['avg_sentence_length'] * 2
        ) / 3 * 2))

        conscientiousness = min(100, max(0, (
            features['modal_will'] * 3 +
            features['certainty_markers'] * 5 +
            features['superlatives'] * 2
        ) / 10 * 4))

        extraversion = min(100, max(0, (
            features['first_singular'] / 10 +
            features['sentiment_pos'] * 100 +
            features['exclamations'] * 5 +
            features['all_caps'] * 2 +
            features['superlatives'] * 3
        ) / 5 * 2))

        agreeableness = min(100, max(0, (
            (1 / (features['i_we_ratio'] + 0.1)) * 10 +
            features['affiliation_words'] * 3 +
            (1 - features['sentiment_neg']) * 50 +
            (100 - features['power_affiliation_ratio'] * 100) * 0.5
        ) / 4))

        neuroticism = min(100, max(0, (
            features['sentiment_neg'] * 200 +
            features['sentiment_variance'] * 100 +
            features['all_caps'] * 1 +
            (100 - features['certainty_markers'] * 5)
        ) / 4))

        return {
            'openness': round(openness, 1),
            'conscientiousness': round(conscientiousness, 1),
            'extraversion': round(extraversion, 1),
            'agreeableness': round(agreeableness, 1),
            'neuroticism': round(neuroticism, 1)
        }

    def predict(self, personality: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict compatibility with Trump.

        Args:
            personality: Dictionary with Big Five traits (0-100 each):
                - openness
                - conscientiousness
                - extraversion
                - agreeableness
                - neuroticism

        Returns:
            Compatibility prediction results
        """
        # Validate input
        required_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in required_traits:
            if trait not in personality:
                return {
                    'status': 'ERROR',
                    'message': f'Missing required trait: {trait}'
                }
            personality[trait] = max(0, min(100, float(personality[trait])))

        trump = self.trump_profile

        # Calculate dominance scores
        trump_dominance = (trump['extraversion'] + (100 - trump['agreeableness'])) / 2
        other_dominance = (personality['extraversion'] + (100 - personality['agreeableness'])) / 2

        # Score components
        factors = {}
        total_score = 0

        # Dominance dynamic (max 25 points)
        dominance_diff = trump_dominance - other_dominance
        if dominance_diff > 30:
            dom_score = 25
            dom_note = "Deferential - Trump will appreciate respect"
        elif dominance_diff > 10:
            dom_score = 20
            dom_note = "Respectful but not pushover"
        elif dominance_diff > -10:
            dom_score = 10
            dom_note = "Power struggle likely"
        else:
            dom_score = 5
            dom_note = "Will trigger competitive response"
        factors['dominance'] = {'score': dom_score, 'note': dom_note}
        total_score += dom_score

        # Agreeableness factor (max 25 points)
        if personality['agreeableness'] > 70:
            agree_score = 25
            agree_note = "Accommodating - will avoid conflict"
        elif personality['agreeableness'] > 50:
            agree_score = 18
            agree_note = "Diplomatic - can manage tensions"
        elif personality['agreeableness'] > 30:
            agree_score = 10
            agree_note = "Both competitive - friction expected"
        else:
            agree_score = 5
            agree_note = "Conflict inevitable"
        factors['agreeableness'] = {'score': agree_score, 'note': agree_note}
        total_score += agree_score

        # Conscientiousness factor (max 20 points)
        if personality['conscientiousness'] > 60:
            consc_score = 20
            consc_note = "Organized and reliable - earns respect"
        elif personality['conscientiousness'] > 40:
            consc_score = 15
            consc_note = "Moderately organized"
        else:
            consc_score = 8
            consc_note = "May be seen as unreliable"
        factors['conscientiousness'] = {'score': consc_score, 'note': consc_note}
        total_score += consc_score

        # Neuroticism factor (max 15 points)
        if personality['neuroticism'] < 30:
            neuro_score = 15
            neuro_note = "Emotionally stable - won't be rattled"
        elif personality['neuroticism'] < 50:
            neuro_score = 12
            neuro_note = "Generally stable"
        else:
            neuro_score = 7
            neuro_note = "Emotional vulnerability may be exploited"
        factors['neuroticism'] = {'score': neuro_score, 'note': neuro_note}
        total_score += neuro_score

        # Status/competence proxy (max 15 points)
        status_proxy = (personality['extraversion'] + personality['conscientiousness']) / 2
        if status_proxy > 60:
            status_score = 15
            status_note = "Projects competence and success"
        elif status_proxy > 40:
            status_score = 10
            status_note = "Average presence"
        else:
            status_score = 5
            status_note = "May be dismissed"
        factors['status'] = {'score': status_score, 'note': status_note}
        total_score += status_score

        # Determine response type
        if total_score >= 75:
            response_type = "COOPERATIVE"
            description = "Trump will likely be friendly and receptive"
        elif total_score >= 55:
            response_type = "TRANSACTIONAL"
            description = "Trump will engage if there's something in it for him"
        elif total_score >= 40:
            if other_dominance > trump_dominance - 10:
                response_type = "COMPETITIVE"
                description = "Trump will try to assert dominance"
            else:
                response_type = "DISMISSIVE"
                description = "Trump may ignore or belittle"
        else:
            if other_dominance > trump_dominance:
                response_type = "HOSTILE"
                description = "Trump will likely attack"
            else:
                response_type = "CONTEMPTUOUS"
                description = "Trump will show disdain"

        return {
            'status': 'SUCCESS',
            'compatibility_score': total_score,
            'response_type': response_type,
            'description': description,
            'your_profile': personality,
            'trump_profile': trump,
            'factors': factors,
            'dominance_analysis': {
                'trump_dominance': round(trump_dominance, 1),
                'your_dominance': round(other_dominance, 1),
                'difference': round(dominance_diff, 1)
            }
        }

    def get_trump_profile(self) -> Dict[str, Any]:
        """Get Trump's computed personality profile."""
        return {
            'status': 'SUCCESS',
            'profile': self.trump_profile,
            'interpretation': {
                'openness': 'Low - Prefers familiar patterns, skeptical of new ideas',
                'conscientiousness': 'High - Goal-oriented, uses certainty language',
                'extraversion': 'Very High - Dominant, expressive, attention-seeking',
                'agreeableness': 'Very Low - Competitive, confrontational, power-focused',
                'neuroticism': 'Moderate - Some emotional reactivity, especially to criticism'
            }
        }
