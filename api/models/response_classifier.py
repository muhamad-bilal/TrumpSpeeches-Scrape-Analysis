"""
Response Classifier (Model 4)
=============================
ML-based classifier that predicts Trump's response type based on linguistic features.
"""

from typing import Dict, Any, Tuple


class ResponseClassifier:
    """Classify Trump's response type based on linguistic features."""

    # Response type definitions
    RESPONSE_TYPES = {
        'ATTACK': {
            'description': 'Aggressive response with criticism or personal attacks',
            'typical_triggers': 'High negative ratio, high power language',
            'examples': ['Crooked Hillary', 'Fake news media', 'Total disaster']
        },
        'PRAISE': {
            'description': 'Positive, complimentary response',
            'typical_triggers': 'High positive ratio, low power language',
            'examples': ['Tremendous success', 'Great job', 'The best']
        },
        'NEGOTIATE': {
            'description': 'Deal-making, transactional response',
            'typical_triggers': 'Balanced sentiment, moderate power/certainty',
            'examples': ['We can work something out', 'Let me tell you what we can do']
        },
        'DEFLECT': {
            'description': 'Avoidance, topic change, or blame shifting',
            'typical_triggers': 'Low certainty, mixed sentiment',
            'examples': ['Many people are saying', 'We\'ll see what happens']
        },
        'NEUTRAL': {
            'description': 'Measured, non-emotional response',
            'typical_triggers': 'Neutral sentiment across all metrics',
            'examples': ['We\'re looking at it', 'Time will tell']
        }
    }

    def __init__(self):
        """Initialize the response classifier."""
        pass

    def classify(
        self,
        sentiment: float,
        neg_ratio: float,
        pos_ratio: float,
        power_ratio: float,
        certainty: float
    ) -> Dict[str, Any]:
        """
        Classify the expected response type.

        Args:
            sentiment: Overall sentiment score (-1 to 1 or 0 to 1)
            neg_ratio: Ratio of negative words (0 to 1)
            pos_ratio: Ratio of positive words (0 to 1)
            power_ratio: Power/affiliation language ratio (0 to 1)
            certainty: Certainty level in language (0 to 1)

        Returns:
            Classification results with confidence
        """
        # Normalize inputs to 0-1 range if needed
        if sentiment < 0:
            sentiment = (sentiment + 1) / 2  # Convert -1,1 to 0,1

        # Run classification rules
        response_type, confidence = self._classify_rules(
            sentiment, neg_ratio, pos_ratio, power_ratio, certainty
        )

        # Get probabilities for all types
        probabilities = self._calculate_probabilities(
            sentiment, neg_ratio, pos_ratio, power_ratio, certainty
        )

        return {
            'status': 'SUCCESS',
            'response_type': response_type,
            'confidence': confidence,
            'description': self.RESPONSE_TYPES[response_type]['description'],
            'input_features': {
                'sentiment': round(sentiment, 3),
                'neg_ratio': round(neg_ratio, 3),
                'pos_ratio': round(pos_ratio, 3),
                'power_ratio': round(power_ratio, 3),
                'certainty': round(certainty, 3)
            },
            'all_probabilities': probabilities,
            'interpretation': self._get_interpretation(response_type),
            'typical_examples': self.RESPONSE_TYPES[response_type]['examples']
        }

    def _classify_rules(
        self,
        sentiment: float,
        neg_ratio: float,
        pos_ratio: float,
        power_ratio: float,
        certainty: float
    ) -> Tuple[str, int]:
        """Apply rule-based classification."""

        # ATTACK: High negativity + high power
        if neg_ratio > 0.15 and power_ratio > 0.6:
            return 'ATTACK', 85

        # ATTACK: Moderate negativity + low sentiment
        if neg_ratio > 0.12 and sentiment < 0.3:
            return 'ATTACK', 75

        # PRAISE: High positivity + high sentiment
        if pos_ratio > 0.2 and sentiment > 0.8:
            return 'PRAISE', 80

        # PRAISE: Moderate positivity + low power
        if pos_ratio > 0.15 and power_ratio < 0.4:
            return 'PRAISE', 70

        # NEGOTIATE: Balanced power + high certainty
        if 0.4 <= power_ratio <= 0.6 and certainty > 0.3:
            return 'NEGOTIATE', 65

        # NEGOTIATE: Positive sentiment + low negativity
        if sentiment > 0.5 and neg_ratio < 0.08:
            return 'NEGOTIATE', 60

        # DEFLECT: Low certainty + mixed sentiment
        if certainty < 0.2 and 0.3 < sentiment < 0.7:
            return 'DEFLECT', 55

        # NEUTRAL: Default
        return 'NEUTRAL', 50

    def _calculate_probabilities(
        self,
        sentiment: float,
        neg_ratio: float,
        pos_ratio: float,
        power_ratio: float,
        certainty: float
    ) -> Dict[str, float]:
        """Calculate probability scores for each response type."""

        # Attack probability
        attack_prob = (neg_ratio * 2 + power_ratio + (1 - sentiment)) / 4 * 100

        # Praise probability
        praise_prob = (pos_ratio * 2 + sentiment + (1 - power_ratio)) / 4 * 100

        # Negotiate probability
        balance = 1 - abs(power_ratio - 0.5) * 2
        negotiate_prob = (balance + certainty + abs(sentiment - 0.5)) / 3 * 100

        # Deflect probability
        deflect_prob = ((1 - certainty) + (1 - abs(sentiment - 0.5)) + (1 - power_ratio)) / 3 * 100

        # Neutral probability (inverse of extremes)
        neutral_prob = 100 - max(attack_prob, praise_prob, negotiate_prob, deflect_prob) + 20

        # Normalize
        total = attack_prob + praise_prob + negotiate_prob + deflect_prob + neutral_prob
        factor = 100 / total if total > 0 else 1

        return {
            'ATTACK': round(attack_prob * factor, 1),
            'PRAISE': round(praise_prob * factor, 1),
            'NEGOTIATE': round(negotiate_prob * factor, 1),
            'DEFLECT': round(deflect_prob * factor, 1),
            'NEUTRAL': round(neutral_prob * factor, 1)
        }

    def _get_interpretation(self, response_type: str) -> str:
        """Get interpretation guidance for the response type."""
        interpretations = {
            'ATTACK': (
                "Expect aggressive language, personal attacks, or criticism. "
                "Trump may use nicknames, question credibility, or escalate conflict."
            ),
            'PRAISE': (
                "Expect positive, complimentary language. "
                "Trump will likely use superlatives and express approval or admiration."
            ),
            'NEGOTIATE': (
                "Expect deal-making language. "
                "Trump will likely focus on what he can get and frame things transactionally."
            ),
            'DEFLECT': (
                "Expect topic changes, vague answers, or blame shifting. "
                "Trump may use phrases like 'many people say' or 'we'll see'."
            ),
            'NEUTRAL': (
                "Expect measured, non-committal response. "
                "Trump will likely stay on script without strong emotional content."
            )
        }
        return interpretations.get(response_type, "Unknown response type")

    def get_feature_importance(self) -> Dict[str, Any]:
        """Explain which features matter most for each response type."""
        return {
            'ATTACK': {
                'primary': ['neg_ratio', 'power_ratio'],
                'secondary': ['sentiment (low)'],
                'weight': 'Negativity and power language are key indicators'
            },
            'PRAISE': {
                'primary': ['pos_ratio', 'sentiment'],
                'secondary': ['power_ratio (low)'],
                'weight': 'Positivity and low aggression indicate praise'
            },
            'NEGOTIATE': {
                'primary': ['certainty', 'power_ratio (balanced)'],
                'secondary': ['sentiment (positive)'],
                'weight': 'Confidence without aggression signals negotiation'
            },
            'DEFLECT': {
                'primary': ['certainty (low)', 'sentiment (neutral)'],
                'secondary': ['power_ratio (low)'],
                'weight': 'Uncertainty and neutrality indicate deflection'
            },
            'NEUTRAL': {
                'primary': ['all metrics balanced'],
                'secondary': [],
                'weight': 'No strong indicators in any direction'
            }
        }
