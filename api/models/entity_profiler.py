"""
Entity Reaction Profiler (Model 1)
==================================
Predicts Trump's emotional reaction to entities (people, organizations, countries).
"""

from typing import Dict, Any, Optional
import pandas as pd
from .data_loader import get_data_loader


class EntityReactionProfiler:
    """Predict Trump's reaction to entities based on speech analysis."""

    def __init__(self):
        """Initialize the entity profiler."""
        self.data_loader = get_data_loader()

    def predict(self, entity_name: str) -> Dict[str, Any]:
        """
        Predict Trump's reaction to a given entity.

        Args:
            entity_name: Name of the entity (person, country, organization)

        Returns:
            Dictionary containing prediction results
        """
        df_profiles = self.data_loader.entity_profiles
        baseline = self.data_loader.baseline_sentiment

        if df_profiles is None:
            return {
                'status': 'ERROR',
                'message': 'Entity profiles data not loaded'
            }

        entity_lower = entity_name.lower().strip()

        # Search for entity (exact or partial match)
        matches = df_profiles[
            df_profiles['entity_name'].str.lower().str.contains(entity_lower, na=False)
        ]

        if len(matches) == 0:
            return {
                'status': 'NOT_FOUND',
                'entity': entity_name,
                'message': f"No data found for '{entity_name}'. Try a different entity."
            }

        # Get the best match (first one)
        match = matches.iloc[0]

        # Calculate sentiment relative to baseline
        raw_sentiment = float(match.get('avg_sentiment', 0.5))
        centered_sentiment = raw_sentiment - baseline

        # Calculate emotion ratios
        neg_emotions = (
            float(match.get('avg_anger', 0)) +
            float(match.get('avg_fear', 0)) +
            float(match.get('avg_disgust', 0))
        )
        pos_emotions = (
            float(match.get('avg_joy', 0)) +
            float(match.get('avg_trust', 0))
        )
        total_emotion = neg_emotions + pos_emotions
        emotion_ratio = neg_emotions / total_emotion if total_emotion > 0 else 0.5

        # Determine sentiment label
        if emotion_ratio > 0.35 or centered_sentiment < -0.03:
            sentiment_label = 'NEGATIVE'
        elif emotion_ratio < 0.30 and centered_sentiment > 0.03:
            sentiment_label = 'POSITIVE'
        else:
            sentiment_label = 'NEUTRAL'

        # Determine volatility
        volatility = float(match.get('volatility', match.get('std_sentiment', 0)))
        if volatility > 0.15:
            volatility_label = 'HIGH'
        elif volatility < 0.05:
            volatility_label = 'LOW'
        else:
            volatility_label = 'MEDIUM'

        # Determine reaction type
        if sentiment_label == 'NEGATIVE':
            reaction_type = 'ATTACK_MODE' if volatility_label == 'HIGH' else 'CRITICISM_MODE'
        elif sentiment_label == 'POSITIVE':
            reaction_type = 'CELEBRATION_MODE' if volatility_label != 'HIGH' else 'UNPREDICTABLE'
        elif volatility_label == 'HIGH':
            reaction_type = 'UNPREDICTABLE'
        else:
            reaction_type = 'NEUTRAL_MODE'

        # Get dominant emotion
        emotion_cols = ['avg_anger', 'avg_fear', 'avg_joy', 'avg_sadness',
                        'avg_surprise', 'avg_disgust', 'avg_trust', 'avg_anticipation']
        emotions = {col.replace('avg_', ''): float(match.get(col, 0)) for col in emotion_cols}
        dominant_emotion = max(emotions, key=emotions.get) if emotions else 'unknown'

        return {
            'status': 'FOUND',
            'entity': str(match.get('entity_name', entity_name)),
            'entity_type': str(match.get('entity_type', 'UNKNOWN')),
            'speech_count': int(match.get('speech_count', 0)),
            'sentiment': {
                'label': sentiment_label,
                'raw_score': round(raw_sentiment, 4),
                'centered_score': round(centered_sentiment, 4)
            },
            'volatility': {
                'label': volatility_label,
                'score': round(volatility, 4)
            },
            'reaction_type': reaction_type,
            'dominant_emotion': dominant_emotion,
            'emotions': {k: round(v, 4) for k, v in emotions.items()},
            'rhetorical_intensity': round(float(match.get('rhetorical_intensity', 0)), 2)
        }

    def get_top_entities(self, entity_type: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """
        Get top entities by speech count.

        Args:
            entity_type: Filter by entity type (PERSON, ORG, GPE, etc.)
            limit: Maximum number of entities to return

        Returns:
            Dictionary with top entities
        """
        df_profiles = self.data_loader.entity_profiles

        if df_profiles is None:
            return {'status': 'ERROR', 'message': 'Data not loaded'}

        df = df_profiles.copy()

        if entity_type:
            df = df[df['entity_type'] == entity_type.upper()]

        df = df.nlargest(limit, 'speech_count')

        entities = []
        for _, row in df.iterrows():
            entities.append({
                'entity': row.get('entity_name'),
                'type': row.get('entity_type'),
                'speech_count': int(row.get('speech_count', 0)),
                'avg_sentiment': round(float(row.get('avg_sentiment', 0)), 4)
            })

        return {
            'status': 'SUCCESS',
            'count': len(entities),
            'entities': entities
        }

    def search_entities(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for entities matching a query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Dictionary with matching entities
        """
        df_profiles = self.data_loader.entity_profiles

        if df_profiles is None:
            return {'status': 'ERROR', 'message': 'Data not loaded'}

        query_lower = query.lower()
        matches = df_profiles[
            df_profiles['entity_name'].str.lower().str.contains(query_lower, na=False)
        ].head(limit)

        results = []
        for _, row in matches.iterrows():
            results.append({
                'entity': row.get('entity_name'),
                'type': row.get('entity_type'),
                'speech_count': int(row.get('speech_count', 0))
            })

        return {
            'status': 'SUCCESS',
            'query': query,
            'count': len(results),
            'results': results
        }
