"""
Negotiation Success Predictor (Model 3)
=======================================
Predicts likelihood of successful negotiation with Trump based on topic,
communication style, and strategies.
"""

from typing import Dict, Any, List, Optional
from .data_loader import get_data_loader


class NegotiationSuccessPredictor:
    """Predict negotiation success probability with Trump."""

    # Topic favorability scores (derived from sentiment analysis)
    TOPIC_FAVORABILITY = {
        'trade': 0.75,
        'economy': 0.85,
        'immigration': 0.65,
        'foreign_policy': 0.60,
        'security': 0.80,
        'healthcare': 0.45,
        'environment': 0.30,
        'taxes': 0.80,
        'china': 0.40,
        'russia': 0.55,
        'nato': 0.50,
        'media': 0.20,
        'democrats': 0.15,
        'military': 0.90,
        'jobs': 0.85,
        'border': 0.75,
        'infrastructure': 0.70,
        'veterans': 0.85,
        'law_enforcement': 0.80,
        'education': 0.50
    }

    # Communication style effectiveness
    STYLE_EFFECTIVENESS = {
        'flattering': 0.85,
        'transactional': 0.80,
        'assertive': 0.55,
        'diplomatic': 0.50,
        'confrontational': 0.25,
        'intellectual': 0.35,
        'emotional': 0.40,
        'humble': 0.60,
        'deferential': 0.70,
        'direct': 0.65,
        'indirect': 0.40
    }

    # Strategy multipliers
    STRATEGY_MULTIPLIERS = {
        'show_win': 1.30,
        'media_angle': 1.20,
        'business_frame': 1.25,
        'loyalty_appeal': 1.15,
        'facts_only': 0.80,
        'moral_argument': 0.60,
        'expert_consensus': 0.70,
        'precedent': 0.75,
        'america_first': 1.25,
        'job_creation': 1.20,
        'cost_savings': 1.15,
        'competition_with_china': 1.20
    }

    def __init__(self):
        """Initialize the negotiation predictor."""
        self.data_loader = get_data_loader()

    def predict(
        self,
        topic: str,
        style: str,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict negotiation success probability.

        Args:
            topic: Negotiation topic (e.g., 'trade', 'immigration')
            style: Communication style (e.g., 'flattering', 'assertive')
            strategies: List of strategies to apply (e.g., ['show_win', 'business_frame'])

        Returns:
            Prediction results with probability and recommendations
        """
        if strategies is None:
            strategies = []

        # Normalize inputs
        topic_lower = topic.lower().strip().replace(' ', '_')
        style_lower = style.lower().strip()

        # Get base scores
        topic_score = self.TOPIC_FAVORABILITY.get(topic_lower, 0.50)
        style_score = self.STYLE_EFFECTIVENESS.get(style_lower, 0.50)

        # Calculate base probability (style weighted more heavily)
        base_prob = (topic_score * 0.4 + style_score * 0.6)

        # Apply strategy multipliers
        applied_strategies = []
        final_prob = base_prob

        for strategy in strategies:
            strategy_key = strategy.lower().strip().replace(' ', '_')
            if strategy_key in self.STRATEGY_MULTIPLIERS:
                multiplier = self.STRATEGY_MULTIPLIERS[strategy_key]
                final_prob *= multiplier
                effect = 'BOOST' if multiplier > 1 else 'PENALTY'
                applied_strategies.append({
                    'strategy': strategy,
                    'key': strategy_key,
                    'multiplier': multiplier,
                    'effect': effect
                })

        # Cap probability
        final_prob = min(0.95, max(0.05, final_prob))
        success_percentage = round(final_prob * 100, 1)

        # Determine outcome category
        if final_prob >= 0.75:
            outcome = "HIGH SUCCESS LIKELY"
            confidence = "High"
        elif final_prob >= 0.55:
            outcome = "MODERATE SUCCESS LIKELY"
            confidence = "Medium"
        elif final_prob >= 0.35:
            outcome = "UNCERTAIN - Could go either way"
            confidence = "Low"
        else:
            outcome = "LOW SUCCESS - Consider different approach"
            confidence = "High (negative)"

        # Generate recommendations
        recommendations = self._generate_recommendations(topic_lower, style_lower, final_prob)

        return {
            'status': 'SUCCESS',
            'success_probability': success_percentage,
            'outcome': outcome,
            'confidence': confidence,
            'input': {
                'topic': topic,
                'style': style,
                'strategies': strategies
            },
            'factor_breakdown': {
                'topic_favorability': round(topic_score * 100, 1),
                'style_effectiveness': round(style_score * 100, 1),
                'base_probability': round(base_prob * 100, 1),
                'strategies_applied': applied_strategies
            },
            'recommendations': recommendations,
            'optimal_strategies': self._get_optimal_strategies()
        }

    def _generate_recommendations(
        self,
        topic: str,
        style: str,
        prob: float
    ) -> List[str]:
        """Generate strategic recommendations."""
        recs = []

        # Style recommendations
        if style == 'confrontational':
            recs.append("AVOID confrontation - switch to transactional approach")
        if style == 'intellectual':
            recs.append("Simplify your language - avoid academic tone")
        if style not in ['flattering', 'transactional', 'deferential']:
            recs.append("Consider adding strategic flattery")

        # Topic recommendations
        if topic in ['media', 'democrats', 'environment']:
            recs.append(f"'{topic}' is a hostile topic - reframe if possible")
        if topic in ['china', 'nato']:
            recs.append(f"Frame '{topic}' discussion as 'America First'")

        # Universal recommendations
        recs.append("Always frame proposals as Trump 'winning'")
        recs.append("Highlight business/economic benefits")
        recs.append("Mention potential for positive media coverage")

        if prob < 0.5:
            recs.append("Consider having an ally introduce the topic first")
            recs.append("Start with smaller requests to build momentum")

        return recs[:6]

    def _get_optimal_strategies(self) -> List[str]:
        """Return the optimal negotiation strategies."""
        return [
            "Lead with flattery about his past achievements",
            "Frame everything as a 'deal' he can 'win'",
            "Show how it benefits him personally or politically",
            "Avoid moral arguments - use business logic",
            "Let him feel like the idea was his",
            "Emphasize job creation and economic impact"
        ]

    def compare_approaches(
        self,
        topic: str,
        styles: List[str]
    ) -> Dict[str, Any]:
        """
        Compare effectiveness of different styles for the same topic.

        Args:
            topic: The negotiation topic
            styles: List of styles to compare

        Returns:
            Ranked comparison of styles
        """
        results = []
        for style in styles:
            pred = self.predict(topic, style, ['show_win'])
            results.append({
                'style': style,
                'success_probability': pred['success_probability'],
                'outcome': pred['outcome']
            })

        results.sort(key=lambda x: x['success_probability'], reverse=True)

        return {
            'status': 'SUCCESS',
            'topic': topic,
            'comparison': results,
            'best_style': results[0]['style'] if results else None
        }

    def get_available_options(self) -> Dict[str, Any]:
        """Get all available topics, styles, and strategies."""
        return {
            'topics': list(self.TOPIC_FAVORABILITY.keys()),
            'styles': list(self.STYLE_EFFECTIVENESS.keys()),
            'strategies': list(self.STRATEGY_MULTIPLIERS.keys()),
            'topic_scores': {k: round(v * 100, 1) for k, v in self.TOPIC_FAVORABILITY.items()},
            'style_scores': {k: round(v * 100, 1) for k, v in self.STYLE_EFFECTIVENESS.items()}
        }
