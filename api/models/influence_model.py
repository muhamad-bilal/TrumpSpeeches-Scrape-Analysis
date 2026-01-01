"""
Psychological Influence Model (Model 5)
=======================================
Recommends psychological influence tactics based on Cialdini's principles of persuasion.
"""

from typing import Dict, Any, List, Optional
from .data_loader import get_data_loader


class InfluenceStrategyModel:
    """Recommend influence strategies for interacting with Trump."""

    # Base effectiveness scores for Cialdini's principles
    BASE_EFFECTIVENESS = {
        'Liking (Flattery)': {
            'score': 90,
            'tactic': 'Compliment his achievements, show admiration for his success',
            'why_it_works': 'Trump responds very positively to ego validation and personal praise',
            'examples': [
                'Reference his business success',
                'Praise his negotiation skills',
                'Acknowledge his crowd sizes/popularity'
            ]
        },
        'Scarcity': {
            'score': 75,
            'tactic': 'Offer exclusive opportunities, "only you can do this"',
            'why_it_works': 'Appeals to his sense of uniqueness and special status',
            'examples': [
                'This is a once-in-a-lifetime opportunity',
                'Only someone with your skills could handle this',
                'No one else has access to this deal'
            ]
        },
        'Social Proof': {
            'score': 70,
            'tactic': 'Show poll numbers, crowd sizes, popularity metrics',
            'why_it_works': 'Trump is highly motivated by public perception and winning',
            'examples': [
                'The polls show strong support',
                'This is very popular with your base',
                'Many successful people support this'
            ]
        },
        'Reciprocity': {
            'score': 55,
            'tactic': 'Do him a favor first, then make your ask',
            'why_it_works': 'Creates obligation, but Trump may not always reciprocate',
            'examples': [
                'Public loyalty demonstrations first',
                'Defend him in media before asking',
                'Give something valuable before requesting'
            ]
        },
        'Commitment': {
            'score': 40,
            'tactic': 'Reference his past statements and promises to get consistency',
            'why_it_works': 'Can backfire - Trump values flexibility over consistency',
            'examples': [
                'You said you would do X (risky)',
                'This aligns with your past position',
                'Your supporters expect this based on your promises'
            ]
        },
        'Authority': {
            'score': 35,
            'tactic': 'Position HIM as the expert, not outside experts',
            'why_it_works': 'External authority is often rejected; make him the authority',
            'examples': [
                'You know more about this than anyone',
                'The experts don\'t understand like you do',
                'Your instincts are better than their data'
            ]
        }
    }

    # Context modifiers
    CONTEXT_MODIFIERS = {
        'business': {
            'Scarcity': 1.2,
            'Reciprocity': 1.1,
            'Liking (Flattery)': 1.0
        },
        'political': {
            'Social Proof': 1.3,
            'Liking (Flattery)': 1.1,
            'Commitment': 0.9
        },
        'personal': {
            'Liking (Flattery)': 1.3,
            'Reciprocity': 1.2,
            'Scarcity': 1.1
        },
        'media': {
            'Social Proof': 1.3,
            'Scarcity': 1.1,
            'Authority': 0.8
        },
        'international': {
            'Scarcity': 1.2,
            'Social Proof': 1.1,
            'Authority': 0.7
        }
    }

    # Goal modifiers
    GOAL_MODIFIERS = {
        'agreement': {
            'Liking (Flattery)': 1.1,
            'Social Proof': 1.1
        },
        'favor': {
            'Reciprocity': 1.2,
            'Liking (Flattery)': 1.1
        },
        'information': {
            'Liking (Flattery)': 1.0,
            'Reciprocity': 0.9
        },
        'relationship': {
            'Liking (Flattery)': 1.2,
            'Commitment': 1.1
        },
        'policy_change': {
            'Social Proof': 1.2,
            'Scarcity': 1.1
        }
    }

    def __init__(self):
        """Initialize the influence model."""
        self.data_loader = get_data_loader()

    def predict(
        self,
        context: str,
        goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ranked influence strategies for a given context.

        Args:
            context: The interaction context (business, political, personal, media, international)
            goal: Your goal (agreement, favor, information, relationship, policy_change)

        Returns:
            Ranked list of influence strategies with effectiveness scores
        """
        context_lower = context.lower().strip()
        goal_lower = goal.lower().strip() if goal else None

        # Calculate modified scores
        strategies = []
        for principle, data in self.BASE_EFFECTIVENESS.items():
            score = data['score']

            # Apply context modifier
            if context_lower in self.CONTEXT_MODIFIERS:
                if principle in self.CONTEXT_MODIFIERS[context_lower]:
                    score *= self.CONTEXT_MODIFIERS[context_lower][principle]

            # Apply goal modifier
            if goal_lower and goal_lower in self.GOAL_MODIFIERS:
                if principle in self.GOAL_MODIFIERS[goal_lower]:
                    score *= self.GOAL_MODIFIERS[goal_lower][principle]

            # Cap at 100
            score = min(100, round(score, 1))

            strategies.append({
                'principle': principle,
                'effectiveness': score,
                'tactic': data['tactic'],
                'why_it_works': data['why_it_works'],
                'examples': data['examples']
            })

        # Sort by effectiveness
        strategies.sort(key=lambda x: x['effectiveness'], reverse=True)

        # Add rank
        for i, s in enumerate(strategies, 1):
            s['rank'] = i

        return {
            'status': 'SUCCESS',
            'context': context,
            'goal': goal,
            'strategies': strategies,
            'top_recommendation': strategies[0] if strategies else None,
            'avoid': [s for s in strategies if s['effectiveness'] < 50],
            'general_tips': self._get_general_tips()
        }

    def _get_general_tips(self) -> List[str]:
        """Get general tips for influencing Trump."""
        return [
            "Always let Trump feel like he's winning",
            "Avoid moral arguments - use practical/business logic",
            "Don't cite external experts as authorities",
            "Frame requests as opportunities, not obligations",
            "Timing matters - approach when he's in a good mood",
            "Use simple, direct language - avoid complexity",
            "Acknowledge his achievements before making asks"
        ]

    def get_principle_details(self, principle: str) -> Dict[str, Any]:
        """Get detailed information about a specific influence principle."""
        # Normalize principle name
        principle_key = None
        for key in self.BASE_EFFECTIVENESS.keys():
            if principle.lower() in key.lower():
                principle_key = key
                break

        if principle_key is None:
            return {
                'status': 'NOT_FOUND',
                'message': f'Principle "{principle}" not found',
                'available_principles': list(self.BASE_EFFECTIVENESS.keys())
            }

        data = self.BASE_EFFECTIVENESS[principle_key]
        return {
            'status': 'SUCCESS',
            'principle': principle_key,
            'base_score': data['score'],
            'tactic': data['tactic'],
            'why_it_works': data['why_it_works'],
            'examples': data['examples'],
            'context_modifiers': {
                ctx: mods.get(principle_key, 1.0)
                for ctx, mods in self.CONTEXT_MODIFIERS.items()
            }
        }

    def compare_contexts(self, contexts: List[str]) -> Dict[str, Any]:
        """Compare effectiveness of strategies across different contexts."""
        comparison = {}

        for context in contexts:
            result = self.predict(context)
            comparison[context] = {
                'top_strategy': result['strategies'][0]['principle'],
                'top_effectiveness': result['strategies'][0]['effectiveness'],
                'rankings': {s['principle']: s['rank'] for s in result['strategies']}
            }

        return {
            'status': 'SUCCESS',
            'comparison': comparison
        }

    def get_available_options(self) -> Dict[str, Any]:
        """Get all available contexts, goals, and principles."""
        return {
            'contexts': list(self.CONTEXT_MODIFIERS.keys()),
            'goals': list(self.GOAL_MODIFIERS.keys()),
            'principles': list(self.BASE_EFFECTIVENESS.keys())
        }
