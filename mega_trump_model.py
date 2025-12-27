"""
🎯 Mega Trump Model - Unified Predictive System
==============================================
Combines all 6 sub-models into one comprehensive predictor:
1. Entity Reaction Profiler
2. Personality Compatibility Predictor
3. Negotiation Success Predictor
4. Response Classifier (ML)
5. Psychological Influence Model
6. Trigger Word Detector
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any


class MegaTrumpPredictor:
    """
    Unified predictive model that combines all 6 sub-models to provide
    comprehensive behavioral predictions about Trump.
    """
    
    def __init__(self, 
                 df_features=None,
                 df_profiles=None,
                 df_triggers=None,
                 baseline=None,
                 model_functions=None):
        """
        Initialize the Mega Trump Model.
        
        Parameters:
        -----------
        df_features : pd.DataFrame
            Speech features data
        df_profiles : pd.DataFrame
            Entity reaction profiles
        df_triggers : pd.DataFrame
            Trigger word data
        baseline : float
            Sentiment baseline for entity reactions
        model_functions : dict
            Dictionary containing all individual model functions
        """
        self.df_features = df_features
        self.df_profiles = df_profiles
        self.df_triggers = df_triggers
        self.baseline = baseline
        self.model_funcs = model_functions or {}
        
        # Compute Trump's personality profile if features available
        self.trump_profile = None
        if df_features is not None and 'compute_trump_big_five' in self.model_funcs:
            self.trump_profile = self.model_funcs['compute_trump_big_five'](df_features)
    
    def predict_comprehensive(self,
                              entity: Optional[str] = None,
                              personality_traits: Optional[Dict] = None,
                              topic: Optional[str] = None,
                              style: Optional[str] = None,
                              strategies: Optional[List[str]] = None,
                              linguistic_features: Optional[Dict] = None,
                              context: Optional[str] = None,
                              goal: Optional[str] = None,
                              words: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified prediction combining all applicable sub-models.
        
        Parameters:
        -----------
        entity : str, optional
            Entity name (person, country, organization)
        personality_traits : dict, optional
            Big Five personality traits (openness, conscientiousness, etc.)
        topic : str, optional
            Negotiation topic
        style : str, optional
            Communication style
        strategies : list, optional
            Negotiation strategies
        linguistic_features : dict, optional
            Linguistic features (sentiment, power_ratio, etc.)
        context : str, optional
            Context for influence strategy (business, political, etc.)
        goal : str, optional
            Goal for influence strategy (agreement, favor, etc.)
        words : str, optional
            Words/phrases to check for triggers
        
        Returns:
        --------
        dict : Comprehensive prediction with:
            - overall_prediction: Main behavioral prediction
            - confidence: Confidence score (0-100)
            - category_scores: Scores for each behavioral category
            - sub_model_contributions: Individual model predictions
            - recommendations: Unified recommendations
            - trump_profile: Trump's personality profile
        """
        
        predictions = {}
        weights = {}
        
        # ========== MODEL 1: Entity Reaction ==========
        if entity and self.df_profiles is not None and 'predict_entity_reaction' in self.model_funcs:
            try:
                pred1 = self.model_funcs['predict_entity_reaction'](
                    entity, self.df_profiles, self.baseline
                )
                if pred1.get('status') == 'FOUND':
                    predictions['entity_reaction'] = pred1
                    weights['entity_reaction'] = 0.25  # 25% weight
            except Exception as e:
                print(f"Entity reaction model error: {e}")
        
        # ========== MODEL 2: Personality Compatibility ==========
        if personality_traits and self.trump_profile and 'predict_compatibility' in self.model_funcs:
            try:
                pred2 = self.model_funcs['predict_compatibility'](
                    personality_traits, self.trump_profile
                )
                predictions['personality'] = pred2
                weights['personality'] = 0.20  # 20% weight
            except Exception as e:
                print(f"Personality compatibility model error: {e}")
        
        # ========== MODEL 3: Negotiation Success ==========
        if topic and style and 'predict_negotiation_success' in self.model_funcs:
            try:
                pred3 = self.model_funcs['predict_negotiation_success'](
                    topic, style, strategies or []
                )
                predictions['negotiation'] = pred3
                weights['negotiation'] = 0.20  # 20% weight
            except Exception as e:
                print(f"Negotiation model error: {e}")
        
        # ========== MODEL 4: Response Classifier ==========
        if linguistic_features and 'classify_response' in self.model_funcs:
            try:
                pred4 = self.model_funcs['classify_response'](**linguistic_features)
                predictions['response_type'] = {
                    'response': pred4[0],
                    'confidence': pred4[1]
                }
                weights['response_type'] = 0.15  # 15% weight
            except Exception as e:
                print(f"Response classifier error: {e}")
        
        # ========== MODEL 5: Influence Strategy ==========
        if context and goal and self.trump_profile and 'predict_influence_strategy' in self.model_funcs:
            try:
                pred5 = self.model_funcs['predict_influence_strategy'](
                    context, goal, self.trump_profile
                )
                predictions['influence'] = pred5
                weights['influence'] = 0.10  # 10% weight
            except Exception as e:
                print(f"Influence strategy model error: {e}")
        
        # ========== MODEL 6: Trigger Word Detector ==========
        if words and self.df_triggers is not None and 'predict_trigger' in self.model_funcs:
            try:
                pred6 = self.model_funcs['predict_trigger'](words, self.df_triggers)
                if pred6.get('status') == 'FOUND':
                    predictions['triggers'] = pred6
                    weights['triggers'] = 0.10  # 10% weight
            except Exception as e:
                print(f"Trigger word model error: {e}")
        
        # Combine all predictions
        if not predictions:
            return {
                'overall_prediction': 'INSUFFICIENT_DATA',
                'confidence': 0,
                'message': 'No models could be run with provided inputs',
                'sub_model_contributions': {}
            }
        
        return self._combine_predictions(predictions, weights)
    
    def _combine_predictions(self, predictions: Dict, weights: Dict) -> Dict[str, Any]:
        """
        Combine all sub-model predictions into unified output.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary of sub-model predictions
        weights : dict
            Dictionary of weights for each sub-model
        
        Returns:
        --------
        dict : Unified prediction result
        """
        
        # Map all predictions to common behavioral categories
        response_mapping = {
            'COOPERATIVE': [
                'COOPERATIVE', 'CELEBRATION_MODE', 'PRAISE', 
                'HIGH SUCCESS LIKELY', 'POSITIVE', 'ACHIEVEMENT_MODE'
            ],
            'TRANSACTIONAL': [
                'TRANSACTIONAL', 'NEGOTIATE', 'MODERATE SUCCESS LIKELY',
                'NEUTRAL_MODE'
            ],
            'COMPETITIVE': [
                'COMPETITIVE', 'ATTACK_MODE', 'CRITICISM_MODE',
                'ATTACK', 'UNPREDICTABLE'
            ],
            'HOSTILE': [
                'HOSTILE', 'CONTEMPTUOUS', 'LOW SUCCESS',
                'NEGATIVE', 'DEFLECT'
            ],
            'NEUTRAL': [
                'NEUTRAL', 'UNCERTAIN', 'NEUTRAL_MODE'
            ]
        }
        
        # Score each category based on sub-model predictions
        category_scores = {cat: 0.0 for cat in response_mapping.keys()}
        
        # Use FIXED weights (not relative to active models)
        # This ensures confidence reflects actual model agreement
        fixed_weights = {
            'entity_reaction': 0.25,
            'personality': 0.20,
            'negotiation': 0.20,
            'response_type': 0.15,
            'influence': 0.10,
            'triggers': 0.10
        }
        
        # Process each sub-model prediction
        for model_name, pred in predictions.items():
            # Use fixed weight, not relative weight
            weight = fixed_weights.get(model_name, 0.0)
            
            # Extract response type from prediction
            response = self._extract_response_type(pred, model_name)
            
            # Add weight to appropriate category
            matched = False
            for category, matches in response_mapping.items():
                if any(m in str(response).upper() for m in matches):
                    category_scores[category] += weight
                    matched = True
                    break
            
            # If no match found, add to NEUTRAL
            if not matched:
                category_scores['NEUTRAL'] += weight
        
        # Normalize category scores to sum to 1.0 (100%)
        total_score = sum(category_scores.values())
        if total_score > 0:
            category_scores = {k: v / total_score for k, v in category_scores.items()}
        
        # Determine overall prediction (category with highest score)
        overall_response = max(category_scores, key=category_scores.get)
        
        # Calculate confidence based on:
        # 1. How much the winning category dominates
        # 2. How many models contributed
        winning_score = category_scores[overall_response]
        num_models = len(predictions)
        
        # Base confidence on dominance + model count
        # More models = higher max confidence possible
        # More dominance = higher confidence
        max_possible_confidence = min(95, 50 + (num_models * 7.5))  # 1 model: 57.5%, 6 models: 95%
        confidence = winning_score * max_possible_confidence
        
        # Generate unified recommendations
        recommendations = self._generate_unified_recommendations(
            predictions, overall_response, category_scores
        )
        
        return {
            'overall_prediction': overall_response,
            'confidence': round(confidence, 1),
            'category_scores': {k: round(v * 100, 1) for k, v in category_scores.items()},
            'sub_model_contributions': predictions,
            'model_weights': {k: fixed_weights.get(k, 0) for k in predictions.keys()},
            'recommendations': recommendations,
            'trump_profile': self.trump_profile,
            'num_models_used': len(predictions)
        }
    
    def _extract_response_type(self, prediction: Dict, model_name: str) -> str:
        """
        Extract response type from a sub-model prediction.
        
        Parameters:
        -----------
        prediction : dict
            Sub-model prediction
        model_name : str
            Name of the sub-model
        
        Returns:
        --------
        str : Response type string
        """
        
        try:
            # Entity reaction model
            if model_name == 'entity_reaction':
                return prediction.get('reaction_type', 'NEUTRAL')
            
            # Personality compatibility model
            elif model_name == 'personality':
                return prediction.get('response_type', 'NEUTRAL')
            
            # Negotiation model
            elif model_name == 'negotiation':
                outcome = prediction.get('outcome', 'NEUTRAL')
                # Map negotiation outcomes more carefully
                if 'HIGH SUCCESS' in str(outcome).upper():
                    return 'COOPERATIVE'
                elif 'LOW SUCCESS' in str(outcome).upper():
                    return 'HOSTILE'
                elif 'MODERATE' in str(outcome).upper():
                    return 'TRANSACTIONAL'
                return outcome
            
            # Response classifier
            elif model_name == 'response_type':
                return prediction.get('response', 'NEUTRAL')
            
            # Influence strategy (get top strategy)
            elif model_name == 'influence':
                if isinstance(prediction, list) and len(prediction) > 0:
                    # Influence doesn't directly map to response type
                    # Return based on effectiveness
                    top_effectiveness = prediction[0].get('effectiveness', 0)
                    if top_effectiveness > 80:
                        return 'COOPERATIVE'
                    elif top_effectiveness < 40:
                        return 'HOSTILE'
                    return 'TRANSACTIONAL'
                return 'NEUTRAL'
            
            # Trigger words
            elif model_name == 'triggers':
                trigger_level = prediction.get('trigger_level', 'LOW')
                valence = prediction.get('valence', 'NEUTRAL')
                if trigger_level == 'HIGH' and valence == 'NEGATIVE':
                    return 'ATTACK'
                elif trigger_level == 'HIGH' and valence == 'POSITIVE':
                    return 'PRAISE'
                return 'NEUTRAL'
        
        except Exception as e:
            print(f"Error extracting response type from {model_name}: {e}")
        
        return 'NEUTRAL'
    
    def _generate_unified_recommendations(self,
                                         predictions: Dict,
                                         overall_response: str,
                                         category_scores: Dict) -> List[str]:
        """
        Generate unified recommendations combining insights from all models.
        
        Parameters:
        -----------
        predictions : dict
            All sub-model predictions
        overall_response : str
            Overall predicted response type
        category_scores : dict
            Scores for each behavioral category
        
        Returns:
        --------
        list : List of recommendation strings
        """
        
        recommendations = []
        
        # Base recommendations on overall prediction
        if overall_response == 'COOPERATIVE':
            recommendations.append("✅ Trump is likely to be receptive - this is a good time to engage")
            recommendations.append("💡 Maintain positive framing and acknowledge his achievements")
        
        elif overall_response == 'TRANSACTIONAL':
            recommendations.append("💼 Frame your proposal as mutually beneficial")
            recommendations.append("💡 Show what's in it for him personally or politically")
            recommendations.append("💡 Use business language and emphasize 'winning'")
        
        elif overall_response == 'COMPETITIVE':
            recommendations.append("⚠️ Expect power dynamics - avoid direct challenges")
            recommendations.append("💡 Use indirect influence rather than confrontation")
            recommendations.append("💡 Let him feel in control while guiding the outcome")
        
        elif overall_response == 'HOSTILE':
            recommendations.append("🚨 High risk of conflict - consider delaying or reframing")
            recommendations.append("💡 Consider having a mutual ally make the introduction")
            recommendations.append("💡 Start with smaller, less contentious requests")
            recommendations.append("💡 Avoid topics that trigger negative reactions")
        
        # Add model-specific recommendations
        if 'personality' in predictions:
            compat_score = predictions['personality'].get('compatibility_score', 0)
            if compat_score < 50:
                recommendations.append("👤 Personality mismatch detected - adjust your approach")
        
        if 'negotiation' in predictions:
            success_prob = predictions['negotiation'].get('success_probability', 0)
            if success_prob < 50:
                recommendations.append("🤝 Low negotiation success probability - change topic or style")
            elif success_prob > 75:
                recommendations.append("🎯 High success probability - proceed with confidence")
        
        if 'influence' in predictions and isinstance(predictions['influence'], list):
            top_strategy = predictions['influence'][0] if predictions['influence'] else None
            if top_strategy:
                recommendations.append(f"🧠 Best influence tactic: {top_strategy.get('principle', 'Unknown')}")
                recommendations.append(f"   → {top_strategy.get('tactic', '')}")
        
        if 'triggers' in predictions:
            trigger_level = predictions['triggers'].get('trigger_level', 'LOW')
            if trigger_level == 'HIGH':
                valence = predictions['triggers'].get('valence', 'MIXED')
                if valence == 'NEGATIVE':
                    recommendations.append("🔥 High negative trigger detected - avoid these words/phrases")
                elif valence == 'POSITIVE':
                    recommendations.append("✅ High positive trigger - use these words to your advantage")
        
        # Add universal Trump-specific advice
        if overall_response not in ['COOPERATIVE', 'TRANSACTIONAL']:
            recommendations.append("💡 Always frame proposals as Trump 'winning'")
            recommendations.append("💡 Use flattery strategically but authentically")
            recommendations.append("💡 Avoid moral arguments - use business logic")
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of the unified model configuration.
        
        Returns:
        --------
        dict : Model summary information
        """
        
        return {
            'num_sub_models': 6,
            'sub_models': [
                'Entity Reaction Profiler',
                'Personality Compatibility Predictor',
                'Negotiation Success Predictor',
                'Response Classifier (ML)',
                'Psychological Influence Model',
                'Trigger Word Detector'
            ],
            'trump_profile_available': self.trump_profile is not None,
            'data_loaded': {
                'features': self.df_features is not None,
                'profiles': self.df_profiles is not None,
                'triggers': self.df_triggers is not None
            },
            'trump_personality': self.trump_profile
        }

