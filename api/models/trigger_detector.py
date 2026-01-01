"""
Trigger Word Detector (Model 6)
===============================
Identifies words and phrases that trigger strong emotional responses.
"""

from typing import Dict, Any, List, Optional
import re
from collections import defaultdict
from .data_loader import get_data_loader


class TriggerWordDetector:
    """Detect emotional trigger words and phrases."""

    # Known trigger words (from analysis)
    POSITIVE_TRIGGERS = [
        'winning', 'beautiful', 'tremendous', 'incredible', 'great', 'amazing',
        'success', 'best', 'wonderful', 'fantastic', 'tremendous', 'huge',
        'winner', 'strong', 'powerful', 'smart', 'genius', 'perfect',
        'billion', 'rich', 'deal', 'jobs', 'america', 'freedom', 'patriot'
    ]

    NEGATIVE_TRIGGERS = [
        'fake', 'disaster', 'terrible', 'horrible', 'corrupt', 'crooked',
        'loser', 'enemy', 'worst', 'nasty', 'weak', 'stupid', 'dumb',
        'failing', 'fraud', 'hoax', 'witch hunt', 'rigged', 'unfair',
        'disgrace', 'pathetic', 'low energy', 'sad', 'bad'
    ]

    # Emotion lexicon for analysis
    EMOTION_LEXICON = {
        'anger': [
            'angry', 'furious', 'hate', 'terrible', 'horrible', 'disaster',
            'corrupt', 'crooked', 'liar', 'enemy', 'destroy', 'attack',
            'stupid', 'idiot', 'pathetic', 'disgrace', 'worst', 'nasty',
            'fake', 'fraud', 'rigged'
        ],
        'fear': [
            'afraid', 'fear', 'terror', 'threat', 'danger', 'scary',
            'worried', 'nervous', 'crime', 'criminal', 'violence', 'terrorist',
            'invasion', 'crisis'
        ],
        'joy': [
            'happy', 'joy', 'love', 'wonderful', 'great', 'amazing',
            'fantastic', 'beautiful', 'excellent', 'incredible', 'tremendous',
            'success', 'win', 'winner', 'best', 'proud'
        ],
        'sadness': [
            'sad', 'sorry', 'unfortunate', 'tragic', 'terrible', 'horrible',
            'devastating', 'heartbreaking', 'loss', 'failing'
        ],
        'trust': [
            'trust', 'believe', 'faith', 'honest', 'true', 'loyal',
            'reliable', 'america', 'freedom', 'liberty', 'patriot'
        ],
        'disgust': [
            'disgusting', 'disgrace', 'shameful', 'pathetic', 'loser',
            'fake', 'fraud', 'corrupt', 'nasty', 'worst', 'horrible'
        ]
    }

    def __init__(self):
        """Initialize the trigger detector."""
        self.data_loader = get_data_loader()
        self._trigger_lookup: Optional[Dict] = None

    def _load_trigger_data(self) -> Dict:
        """Load trigger word data from analysis results."""
        if self._trigger_lookup is not None:
            return self._trigger_lookup

        df_triggers = self.data_loader.trigger_words

        if df_triggers is not None and 'word' in df_triggers.columns:
            self._trigger_lookup = df_triggers.set_index('word').to_dict('index')
        else:
            # Build from known lists
            self._trigger_lookup = {}
            for word in self.POSITIVE_TRIGGERS:
                self._trigger_lookup[word] = {
                    'trigger_score': 70,
                    'negative_ratio': 0.2,
                    'dominant_emotion': 'joy'
                }
            for word in self.NEGATIVE_TRIGGERS:
                self._trigger_lookup[word] = {
                    'trigger_score': 75,
                    'negative_ratio': 0.8,
                    'dominant_emotion': 'anger'
                }

        return self._trigger_lookup

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect trigger words in text.

        Args:
            text: Text to analyze for trigger words

        Returns:
            Detection results with trigger level and details
        """
        trigger_lookup = self._load_trigger_data()

        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        if not words:
            return {
                'status': 'INVALID',
                'message': 'No valid words found in text'
            }

        # Find matching trigger words
        found_triggers = []
        for word in set(words):
            # Check lookup table
            if word in trigger_lookup:
                data = trigger_lookup[word]
                trigger_type = 'negative' if data.get('negative_ratio', 0.5) > 0.5 else 'positive'
                found_triggers.append({
                    'word': word,
                    'type': trigger_type,
                    'score': data.get('trigger_score', 50),
                    'dominant_emotion': data.get('dominant_emotion', 'unknown')
                })
            # Check known lists
            elif word in self.POSITIVE_TRIGGERS:
                found_triggers.append({
                    'word': word,
                    'type': 'positive',
                    'score': 70,
                    'dominant_emotion': 'joy'
                })
            elif word in self.NEGATIVE_TRIGGERS:
                found_triggers.append({
                    'word': word,
                    'type': 'negative',
                    'score': 75,
                    'dominant_emotion': 'anger'
                })

        if not found_triggers:
            return {
                'status': 'NO_TRIGGERS',
                'text': text,
                'message': 'No known trigger words detected',
                'words_analyzed': len(set(words))
            }

        # Calculate aggregate metrics
        max_score = max(t['score'] for t in found_triggers)
        neg_count = sum(1 for t in found_triggers if t['type'] == 'negative')
        pos_count = sum(1 for t in found_triggers if t['type'] == 'positive')
        total = neg_count + pos_count

        avg_negative_ratio = neg_count / total if total > 0 else 0.5

        # Determine trigger level
        if max_score >= 60:
            trigger_level = 'HIGH'
        elif max_score >= 40:
            trigger_level = 'MEDIUM'
        else:
            trigger_level = 'LOW'

        # Determine overall valence
        if avg_negative_ratio > 0.6:
            valence = 'NEGATIVE'
        elif avg_negative_ratio < 0.4:
            valence = 'POSITIVE'
        else:
            valence = 'MIXED'

        return {
            'status': 'FOUND',
            'text': text,
            'trigger_level': trigger_level,
            'trigger_score': round(max_score, 1),
            'valence': valence,
            'triggers_found': found_triggers,
            'summary': {
                'total_triggers': len(found_triggers),
                'positive_count': pos_count,
                'negative_count': neg_count,
                'negative_ratio': round(avg_negative_ratio, 3)
            },
            'words_analyzed': len(set(words))
        }

    def analyze_emotional_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotional content of text using emotion lexicon.

        Args:
            text: Text to analyze

        Returns:
            Emotion breakdown
        """
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_set = set(words)

        emotion_counts = defaultdict(int)
        emotion_words = defaultdict(list)

        for word in word_set:
            for emotion, lexicon in self.EMOTION_LEXICON.items():
                if word in lexicon:
                    emotion_counts[emotion] += words.count(word)
                    emotion_words[emotion].append(word)

        total_emotion_words = sum(emotion_counts.values())

        if total_emotion_words == 0:
            return {
                'status': 'NEUTRAL',
                'message': 'No strong emotional content detected',
                'text': text
            }

        # Calculate percentages
        emotion_percentages = {
            emotion: round(count / total_emotion_words * 100, 1)
            for emotion, count in emotion_counts.items()
        }

        # Find dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        return {
            'status': 'SUCCESS',
            'text': text,
            'dominant_emotion': dominant_emotion,
            'emotion_breakdown': emotion_percentages,
            'emotion_words': dict(emotion_words),
            'total_emotional_words': total_emotion_words,
            'total_words': len(words)
        }

    def get_trigger_lists(self) -> Dict[str, Any]:
        """Get the lists of known trigger words."""
        return {
            'positive_triggers': self.POSITIVE_TRIGGERS,
            'negative_triggers': self.NEGATIVE_TRIGGERS,
            'emotion_categories': list(self.EMOTION_LEXICON.keys())
        }

    def check_word(self, word: str) -> Dict[str, Any]:
        """
        Check if a single word is a trigger.

        Args:
            word: Word to check

        Returns:
            Trigger information for the word
        """
        word_lower = word.lower().strip()
        trigger_lookup = self._load_trigger_data()

        result = {
            'word': word,
            'is_trigger': False,
            'type': None,
            'score': 0
        }

        if word_lower in trigger_lookup:
            data = trigger_lookup[word_lower]
            result['is_trigger'] = True
            result['type'] = 'negative' if data.get('negative_ratio', 0.5) > 0.5 else 'positive'
            result['score'] = data.get('trigger_score', 50)
            result['dominant_emotion'] = data.get('dominant_emotion', 'unknown')
        elif word_lower in self.POSITIVE_TRIGGERS:
            result['is_trigger'] = True
            result['type'] = 'positive'
            result['score'] = 70
        elif word_lower in self.NEGATIVE_TRIGGERS:
            result['is_trigger'] = True
            result['type'] = 'negative'
            result['score'] = 75

        # Check emotion lexicon
        emotions_found = []
        for emotion, lexicon in self.EMOTION_LEXICON.items():
            if word_lower in lexicon:
                emotions_found.append(emotion)
        result['associated_emotions'] = emotions_found

        return result
