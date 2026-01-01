"""
Trump Speech Analytics API
==========================
Flask REST API for predictive analytics models.

Run with: python -m api.app
Or: python api/app.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime

# Import models
from .models import (
    DataLoader,
    EntityReactionProfiler,
    PersonalityCompatibilityPredictor,
    NegotiationSuccessPredictor,
    ResponseClassifier,
    InfluenceStrategyModel,
    TriggerWordDetector
)
from .models.data_loader import get_data_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize models (lazy loading)
_entity_profiler = None
_personality_predictor = None
_negotiation_predictor = None
_response_classifier = None
_influence_model = None
_trigger_detector = None


def get_entity_profiler():
    global _entity_profiler
    if _entity_profiler is None:
        _entity_profiler = EntityReactionProfiler()
    return _entity_profiler


def get_personality_predictor():
    global _personality_predictor
    if _personality_predictor is None:
        _personality_predictor = PersonalityCompatibilityPredictor()
    return _personality_predictor


def get_negotiation_predictor():
    global _negotiation_predictor
    if _negotiation_predictor is None:
        _negotiation_predictor = NegotiationSuccessPredictor()
    return _negotiation_predictor


def get_response_classifier():
    global _response_classifier
    if _response_classifier is None:
        _response_classifier = ResponseClassifier()
    return _response_classifier


def get_influence_model():
    global _influence_model
    if _influence_model is None:
        _influence_model = InfluenceStrategyModel()
    return _influence_model


def get_trigger_detector():
    global _trigger_detector
    if _trigger_detector is None:
        _trigger_detector = TriggerWordDetector()
    return _trigger_detector


# =============================================================================
# HEALTH & INFO ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """API root - returns API info."""
    return jsonify({
        'name': 'Trump Speech Analytics API',
        'version': '1.0.0',
        'description': 'Predictive analytics for Trump speech patterns',
        'endpoints': {
            'health': '/api/health',
            'stats': '/api/stats',
            'entity': '/api/predict/entity',
            'personality': '/api/predict/personality',
            'negotiation': '/api/predict/negotiation',
            'response': '/api/predict/response',
            'influence': '/api/predict/influence',
            'trigger': '/api/predict/trigger'
        },
        'documentation': '/api/docs'
    })


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/stats')
def stats():
    """Get data statistics."""
    loader = get_data_loader()
    return jsonify({
        'status': 'SUCCESS',
        'data': loader.get_stats()
    })


@app.route('/api/docs')
def docs():
    """API documentation."""
    return jsonify({
        'endpoints': [
            {
                'path': '/api/predict/entity',
                'method': 'POST',
                'description': 'Predict Trump\'s reaction to an entity',
                'body': {'entity': 'string (required)'},
                'example': {'entity': 'China'}
            },
            {
                'path': '/api/predict/personality',
                'method': 'POST',
                'description': 'Predict compatibility based on Big Five personality',
                'body': {
                    'openness': 'number (0-100)',
                    'conscientiousness': 'number (0-100)',
                    'extraversion': 'number (0-100)',
                    'agreeableness': 'number (0-100)',
                    'neuroticism': 'number (0-100)'
                }
            },
            {
                'path': '/api/predict/negotiation',
                'method': 'POST',
                'description': 'Predict negotiation success probability',
                'body': {
                    'topic': 'string (required)',
                    'style': 'string (required)',
                    'strategies': 'array of strings (optional)'
                }
            },
            {
                'path': '/api/predict/response',
                'method': 'POST',
                'description': 'Classify expected response type',
                'body': {
                    'sentiment': 'number (0-1)',
                    'neg_ratio': 'number (0-1)',
                    'pos_ratio': 'number (0-1)',
                    'power_ratio': 'number (0-1)',
                    'certainty': 'number (0-1)'
                }
            },
            {
                'path': '/api/predict/influence',
                'method': 'POST',
                'description': 'Get ranked influence strategies',
                'body': {
                    'context': 'string (required)',
                    'goal': 'string (optional)'
                }
            },
            {
                'path': '/api/predict/trigger',
                'method': 'POST',
                'description': 'Detect trigger words in text',
                'body': {'text': 'string (required)'}
            }
        ]
    })


# =============================================================================
# MODEL 1: ENTITY REACTION PROFILER
# =============================================================================

@app.route('/api/predict/entity', methods=['POST'])
def predict_entity():
    """Predict Trump's reaction to an entity."""
    try:
        data = request.get_json()

        if not data or 'entity' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required field: entity'
            }), 400

        profiler = get_entity_profiler()
        result = profiler.predict(data['entity'])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in entity prediction: {e}")
        return jsonify({
            'status': 'ERROR',
            'message': str(e)
        }), 500


@app.route('/api/entities/top', methods=['GET'])
def get_top_entities():
    """Get top entities by speech count."""
    try:
        entity_type = request.args.get('type')
        limit = request.args.get('limit', 20, type=int)

        profiler = get_entity_profiler()
        result = profiler.get_top_entities(entity_type, limit)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting top entities: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/entities/search', methods=['GET'])
def search_entities():
    """Search for entities."""
    try:
        query = request.args.get('q', '')
        limit = request.args.get('limit', 10, type=int)

        if not query:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing query parameter: q'
            }), 400

        profiler = get_entity_profiler()
        result = profiler.search_entities(query, limit)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# =============================================================================
# MODEL 2: PERSONALITY COMPATIBILITY
# =============================================================================

@app.route('/api/predict/personality', methods=['POST'])
def predict_personality():
    """Predict personality compatibility with Trump."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing request body'
            }), 400

        # Extract personality traits
        required = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        personality = {}

        for trait in required:
            if trait not in data:
                return jsonify({
                    'status': 'ERROR',
                    'message': f'Missing required field: {trait}'
                }), 400
            personality[trait] = float(data[trait])

        predictor = get_personality_predictor()
        result = predictor.predict(personality)

        return jsonify(result)

    except ValueError as e:
        return jsonify({
            'status': 'ERROR',
            'message': f'Invalid value: {e}'
        }), 400
    except Exception as e:
        logger.error(f"Error in personality prediction: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/personality/trump', methods=['GET'])
def get_trump_personality():
    """Get Trump's computed personality profile."""
    try:
        predictor = get_personality_predictor()
        result = predictor.get_trump_profile()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting Trump profile: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# =============================================================================
# MODEL 3: NEGOTIATION SUCCESS
# =============================================================================

@app.route('/api/predict/negotiation', methods=['POST'])
def predict_negotiation():
    """Predict negotiation success probability."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing request body'
            }), 400

        if 'topic' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required field: topic'
            }), 400

        if 'style' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required field: style'
            }), 400

        topic = data['topic']
        style = data['style']
        strategies = data.get('strategies', [])

        predictor = get_negotiation_predictor()
        result = predictor.predict(topic, style, strategies)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in negotiation prediction: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/negotiation/compare', methods=['POST'])
def compare_negotiation_styles():
    """Compare different negotiation styles for a topic."""
    try:
        data = request.get_json()

        if not data or 'topic' not in data or 'styles' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required fields: topic, styles'
            }), 400

        predictor = get_negotiation_predictor()
        result = predictor.compare_approaches(data['topic'], data['styles'])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error comparing styles: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/negotiation/options', methods=['GET'])
def get_negotiation_options():
    """Get available topics, styles, and strategies."""
    try:
        predictor = get_negotiation_predictor()
        result = predictor.get_available_options()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting options: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# =============================================================================
# MODEL 4: RESPONSE CLASSIFIER
# =============================================================================

@app.route('/api/predict/response', methods=['POST'])
def predict_response():
    """Classify expected response type."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing request body'
            }), 400

        required = ['sentiment', 'neg_ratio', 'pos_ratio', 'power_ratio', 'certainty']
        for field in required:
            if field not in data:
                return jsonify({
                    'status': 'ERROR',
                    'message': f'Missing required field: {field}'
                }), 400

        classifier = get_response_classifier()
        result = classifier.classify(
            sentiment=float(data['sentiment']),
            neg_ratio=float(data['neg_ratio']),
            pos_ratio=float(data['pos_ratio']),
            power_ratio=float(data['power_ratio']),
            certainty=float(data['certainty'])
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({
            'status': 'ERROR',
            'message': f'Invalid value: {e}'
        }), 400
    except Exception as e:
        logger.error(f"Error in response classification: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/response/features', methods=['GET'])
def get_response_features():
    """Get feature importance for response types."""
    try:
        classifier = get_response_classifier()
        result = classifier.get_feature_importance()
        return jsonify({'status': 'SUCCESS', 'features': result})
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# =============================================================================
# MODEL 5: INFLUENCE STRATEGY
# =============================================================================

@app.route('/api/predict/influence', methods=['POST'])
def predict_influence():
    """Get ranked influence strategies."""
    try:
        data = request.get_json()

        if not data or 'context' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required field: context'
            }), 400

        context = data['context']
        goal = data.get('goal')

        model = get_influence_model()
        result = model.predict(context, goal)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in influence prediction: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/influence/options', methods=['GET'])
def get_influence_options():
    """Get available contexts, goals, and principles."""
    try:
        model = get_influence_model()
        result = model.get_available_options()
        return jsonify({'status': 'SUCCESS', **result})
    except Exception as e:
        logger.error(f"Error getting options: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# =============================================================================
# MODEL 6: TRIGGER WORD DETECTOR
# =============================================================================

@app.route('/api/predict/trigger', methods=['POST'])
def predict_trigger():
    """Detect trigger words in text."""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required field: text'
            }), 400

        detector = get_trigger_detector()
        result = detector.detect(data['text'])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in trigger detection: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/trigger/analyze', methods=['POST'])
def analyze_emotions():
    """Analyze emotional content of text."""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing required field: text'
            }), 400

        detector = get_trigger_detector()
        result = detector.analyze_emotional_content(data['text'])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/trigger/check', methods=['GET'])
def check_trigger_word():
    """Check if a single word is a trigger."""
    try:
        word = request.args.get('word', '')

        if not word:
            return jsonify({
                'status': 'ERROR',
                'message': 'Missing query parameter: word'
            }), 400

        detector = get_trigger_detector()
        result = detector.check_word(word)

        return jsonify({'status': 'SUCCESS', **result})

    except Exception as e:
        logger.error(f"Error checking word: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/trigger/lists', methods=['GET'])
def get_trigger_lists():
    """Get lists of known trigger words."""
    try:
        detector = get_trigger_detector()
        result = detector.get_trigger_lists()
        return jsonify({'status': 'SUCCESS', **result})
    except Exception as e:
        logger.error(f"Error getting lists: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'ERROR',
        'message': 'Endpoint not found',
        'available_endpoints': '/'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'status': 'ERROR',
        'message': 'Internal server error'
    }), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    logger.info("Starting Trump Speech Analytics API...")

    # Pre-load data
    loader = get_data_loader()
    _ = loader.entity_profiles
    _ = loader.features_df

    logger.info(f"Data loaded: {loader.get_stats()}")

    # Run server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
