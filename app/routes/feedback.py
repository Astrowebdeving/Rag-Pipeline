import logging
from flask import Blueprint, request, jsonify
from app.utils.auth import require_api_key
from app.personalization.models import create_session, Feedback

logger = logging.getLogger(__name__)

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api')


@feedback_bp.route('/feedback', methods=['POST'])
@require_api_key
def submit_feedback():
    data = request.get_json(force=True)
    required = ['user_id', 'query', 'answer', 'rating', 'modules_used']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'missing required fields: {", ".join(missing)}'}), 400

    try:
        Session = create_session()
        s = Session()
        fb = Feedback(
            user_id=data['user_id'],
            session_id=data.get('session_id'),
            query=data['query'],
            answer=data['answer'],
            rating=int(data['rating']),
            comments=data.get('comments', ''),
            aspects=data.get('aspects', {}),
            context_embedding=data.get('context_embedding'),
            modules_used=data['modules_used'],
            latency_ms=data.get('latency_ms'),
        )
        s.add(fb)
        s.commit()
        s.close()
        logger.info('Feedback stored successfully')
        return jsonify({'ok': True}), 200
    except Exception as e:
        logger.error(f'Failed to store feedback: {e}')
        return jsonify({'error': str(e)}), 500

