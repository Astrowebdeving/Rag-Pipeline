import logging
from flask import Blueprint, jsonify, request
from app.utils.auth import require_api_key
from app.services.state_service import rag_state

logger = logging.getLogger(__name__)

model_bp = Blueprint('model', __name__, url_prefix='/api')


@model_bp.route('/model/reload', methods=['POST'])
@require_api_key
def reload_model():
    data = request.get_json(force=True)
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({'error': 'model_name required'}), 400

    # Clear cached generator so next use will re-instantiate with new model
    rag_state.cached_generator = None
    rag_state.update_config({'generation_model': model_name})
    logger.info(f"Reloaded generation model: {model_name}")
    return jsonify({'ok': True, 'active_model': model_name}), 200

