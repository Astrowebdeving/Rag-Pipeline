"""System Routes

Routes for system health, configuration, and analytics.
"""

import logging
import traceback
from datetime import datetime
from flask import Blueprint, jsonify, request
from app.services.state_service import rag_state
from app.utils.embedding import test_embedding_method
from app.utils.auth import require_api_key

logger = logging.getLogger(__name__)

# Create blueprint for system routes
system_bp = Blueprint('system', __name__, url_prefix='/api')


@system_bp.route('/health', methods=['GET'])
def health_check():
    """Check system health and return comprehensive status information."""
    try:
        # Get system statistics
        stats = rag_state.get_system_stats()
        
        # Test embedding system
        embedding_test = test_embedding_method(rag_state.config['embedding_method'])
        
        # Prepare health status
        health_status = {
            'status': 'healthy',
            'timestamp': str(datetime.now()),
            'documents_count': stats['documents_count'],
            'chunks_count': stats['chunks_count'],
            'config': stats['config'],
            'vector_store_ready': rag_state.vector_store is not None,
            'health_score': stats['health_score'],
            'diagnostics': {
                'embedding_test': embedding_test,
                'vector_store_status': 'ready' if rag_state.vector_store else 'not_ready',
                'cached_generator': rag_state.cached_generator is not None
            }
        }
        
        # Determine overall status
        if embedding_test['status'] != 'success':
            health_status['status'] = 'degraded'
        
        if stats['health_score'] < 0.5:
            health_status['status'] = 'unhealthy'
        
        return jsonify(health_status), 200
        
    except Exception as e:
        # Handle any errors in health check
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': str(datetime.now())
        }), 500


@system_bp.route('/config', methods=['GET'])
def get_config():
    """Get current system configuration."""
    try:
        return jsonify(rag_state.config), 200
    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        return jsonify({'error': str(e)}), 500


@system_bp.route('/config', methods=['POST'])
@require_api_key
def update_config():
    """Update system configuration with user preferences."""
    try:
        # Get new configuration from request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        new_config = request.get_json()
        if not new_config:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        # Update configuration using state service
        success = rag_state.update_config(new_config)
        
        if success:
            return jsonify({
                'message': 'Configuration updated successfully',
                'config': rag_state.config
            }), 200
        else:
            return jsonify({'error': 'Invalid configuration values'}), 400
        
    except Exception as e:
        logger.error(f"Config update failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@system_bp.route('/analytics', methods=['GET'])
def get_analytics():
    """Get comprehensive system analytics and performance metrics."""
    try:
        # Get system statistics
        stats = rag_state.get_system_stats()
        
        # Update real-time metrics
        memory_usage = rag_state.vector_store.get_memory_usage_estimate()
        rag_state.performance_metrics['memory_usage'] = memory_usage
        
        # Prepare comprehensive analytics response
        analytics_data = {
            'system_overview': {
                'total_documents_processed': stats['analytics']['total_documents_processed'],
                'total_queries_processed': stats['analytics']['total_queries_processed'],
                'total_chunks_created': stats['analytics']['total_chunks_created'],
                'error_count': stats['analytics']['error_count'],
                'last_activity': stats['analytics']['last_activity_timestamp'],
                'system_health_score': stats['health_score']
            },
            'usage_statistics': {
                'retrieval_methods': stats['analytics']['retrieval_stats'],
                'chunking_methods': stats['analytics']['chunking_stats'],
                'embedding_methods': stats['analytics']['embedding_stats']
            },
            'performance_metrics': {
                'memory_usage': stats['performance']['memory_usage'],
                'average_response_time': stats['analytics']['average_response_time'],
                'cache_hit_rate': stats['performance']['cache_hit_rate']
            },
            'configuration': {
                'current_config': stats['config'],
                'analytics_enabled': stats['config'].get('enable_analytics', True),
                'caching_enabled': stats['config'].get('cache_embeddings', True)
            },
            'recommendations': []
        }
        
        # Generate performance recommendations
        recommendations = []
        
        if stats['analytics']['error_count'] > 0:
            error_rate = stats['analytics']['error_count'] / max(
                stats['analytics']['total_documents_processed'] + 
                stats['analytics']['total_queries_processed'], 1
            )
            if error_rate > 0.1:
                recommendations.append({
                    'type': 'warning',
                    'message': f'High error rate detected ({error_rate:.1%}). Check system logs.',
                    'action': 'review_logs'
                })
        
        if stats['chunks_count'] > 10000:
            recommendations.append({
                'type': 'info',
                'message': 'Large number of chunks may impact performance. Consider document cleanup.',
                'action': 'optimize_storage'
            })
        
        if stats['performance']['cache_hit_rate'] < 0.5 and stats['config'].get('cache_embeddings', True):
            recommendations.append({
                'type': 'optimization',
                'message': 'Low cache hit rate. Consider increasing cache size.',
                'action': 'tune_cache'
            })
        
        analytics_data['recommendations'] = recommendations
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@system_bp.route('/analytics/reset', methods=['POST'])
@require_api_key
def reset_analytics():
    """Reset analytics counters (useful for testing or clean start)."""
    try:
        rag_state.reset_analytics()
        
        return jsonify({
            'message': 'Analytics counters reset successfully',
            'reset_timestamp': str(datetime.now())
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics reset failed: {str(e)}")
        return jsonify({'error': str(e)}), 500