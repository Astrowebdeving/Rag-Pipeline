"""RAG System Web API - Clean and Modular

This is the main Flask application that orchestrates all RAG components.
All business logic has been moved to dedicated services and routes.
"""

import os
import sys
import atexit
import signal
import logging
import tempfile
from flask import Flask, render_template, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

def cleanup_resources():
    """Cleanup all system resources to prevent leaks."""
    try:
        logger.info("Cleaning up application resources...")
        
        # Import cleanup functions
        try:
            from app.embedders.sentence_transformer_embedder import cleanup_model
            cleanup_model()  # Cleanup sentence transformer model
        except ImportError:
            logger.warning("sentence_transformer_embedder not available for cleanup")
        
        # Import services to cleanup
        try:
            from app.services.state_service import rag_state
            rag_state.cleanup_resources()  # Cleanup RAG system state
        except ImportError:
            logger.warning("state_service not available for cleanup")
        
        logger.info("Resource cleanup completed")
    except Exception as e:
        logger.warning(f"Error during resource cleanup: {e}")

# Initialize Flask application
app = Flask(__name__, 
            static_folder='../web_ui/assets',
            template_folder='../web_ui/pages')

# Enable CORS for all routes
CORS(app)

# Configure file upload settings
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Import route blueprints
from app.routes.system import system_bp
from app.routes.documents import documents_bp
from app.routes.query import query_bp
from app.routes.feedback import feedback_bp
from app.routes.model import model_bp

# Import services to initialize them
from app.services.state_service import rag_state
from app.services.file_service import file_service

# Register route blueprints
app.register_blueprint(system_bp)
app.register_blueprint(documents_bp)
app.register_blueprint(query_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(model_bp)

# Error handler for file size limit exceeded
@app.errorhandler(413)
def too_large(e):
    """Handle file size too large errors."""
    return {'error': 'File size exceeds the 50MB limit. Please choose a smaller file.'}, 413

# Main web interface routes
@app.route('/')
def index():
    """Serve the main web interface."""
    api_key = os.environ.get('RAG_API_KEY', '')
    return render_template('index.html', api_key=api_key)

@app.route('/settings')
def settings():
    """Serve the settings configuration page."""
    api_key = os.environ.get('RAG_API_KEY', '')
    return render_template('settings.html', api_key=api_key)

# Development server startup
if __name__ == '__main__':
    """Run the Flask development server."""
    print("RAG System ready at: http://localhost:5000")
    
    # Log system initialization
    logger.info("RAG system initialized successfully")
    logger.info(f"Supported file types: {file_service.get_supported_extensions()}")
    logger.info(f"Current configuration: {rag_state.config}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)