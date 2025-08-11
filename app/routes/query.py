"""Query Routes

Routes for querying documents and retrieving relevant information.
"""

import logging
import traceback
from datetime import datetime
from flask import Blueprint, jsonify, request
from app.services.state_service import rag_state
from app.utils.embedding import create_query_embedding_with_method
from app.retriever.dense_retriever import DenseRetriever
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.advanced_retriever import AdvancedRetriever
from app.augmented_generation.huggingface_generator import create_huggingface_generator
from app.augmented_generation.ollama_generator import OllamaGenerator
from app.utils.auth import require_api_key

logger = logging.getLogger(__name__)

# Create blueprint for query routes
query_bp = Blueprint('query', __name__, url_prefix='/api')


def create_retriever_with_method(method: str):
    """Create retriever instance based on configured method.
    
    Args:
        method: Retrieval method ('dense', 'hybrid', 'advanced')
        
    Returns:
        Retriever instance
    """
    # Update analytics if enabled
    rag_state.update_analytics('retrieval', method=method)
    
    # Create retriever based on method
    if method == 'hybrid':
        return HybridRetriever()
    elif method == 'advanced':
        # Create advanced retriever with configuration and LLM enhancements
        return AdvancedRetriever(
            base_retriever_type='hybrid',
            rerank_results=rag_state.config.get('enable_reranking', True),
            expand_queries=rag_state.config.get('enable_query_expansion', True),
            diversity_factor=rag_state.config.get('diversity_factor', 0.3),
            use_llm_query_enhancement=True  # Enable LLM query enhancement for advanced retrieval
        )
    else:  # default to dense
        return DenseRetriever()


def get_generator():
    """Get or create LLM generator based on current configuration."""
    # Check if we need to create/update the generator
    current_method = rag_state.config['generation_method']
    current_model = rag_state.config.get('generation_model', 'google/flan-t5-base')
    
    # Return None if generation is disabled
    if current_method == 'none':
        return None
    
    # Check if we can reuse cached generator
    if (rag_state.cached_generator is not None and 
        rag_state.cached_generator_type == current_method):
        return rag_state.cached_generator
    
    # Create new generator based on method
    try:
        if current_method == 'ollama':
            logger.info("Creating Ollama generator with model: deepseek-r1:8b")
            generator = OllamaGenerator(
                model_name="deepseek-r1:8b",
                max_tokens=500,
                temperature=0.7
            )
            
            # Cache the generator
            rag_state.cached_generator = generator
            rag_state.cached_generator_type = 'ollama'
            logger.info("Ollama generator cached successfully")
            return generator
            
        elif current_method == 'huggingface':
            generator = create_huggingface_generator(
                "google/flan-t5-base", use_small_model=True
            )
            
            # Cache the generator for future use
            rag_state.cached_generator = generator
            rag_state.cached_generator_type = 'huggingface'
            
            return generator
        else:
            # No generation method or unsupported method
            return None
        
    except Exception as e:
        # Handle generator creation errors
        logger.error(f"Failed to create generator: {str(e)}")
        return None


def validate_chunk_relevance(chunks_with_scores, query, min_score=0.1):
    """Validate and filter chunks for quality and relevance.
    
    Args:
        chunks_with_scores: List of (chunk, score) tuples
        query: Original query text
        min_score: Minimum relevance score threshold
        
    Returns:
        Filtered list of (chunk, score) tuples
    """
    if not chunks_with_scores:
        return []
    
    # Filter by minimum score
    filtered_chunks = [(chunk, score) for chunk, score in chunks_with_scores 
                      if score >= min_score]
    
    # Remove duplicate chunks
    seen_chunks = set()
    unique_chunks = []
    
    for chunk, score in filtered_chunks:
        chunk_text = chunk.strip().lower()
        if chunk_text not in seen_chunks and len(chunk_text) > 20:
            seen_chunks.add(chunk_text)
            unique_chunks.append((chunk, score))
    
    logger.debug(f"Filtered {len(chunks_with_scores)} chunks to {len(unique_chunks)} unique, relevant chunks")
    
    return unique_chunks


@query_bp.route('/query', methods=['POST'])
@require_api_key
def query_documents():
    """Process user query and return AI-generated response with sources."""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        # Get query data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        query = data.get('query', '').strip()
        top_k = data.get('top_k', rag_state.config.get('retrieval_top_k', 5))
        use_generation = data.get('use_generation', True)
        
        # Validate input
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        if not rag_state.all_chunks:
            return jsonify({'error': 'No documents have been ingested yet'}), 400
        
        logger.info(f"Processing query: '{query[:50]}...' (top_k={top_k})")
        
        # Create query embedding using the same method as stored documents
        embedding_method = rag_state.actual_embedding_method or rag_state.config['embedding_method']
        logger.info(f"Using embedding method for query: {embedding_method}")
        
        query_embedding = create_query_embedding_with_method(
            query, 
            embedding_method
        )
        
        if query_embedding is None:
            return jsonify({'error': 'Failed to create query embedding'}), 500
        
        # Retrieve relevant chunks using configured method
        retriever = create_retriever_with_method(rag_state.config['retrieval_method'])
        
        # Handle different retriever types properly
        chunks_with_scores = []
        
        try:
            if rag_state.config['retrieval_method'] == 'advanced':
                chunks_with_scores = retriever.retrieve_with_scores(
                    rag_state.vector_store, query_embedding, query, top_k
                )
            elif rag_state.config['retrieval_method'] == 'hybrid':
                chunks_with_scores = retriever.retrieve_with_scores(
                    rag_state.vector_store, query_embedding, query, top_k
                )
            else:
                chunks_with_scores = retriever.retrieve_with_scores(
                    rag_state.vector_store, query_embedding, top_k
                )
            
            # Ensure we have a valid list (not None)
            if chunks_with_scores is None:
                chunks_with_scores = []
                logger.warning("Retriever returned None, using empty list")
                
        except Exception as retrieval_error:
            logger.error(f"Retrieval failed: {str(retrieval_error)}")
            chunks_with_scores = []
        
        # Update analytics
        rag_state.update_analytics('query')
        
        # Validate and filter chunks for quality
        validated_chunks = validate_chunk_relevance(chunks_with_scores, query)
        
        # Prepare response data
        response_data = {
            'chunks': [chunk for chunk, _ in validated_chunks],
            'scores': [float(score) for _, score in validated_chunks],
            'num_results': len(validated_chunks),
            'retrieval_method': rag_state.config['retrieval_method'],
            'embedding_method': rag_state.config['embedding_method'],
            'generation_enabled': use_generation,
            'query_processed': query,
            'validation_applied': True,
            'original_results_count': len(chunks_with_scores),
            'config_used': {
                'top_k': top_k,
                'retrieval_method': rag_state.config['retrieval_method'],
                'embedding_method': rag_state.config['embedding_method']
            }
        }
        
        # Generate AI response if requested and possible
        if use_generation and rag_state.config['generation_method'] != 'none':
            generator = get_generator()
            
            if generator:
                # Extract just the chunk texts for generation (use validated chunks)
                chunk_texts = [chunk for chunk, _ in validated_chunks]
                
                if chunk_texts:
                    # Generate response using LLM
                    generation_result = generator.generate_response(query, chunk_texts)
                    
                    if generation_result.get('success'):
                        response_data['generated_response'] = generation_result['response']
                        response_data['model_used'] = generation_result.get('model_used', generation_result.get('model', 'unknown'))
                        response_data['generation_method'] = rag_state.config['generation_method']
                        rag_state.analytics['successful_generations'] += 1
                    else:
                        response_data['generation_error'] = generation_result.get('error', 'Unknown error')
                        rag_state.analytics['failed_generations'] += 1
                else:
                    response_data['generation_error'] = 'No relevant chunks found for generation'
            else:
                response_data['generation_error'] = 'No generator available'
        
        logger.info(f"Query processed: '{query[:50]}...' -> {len(validated_chunks)} chunks")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        # Handle query processing errors
        logger.error(f"Query processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        rag_state.update_analytics('error')
        return jsonify({'error': str(e)}), 500


@query_bp.route('/search', methods=['POST'])
@require_api_key
def search_chunks():
    """Simple search through document chunks without AI generation."""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        # Get search data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        
        # Validate input
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        if not rag_state.all_chunks:
            return jsonify({'error': 'No documents have been ingested yet'}), 400
        
        logger.info(f"Searching chunks for: '{query[:50]}...'")
        
        # Create query embedding
        query_embedding = create_query_embedding_with_method(
            query, 
            rag_state.config['embedding_method']
        )
        
        if query_embedding is None:
            return jsonify({'error': 'Failed to create query embedding'}), 500
        
        # Use simple dense retriever for search
        retriever = DenseRetriever()
        chunks_with_scores = retriever.retrieve_with_scores(
            rag_state.vector_store, query_embedding, top_k
        )
        
        if chunks_with_scores is None:
            chunks_with_scores = []
        
        # Prepare search results
        search_results = []
        for chunk, score in chunks_with_scores:
            search_results.append({
                'chunk': chunk,
                'score': float(score),
                'length': len(chunk)
            })
        
        response_data = {
            'results': search_results,
            'num_results': len(search_results),
            'query': query,
            'embedding_method': rag_state.config['embedding_method'],
            'total_chunks_searched': len(rag_state.all_chunks)
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@query_bp.route('/similar', methods=['POST'])
@require_api_key
def find_similar():
    """Find chunks similar to provided text."""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        # Get similarity data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        text = data.get('text', '').strip()
        top_k = data.get('top_k', 5)
        
        # Validate input
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if not rag_state.all_chunks:
            return jsonify({'error': 'No documents have been ingested yet'}), 400
        
        logger.info(f"Finding similar chunks for: '{text[:50]}...'")
        
        # Create embedding for input text
        text_embedding = create_query_embedding_with_method(
            text, 
            rag_state.config['embedding_method']
        )
        
        if text_embedding is None:
            return jsonify({'error': 'Failed to create text embedding'}), 500
        
        # Use dense retriever to find similar chunks
        retriever = DenseRetriever()
        similar_chunks = retriever.retrieve_with_scores(
            rag_state.vector_store, text_embedding, top_k
        )
        
        if similar_chunks is None:
            similar_chunks = []
        
        # Prepare similarity results
        similarity_results = []
        for chunk, score in similar_chunks:
            similarity_results.append({
                'chunk': chunk,
                'similarity_score': float(score),
                'length': len(chunk)
            })
        
        response_data = {
            'similar_chunks': similarity_results,
            'num_results': len(similarity_results),
            'input_text': text,
            'embedding_method': rag_state.config['embedding_method']
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}")
        return jsonify({'error': str(e)}), 500