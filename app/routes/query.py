"""Query Processing and Response Generation Routes"""

import logging
import time
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify

from app.services.state_service import rag_state
from app.utils.embedding import create_single_embedding
from app.retriever.dense_retriever import DenseRetriever
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.advanced_retriever import AdvancedRetriever
from app.augmented_generation.huggingface_generator import HuggingFaceGenerator
from app.augmented_generation.ollama_generator import OllamaGenerator

logger = logging.getLogger(__name__)

# Create blueprint
query_bp = Blueprint('query', __name__, url_prefix='/api')

def create_retriever_with_method(method: str):
    """Create retriever based on specified method."""
    if method == 'dense':
        return DenseRetriever()
    elif method == 'hybrid':
        return HybridRetriever()
    elif method == 'advanced':
        return AdvancedRetriever(
            base_retriever_type='hybrid',
            rerank_results=rag_state.config.get('enable_reranking', True),
            expand_queries=rag_state.config.get('enable_query_expansion', True),
            diversity_factor=rag_state.config.get('diversity_factor', 0.3),
            use_llm_query_enhancement=True  # Enable LLM query enhancement for advanced retrieval
        )
    else:
        logger.warning(f"Unknown retrieval method: {method}, defaulting to dense")
        return DenseRetriever()

def get_generator() -> Optional[Any]:
    """Get or create the appropriate generator based on configuration."""
    generation_method = rag_state.config.get('generation_method', 'huggingface')
    
    # Check if we have a cached generator of the right type
    if (rag_state.cached_generator is not None and 
        rag_state.cached_generator_type == generation_method):
        return rag_state.cached_generator
    
    try:
        if generation_method == 'ollama':
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
            
        elif generation_method == 'huggingface':
            logger.info("Creating HuggingFace generator")
            model_name = rag_state.config.get('generation_model', 'google/flan-t5-large')
            generator = HuggingFaceGenerator(model_name=model_name)
            
            # Cache the generator
            rag_state.cached_generator = generator
            rag_state.cached_generator_type = 'huggingface'
            logger.info(f"HuggingFace generator cached: {model_name}")
            return generator
            
        else:
            logger.warning(f"Unknown generation method: {generation_method}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create generator ({generation_method}): {e}")
        return None

@query_bp.route('/query', methods=['POST'])
def process_query():
    """Process user query and return response."""
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query_text = data['query'].strip()
        if not query_text:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Processing query: {query_text[:100]}...")
        
        # Check if we have any documents
        if rag_state.get_chunk_count() == 0:
            return jsonify({
                'success': False,
                'error': 'No documents available. Please upload documents first.',
                'response': 'I don\'t have any documents to search through. Please upload some documents first.',
                'retrieved_chunks': [],
                'processing_time': time.time() - start_time
            })
        
        # Get configuration
        retrieval_method = rag_state.config.get('retrieval_method', 'dense')
        embedding_method = rag_state.config.get('embedding_method', 'sentence_transformer')
        generation_method = rag_state.config.get('generation_method', 'huggingface')
        top_k = rag_state.config.get('retrieval_top_k', 5)
        
        # Create query embedding
        logger.info(f"Creating query embedding using: {embedding_method}")
        query_embedding = create_single_embedding(query_text, method=embedding_method)
        
        if query_embedding is None:
            return jsonify({
                'error': 'Failed to create query embedding',
                'details': 'Embedding service may be unavailable'
            }), 500
        
        # Create retriever and retrieve relevant chunks
        logger.info(f"Retrieving chunks using method: {retrieval_method}")
        retriever = create_retriever_with_method(retrieval_method)
        
        if retrieval_method == 'advanced':
            # Advanced retriever needs both embedding and text
            retrieved_chunks = retriever.retrieve(
                vector_store=rag_state.vector_store,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k
            )
        elif retrieval_method == 'hybrid':
            # Hybrid retriever needs both embedding and text
            retrieved_chunks = retriever.retrieve(
                vector_store=rag_state.vector_store,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k
            )
        else:
            # Dense retriever only needs embedding
            retrieved_chunks = retriever.retrieve(
                vector_store=rag_state.vector_store,
                query_embedding=query_embedding,
                top_k=top_k
            )
        
        if not retrieved_chunks:
            return jsonify({
                'success': False,
                'error': 'No relevant content found',
                'response': 'I couldn\'t find any relevant information for your query.',
                'retrieved_chunks': [],
                'processing_time': time.time() - start_time
            })
        
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Update analytics
        rag_state.update_analytics('query')
        rag_state.update_analytics('retrieval', method=retrieval_method)
        
        # Generate response if generation is enabled
        response_text = ""
        generation_info = {}
        
        if generation_method != 'none':
            logger.info(f"Generating response using: {generation_method}")
            generator = get_generator()
            
            if generator is None:
                logger.error("No AI generator available")
                generation_info = {
                    'method': generation_method,
                    'success': False,
                    'error': 'No AI generator available'
                }
                rag_state.update_analytics('error')
                
                # Return retrieval results without generation
                return jsonify({
                    'success': True,
                    'response': 'I found relevant information but couldn\'t generate a response. Here are the relevant chunks:',
                    'retrieved_chunks': retrieved_chunks,
                    'generation_info': generation_info,
                    'retrieval_method': retrieval_method,
                    'processing_time': time.time() - start_time
                })
            
            try:
                # Generate response
                generation_result = generator.generate_response(
                    query=query_text,
                    retrieved_chunks=retrieved_chunks
                )
                
                if generation_result.get('success', False):
                    response_text = generation_result.get('response', '')
                    generation_info = {
                        'method': generation_method,
                        'success': True,
                        'model': generation_result.get('model', 'unknown')
                    }
                    rag_state.analytics['successful_generations'] += 1
                    logger.info("Response generated successfully")
                else:
                    error_msg = generation_result.get('error', 'Unknown generation error')
                    logger.error(f"AI Generation Issue: {error_msg}")
                    generation_info = {
                        'method': generation_method,
                        'success': False,
                        'error': error_msg
                    }
                    rag_state.analytics['failed_generations'] += 1
                    rag_state.update_analytics('error')
                    
                    # Fallback to chunk-based response
                    response_text = "I found relevant information but couldn't generate a synthesized response. Here's what I found:"
                    
            except Exception as e:
                logger.error(f"Error during response generation: {e}")
                generation_info = {
                    'method': generation_method,
                    'success': False,
                    'error': str(e)
                }
                rag_state.analytics['failed_generations'] += 1
                rag_state.update_analytics('error')
                response_text = "I found relevant information but encountered an error during response generation."
        else:
            # No generation - just return retrieved chunks
            response_text = "Here are the most relevant chunks from your documents:"
            generation_info = {'method': 'none', 'success': True}
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return jsonify({
            'success': True,
            'response': response_text,
            'retrieved_chunks': retrieved_chunks,
            'generation_info': generation_info,
            'retrieval_method': retrieval_method,
            'embedding_method': embedding_method,
            'chunks_retrieved': len(retrieved_chunks),
            'processing_time': processing_time
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing query: {e}")
        rag_state.update_analytics('error')
        
        return jsonify({
            'success': False,
            'error': 'Failed to process query',
            'details': str(e),
            'processing_time': processing_time
        }), 500

@query_bp.route('/query/test', methods=['GET'])
def test_query():
    """Test endpoint to check if query processing is working."""
    try:
        stats = {
            'documents_available': rag_state.get_document_count(),
            'chunks_available': rag_state.get_chunk_count(),
            'current_config': rag_state.get_config(),
            'generator_available': get_generator() is not None,
            'vector_store_ready': hasattr(rag_state.vector_store, 'search')
        }
        
        return jsonify({
            'success': True,
            'message': 'Query system is ready',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in query test: {e}")
        return jsonify({
            'success': False,
            'error': 'Query system test failed',
            'details': str(e)
        }), 500