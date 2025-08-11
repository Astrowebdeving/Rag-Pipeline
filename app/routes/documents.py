"""Document Upload and Processing Routes"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify

from app.services.state_service import rag_state
from app.services.file_service import file_service
from app.utils.chunking import create_chunks
from app.utils.embedding import create_embeddings
from app.augmented_generation.ollama_generator import OllamaGenerator

logger = logging.getLogger(__name__)

# Create blueprint
documents_bp = Blueprint('documents', __name__, url_prefix='/api')

# Global metadata generator for LLM-based metadata generation
_metadata_generator = None

def get_metadata_generator():
    """Get or create the metadata generator using Ollama."""
    global _metadata_generator
    if _metadata_generator is None:
        try:
            _metadata_generator = OllamaGenerator(
                model_name="deepseek-r1:8b",
                max_tokens=100,
                temperature=0.3
            )
            logger.info("Ollama metadata generator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama metadata generator: {e}")
            _metadata_generator = None
    return _metadata_generator

def generate_chunk_metadata(chunks, doc_id, filename, file_type, enable_llm_analysis=True):
    """Generate metadata for each chunk, optionally using LLM for semantic analysis.
    
    Args:
        chunks: List of text chunks
        doc_id: Document ID
        filename: Original filename
        file_type: File type
        enable_llm_analysis: Whether to use LLM for semantic analysis
        
    Returns:
        List of metadata dictionaries for each chunk
    """
    metadata_list = []
    generator = None
    
    if enable_llm_analysis:
        generator = get_metadata_generator()
        if generator is None:
            logger.warning("LLM metadata analysis disabled - generator not available")
    
    for i, chunk in enumerate(chunks):
        # Basic metadata for every chunk
        metadata = {
            'chunk_id': f"{doc_id}_chunk_{i}",
            'document_id': doc_id,
            'filename': filename,
            'file_type': file_type,
            'chunk_index': i,
            'chunk_length': len(chunk),
            'source': f"{filename}:chunk_{i}",
            'topics_analyzed': False,
            'semantic_description': ''
        }
        
        # Try to generate semantic description using LLM
        if enable_llm_analysis and generator is not None:
            try:
                # Create a focused prompt for semantic analysis
                analysis_prompt = f"""Analyze this text and describe what it's about in 1-2 sentences:

{chunk[:400]}

Description:"""
                
                response_data = generator.generate_response(
                    query=analysis_prompt,
                    retrieved_chunks=[]
                )
                
                if response_data.get('success', False):
                    semantic_description = response_data.get('response', '').strip()
                    
                    if semantic_description:
                        # Clean up the response
                        prefixes_to_remove = [
                            "This text discusses", "The text covers", "This chunk talks about",
                            "The main topics are", "Summary:", "Description:", "The text is about",
                            "This passage"
                        ]
                        
                        for prefix in prefixes_to_remove:
                            if semantic_description.lower().startswith(prefix.lower()):
                                semantic_description = semantic_description[len(prefix):].strip()
                        
                        # Limit length
                        if len(semantic_description) > 200:
                            semantic_description = semantic_description[:200] + "..."
                        
                        # Only use if meaningful
                        if len(semantic_description) > 10:
                            metadata['semantic_description'] = semantic_description
                            metadata['topics_analyzed'] = True
                            logger.debug(f"Generated semantic description for chunk {i}: {semantic_description[:50]}...")
                        else:
                            logger.debug(f"Generated description too short for chunk {i}, using fallback")
                    else:
                        logger.debug(f"Empty response for chunk {i}, using fallback")
                else:
                    logger.warning(f"LLM failed for chunk {i}: {response_data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate semantic metadata for chunk {i}: {e}")
        
        # Fallback semantic description if LLM analysis failed or is disabled
        if not metadata['semantic_description']:
            # Create a simple description based on first words
            chunk_words = chunk.lower().split()[:50]
            if len(chunk_words) >= 10:
                metadata['semantic_description'] = f"Text content covering: {' '.join(chunk_words[:15])}..."
            else:
                metadata['semantic_description'] = f"Text content from {filename}, section {i+1}"
        
        metadata_list.append(metadata)
        
        # Log progress for large documents
        if (i + 1) % 10 == 0:
            logger.info(f"Generated metadata for {i+1}/{len(chunks)} chunks")
    
    return metadata_list

@documents_bp.route('/documents/upload', methods=['POST'])
def upload_document():
    """Upload and process a document with metadata generation."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not file_service.is_supported_file(file.filename):
            supported = ', '.join(file_service.get_supported_extensions())
            return jsonify({
                'error': f'Unsupported file type. Supported types: {supported}'
            }), 400
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Extract text from file
        text_content = file_service.extract_text(file)
        if not text_content or not text_content.strip():
            return jsonify({'error': 'No text content found in file'}), 400
        
        # Get configuration
        chunking_method = rag_state.config.get('chunking_method', 'semantic')
        embedding_method = rag_state.config.get('embedding_method', 'sentence_transformer')
        
        # Create chunks
        logger.info(f"Creating chunks using method: {chunking_method}")
        chunks = create_chunks(text_content, method=chunking_method)
        
        if not chunks:
            return jsonify({'error': 'Failed to create text chunks'}), 500
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate document ID
        doc_id = rag_state.get_next_doc_id()
        
        # Determine if we should enable metadata generation (for advanced retrieval)
        enable_metadata = rag_state.config.get('retrieval_method') == 'advanced'
        
        # Generate metadata for chunks
        logger.info(f"Generating metadata for chunks (LLM analysis: {enable_metadata})")
        chunk_metadata = generate_chunk_metadata(
            chunks, doc_id, file.filename, 
            file_service.get_file_type(file.filename),
            enable_llm_analysis=enable_metadata
        )
        
        # Create embeddings
        logger.info(f"Creating embeddings using method: {embedding_method}")
        embeddings = create_embeddings(chunks, method=embedding_method)
        
        if not embeddings:
            return jsonify({'error': 'Failed to create embeddings'}), 500
        
        logger.info(f"Created {len(embeddings)} embeddings")
        
        # Add to vector store with metadata
        rag_state.vector_store.add_documents(
            chunks, 
            embeddings,
            metadata=chunk_metadata
        )
        
        # Store in state
        rag_state.add_document(
            doc_id=doc_id,
            text=text_content,
            chunks=chunks,
            filename=file.filename,
            file_type=file_service.get_file_type(file.filename)
        )
        
        # Update analytics
        rag_state.update_analytics('chunking', method=chunking_method)
        rag_state.update_analytics('embedding', method=embedding_method)
        
        logger.info(f"Document {doc_id} processed successfully")
        
        return jsonify({
            'success': True,
            'message': 'Document uploaded and processed successfully',
            'document_id': doc_id,
            'chunks_created': len(chunks),
            'embeddings_created': len(embeddings),
            'metadata_generated': len([m for m in chunk_metadata if m['topics_analyzed']]),
            'chunking_method': chunking_method,
            'embedding_method': embedding_method,
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error processing document upload: {e}")
        rag_state.update_analytics('error')
        return jsonify({
            'error': 'Failed to process document',
            'details': str(e)
        }), 500

@documents_bp.route('/documents/list', methods=['GET'])
def list_documents():
    """Get list of uploaded documents."""
    try:
        documents = rag_state.get_all_documents()
        
        # Format document info for response
        doc_list = []
        for doc_id, doc_info in documents.items():
            doc_list.append({
                'id': doc_id,
                'filename': doc_info.get('filename', 'Unknown'),
                'file_type': doc_info.get('file_type', 'Unknown'),
                'chunk_count': doc_info.get('chunk_count', 0),
                'processed_with': doc_info.get('processed_with', {})
            })
        
        return jsonify({
            'success': True,
            'documents': doc_list,
            'total_documents': len(doc_list)
        })
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({
            'error': 'Failed to list documents',
            'details': str(e)
        }), 500

@documents_bp.route('/documents/clear', methods=['POST'])
def clear_documents():
    """Clear all documents and data."""
    try:
        rag_state.clear_all_data()
        
        logger.info("All documents cleared")
        
        return jsonify({
            'success': True,
            'message': 'All documents cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        return jsonify({
            'error': 'Failed to clear documents',
            'details': str(e)
        }), 500

@documents_bp.route('/documents/stats', methods=['GET'])
def document_stats():
    """Get document processing statistics."""
    try:
        stats = {
            'total_documents': rag_state.get_document_count(),
            'total_chunks': rag_state.get_chunk_count(),
            'vector_store_count': rag_state.vector_store.get_document_count() if hasattr(rag_state.vector_store, 'get_document_count') else rag_state.get_chunk_count(),
            'analytics': rag_state.get_analytics(),
            'current_config': rag_state.get_config()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        return jsonify({
            'error': 'Failed to get document statistics',
            'details': str(e)
        }), 500