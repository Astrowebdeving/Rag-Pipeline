"""Document Routes

Routes for document upload, ingestion, and management.
"""

import logging
import traceback
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from app.services.state_service import rag_state
from app.services.file_service import file_service
from app.utils.chunking import create_chunks_with_method, create_chunks
from app.utils.embedding import create_embeddings_with_method, create_embeddings
from app.utils.auth import require_api_key
from app.augmented_generation.ollama_generator import OllamaGenerator

logger = logging.getLogger(__name__)

# Create blueprint for document routes
documents_bp = Blueprint('documents', __name__, url_prefix='/api/documents')

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

@documents_bp.route('/upload', methods=['POST'])
@require_api_key
def upload_document():
    """Upload and process document files (.txt, .pdf, etc.)."""
    temp_file_path = None
    
    try:
        # Debug logging
        logger.info(f"Upload request received - Content-Type: {request.content_type}")
        
        # Check content type
        content_type = request.content_type or ''
        if 'multipart' not in content_type.lower():
            return jsonify({'error': 'Request must be multipart/form-data'}), 400
        
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file_obj = request.files['file']
        
        # Validate file using file service
        is_valid, error_message, file_info = file_service.validate_file(file_obj)
        
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        logger.info(f"File received: {file_info['filename']}, size: {file_info['size']} bytes")
        
        # Save file to temporary location
        temp_file_path = file_service.save_temp_file(
            file_info['content'], 
            file_info['extension']
        )
        
        try:
            # Extract text from uploaded file
            text_content = file_service.extract_text_from_file(temp_file_path, file_info['filename'])
            
            # Validate extracted text
            if not text_content or not text_content.strip():
                return jsonify({'error': 'No text could be extracted from the file'}), 400
        
            # Get document ID
            doc_id = rag_state.get_next_doc_id()
            
            # Process document using current configuration
            chunks = create_chunks_with_method(
                text_content, 
                rag_state.config['chunking_method'],
                **rag_state.config
            )
            
            # Determine if we should enable metadata generation (for advanced retrieval)
            enable_metadata = rag_state.config.get('retrieval_method') == 'advanced'
            
            # Generate metadata for chunks if advanced retrieval is enabled
            chunk_metadata = None
            if enable_metadata:
                logger.info("Generating metadata for chunks with LLM analysis")
                chunk_metadata = generate_chunk_metadata(
                    chunks, doc_id, file_info['filename'], 
                    file_info['extension'],
                    enable_llm_analysis=True
                )
            
            chunk_embeddings = create_embeddings_with_method(
                chunks, 
                rag_state.config['embedding_method']
            )
            
            # Track the actual embedding method used (including fallbacks)
            # Determine which method was actually used based on embedding dimensions
            if len(chunk_embeddings) > 0:
                embedding_dim = chunk_embeddings[0].shape[0] if hasattr(chunk_embeddings[0], 'shape') else len(chunk_embeddings[0])
                if embedding_dim == 384:  # Sentence transformer dimension
                    rag_state.actual_embedding_method = 'sentence_transformer'
                else:  # TF-IDF or other
                    rag_state.actual_embedding_method = 'tfidf'
                logger.info(f"Detected actual embedding method: {rag_state.actual_embedding_method} (dim: {embedding_dim})")
            
            # Add document to state
            rag_state.add_document(
                doc_id=doc_id,
                text=text_content,
                chunks=chunks,
                filename=file_info['filename'],
                file_type=file_info['extension']
            )
            
            # Add embeddings to vector store (with metadata if available)
            if chunk_metadata:
                rag_state.vector_store.add_documents(chunks, chunk_embeddings, metadata=chunk_metadata)
            else:
                rag_state.vector_store.add_documents(chunks, chunk_embeddings)
            
            # Update analytics
            rag_state.update_analytics('chunking', method=rag_state.config['chunking_method'])
            rag_state.update_analytics('embedding', method=rag_state.config['embedding_method'])
            
            logger.info(f"File {file_info['filename']} processed as {doc_id}: {len(chunks)} chunks")
            
            response_data = {
                'message': 'File processed successfully',
                'document_id': doc_id,
                'filename': file_info['filename'],
                'chunks_created': len(chunks),
                'file_type': file_info['extension'],
                'text_length': len(text_content),
                'config_used': {
                    'chunking_method': rag_state.config['chunking_method'],
                    'embedding_method': rag_state.config['embedding_method']
                }
            }
            
            # Add metadata info if generated
            if chunk_metadata:
                response_data['metadata_generated'] = len([m for m in chunk_metadata if m['topics_analyzed']])
            
            return jsonify(response_data), 200
            
        finally:
            # Clean up temporary file
            if temp_file_path:
                file_service.cleanup_temp_file(temp_file_path)
                
    except Exception as e:
        # Clean up on error
        if temp_file_path:
            file_service.cleanup_temp_file(temp_file_path)
            
        logger.error(f"Document upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@documents_bp.route('/ingest', methods=['POST'])
@require_api_key
def ingest_document():
    """Process and store a new document from text input."""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        # Get document data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        text = data.get('text', '').strip()
        
        # Validate input
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get document ID
        doc_id = rag_state.get_next_doc_id()
        
        # Process document using current configuration
        chunks = create_chunks_with_method(
            text, 
            rag_state.config['chunking_method'],
            **rag_state.config
        )
        
        # Generate metadata if advanced retrieval is enabled
        chunk_metadata = None
        enable_metadata = rag_state.config.get('retrieval_method') == 'advanced'
        if enable_metadata:
            logger.info("Generating metadata for ingested text chunks")
            chunk_metadata = generate_chunk_metadata(
                chunks, doc_id, "text_input", "text",
                enable_llm_analysis=True
            )
        
        chunk_embeddings = create_embeddings_with_method(
            chunks, 
            rag_state.config['embedding_method']
        )
        
        # Add document to state
        rag_state.add_document(
            doc_id=doc_id,
            text=text,
            chunks=chunks
        )
        
        # Add embeddings to vector store (with metadata if available)
        if chunk_metadata:
            rag_state.vector_store.add_documents(chunks, chunk_embeddings, metadata=chunk_metadata)
        else:
            rag_state.vector_store.add_documents(chunks, chunk_embeddings)
        
        # Update analytics
        rag_state.update_analytics('chunking', method=rag_state.config['chunking_method'])
        rag_state.update_analytics('embedding', method=rag_state.config['embedding_method'])
        
        logger.info(f"Document {doc_id} successfully processed: {len(chunks)} chunks created")
        
        # Return success response with detailed information
        response_data = {
            'success': True,
            'message': f'Document processed successfully with {len(chunks)} chunks',
            'document_id': doc_id,
            'chunks_created': len(chunks),
            'text_length': len(text),
            'embedding_method': rag_state.config['embedding_method'],
            'chunking_method': rag_state.config['chunking_method'],
            'total_documents': len(rag_state.documents),
            'total_chunks': len(rag_state.all_chunks)
        }
        
        # Add metadata info if generated
        if chunk_metadata:
            response_data['metadata_generated'] = len([m for m in chunk_metadata if m['topics_analyzed']])
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        logger.error(traceback.format_exc())
        rag_state.update_analytics('error')
        return jsonify({'error': str(e)}), 500


@documents_bp.route('', methods=['GET'])
def list_documents():
    """Get list of all processed documents."""
    try:
        # Prepare document list with metadata
        document_list = []
        
        for doc_id, doc_info in rag_state.documents.items():
            document_summary = {
                'id': doc_id,
                'chunk_count': doc_info['chunk_count'],
                'text_preview': doc_info['text'][:200] + "..." if len(doc_info['text']) > 200 else doc_info['text'],
                'processed_with': doc_info['processed_with'],
                'size_bytes': len(doc_info['text'].encode('utf-8'))
            }
            
            # Add file info if available
            if 'filename' in doc_info:
                document_summary['filename'] = doc_info['filename']
            if 'file_type' in doc_info:
                document_summary['file_type'] = doc_info['file_type']
                
            document_list.append(document_summary)
        
        # Get system statistics
        stats = rag_state.get_system_stats()
        
        # Prepare response with system statistics
        response_data = {
            'documents': document_list,
            'total_documents': len(rag_state.documents),
            'total_chunks': len(rag_state.all_chunks),
            'vector_store_count': rag_state.vector_store.get_document_count(),
            'current_config': rag_state.config.copy(),
            'supported_file_types': file_service.get_supported_extensions()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({'error': f'Failed to list documents: {str(e)}'}), 500


@documents_bp.route('/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get details of a specific document."""
    try:
        if doc_id not in rag_state.documents:
            return jsonify({'error': 'Document not found'}), 404
        
        doc_info = rag_state.documents[doc_id].copy()
        
        # Add additional metadata
        doc_info['id'] = doc_id
        doc_info['size_bytes'] = len(doc_info['text'].encode('utf-8'))
        
        return jsonify(doc_info), 200
        
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@documents_bp.route('/<doc_id>', methods=['DELETE'])
@require_api_key
def delete_document(doc_id):
    """Delete a specific document and its chunks."""
    try:
        if doc_id not in rag_state.documents:
            return jsonify({'error': 'Document not found'}), 404
        
        # Get document info before deletion
        doc_info = rag_state.documents[doc_id]
        chunks_to_remove = doc_info['chunks']
        
        # Remove from state
        del rag_state.documents[doc_id]
        
        # Remove chunks from global list
        for chunk in chunks_to_remove:
            if chunk in rag_state.all_chunks:
                rag_state.all_chunks.remove(chunk)
        
        # Note: Vector store cleanup would need more sophisticated indexing
        # For now, we'll log that manual cleanup may be needed
        logger.warning(f"Document {doc_id} deleted. Vector store may need manual cleanup.")
        
        return jsonify({
            'message': f'Document {doc_id} deleted successfully',
            'chunks_removed': len(chunks_to_remove),
            'remaining_documents': len(rag_state.documents),
            'remaining_chunks': len(rag_state.all_chunks)
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500