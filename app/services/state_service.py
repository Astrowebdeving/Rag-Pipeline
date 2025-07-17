"""RAG System State Management Service

Manages the global state of the RAG system including configuration,
document storage, analytics, and cached models.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from app.vector_db.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGSystemState:
    """Manages the current state of the RAG system including configuration and data."""
    
    def __init__(self):
        """Initialize RAG system with default configuration."""
        # Enhanced configuration with more options
        self.config = {
            'chunking_method': 'adaptive',  # Use adaptive chunking for better performance
            'embedding_method': 'sentence_transformer',  # Use neural embeddings for better semantic understanding
            'retrieval_method': 'advanced',  # Use advanced retrieval with re-ranking
            'generation_method': 'huggingface',  # Enable AI generation for synthesized answers
            'generation_model': 'google/flan-t5-large',  # Use FLAN-T5-large for better technical synthesis
            # Advanced configuration options
            'chunk_size': 800,  # Target chunk size for adaptive chunking
            'min_chunk_size': 100,  # Minimum chunk size
            'max_chunk_size': 2000,  # Maximum chunk size
            'retrieval_top_k': 5,  # Number of chunks to retrieve
            'enable_query_expansion': True,  # Enable query expansion in advanced retrieval
            'enable_reranking': True,  # Enable result re-ranking
            'diversity_factor': 0.3,  # Diversity factor for result filtering
            'batch_size': 1000,  # Batch size for processing large datasets
            'enable_analytics': True,  # Enable analytics tracking
            'cache_embeddings': True  # Enable embedding caching
        }
        
        # Log the configuration being loaded for debugging
        logger.info(f"RAG system initialized with enhanced configuration: {self.config}")
        
        # System data storage
        self.documents = {}  # Store processed documents by ID
        self.vector_store = FAISSVectorStore(batch_size=self.config['batch_size'])
        self.chunk_embeddings = []  # Store all chunk embeddings
        self.all_chunks = []  # Store all text chunks
        self.next_doc_id = 1  # Counter for document IDs
        self.actual_embedding_method = None  # Track the actually used embedding method
        
        # Cached models to avoid reloading
        self.cached_generator = None  # Cache LLM generator to avoid reloading
        self.cached_generator_type = None  # Track which generator is cached
        
        # Analytics and monitoring
        self.analytics = {
            'total_documents_processed': 0,  # Total documents ingested
            'total_queries_processed': 0,  # Total queries handled
            'total_chunks_created': 0,  # Total chunks generated
            'average_response_time': 0.0,  # Average query response time
            'last_activity_timestamp': None,  # Last system activity
            'error_count': 0,  # Number of errors encountered
            'retrieval_stats': {  # Retrieval method usage statistics
                'dense': 0,
                'hybrid': 0,
                'advanced': 0
            },
            'chunking_stats': {  # Chunking method usage statistics
                'fixed_length': 0,
                'semantic': 0,
                'adaptive': 0
            },
            'embedding_stats': {  # Embedding method usage statistics
                'tfidf': 0,
                'sentence_transformer': 0
            }
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_usage': {},  # Memory usage tracking
            'processing_times': [],  # Processing time history
            'cache_hit_rate': 0.0,  # Embedding cache hit rate
            'system_health_score': 1.0  # Overall system health (0-1)
        }

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update system configuration with validation.
        
        Args:
            new_config: Dictionary of new configuration values
            
        Returns:
            bool: True if update successful, False otherwise
        """
        # Validate configuration values
        valid_chunking = ['semantic', 'fixed_length', 'adaptive']
        valid_embedding = ['tfidf', 'sentence_transformer']
        valid_retrieval = ['dense', 'hybrid', 'advanced']
        valid_generation = ['huggingface', 'none']
        
        # Check if all provided values are valid
        for key, value in new_config.items():
            if key == 'chunking_method' and value not in valid_chunking:
                logger.error(f"Invalid chunking method: {value}")
                return False
            elif key == 'embedding_method' and value not in valid_embedding:
                logger.error(f"Invalid embedding method: {value}")
                return False
            elif key == 'retrieval_method' and value not in valid_retrieval:
                logger.error(f"Invalid retrieval method: {value}")
                return False
            elif key == 'generation_method' and value not in valid_generation:
                logger.error(f"Invalid generation method: {value}")
                return False
        
        # Update configuration with valid values
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Clear cached generator if generation method changed
        if 'generation_method' in new_config or 'generation_model' in new_config:
            self.cached_generator = None
            self.cached_generator_type = None
            logger.info("Cleared cached generator due to config change")
        
        logger.info(f"Configuration updated: {new_config}")
        return True

    def add_document(self, doc_id: str, text: str, chunks: List[str], filename: Optional[str] = None, file_type: Optional[str] = None):
        """Add a processed document to the system state.
        
        Args:
            doc_id: Unique document identifier
            text: Original document text
            chunks: Text chunks from the document
            filename: Optional filename
            file_type: Optional file type
        """
        # Store document information
        doc_info = {
            'text': text,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'processed_with': self.config.copy()
        }
        
        if filename:
            doc_info['filename'] = filename
        if file_type:
            doc_info['file_type'] = file_type
            
        self.documents[doc_id] = doc_info
        
        # Add chunks to global storage
        self.all_chunks.extend(chunks)
        
        # Update analytics if enabled
        if self.config.get('enable_analytics', True):
            self.analytics['total_documents_processed'] += 1
            self.analytics['total_chunks_created'] += len(chunks)
            self.analytics['last_activity_timestamp'] = str(datetime.now())
        
        logger.info(f"Document {doc_id} added: {len(chunks)} chunks")

    def get_next_doc_id(self) -> str:
        """Generate next document ID.
        
        Returns:
            str: Next available document ID
        """
        doc_id = f"doc_{self.next_doc_id}"
        self.next_doc_id += 1
        return doc_id

    def update_analytics(self, operation: str, **kwargs):
        """Update analytics counters.
        
        Args:
            operation: Type of operation ('query', 'error', etc.)
            **kwargs: Additional analytics data
        """
        if not self.config.get('enable_analytics', True):
            return
            
        if operation == 'query':
            self.analytics['total_queries_processed'] += 1
        elif operation == 'error':
            self.analytics['error_count'] += 1
        elif operation == 'chunking':
            method = kwargs.get('method', 'unknown')
            if method in self.analytics['chunking_stats']:
                self.analytics['chunking_stats'][method] += 1
        elif operation == 'embedding':
            method = kwargs.get('method', 'unknown')
            if method in self.analytics['embedding_stats']:
                self.analytics['embedding_stats'][method] += 1
        elif operation == 'retrieval':
            method = kwargs.get('method', 'unknown')
            if method in self.analytics['retrieval_stats']:
                self.analytics['retrieval_stats'][method] += 1
        
        self.analytics['last_activity_timestamp'] = str(datetime.now())

    def reset_analytics(self):
        """Reset all analytics counters."""
        self.analytics = {
            'total_documents_processed': 0,
            'total_queries_processed': 0,
            'total_chunks_created': 0,
            'average_response_time': 0.0,
            'last_activity_timestamp': None,
            'error_count': 0,
            'retrieval_stats': {'dense': 0, 'hybrid': 0, 'advanced': 0},
            'chunking_stats': {'fixed_length': 0, 'semantic': 0, 'adaptive': 0},
            'embedding_stats': {'tfidf': 0, 'sentence_transformer': 0}
        }
        
        self.performance_metrics = {
            'memory_usage': {},
            'processing_times': [],
            'cache_hit_rate': 0.0,
            'system_health_score': 1.0
        }
        
        logger.info("Analytics counters reset")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics.
        
        Returns:
            Dict containing system statistics
        """
        # Calculate health score
        total_operations = (self.analytics['total_documents_processed'] + 
                          self.analytics['total_queries_processed'])
        health_score = 1.0
        
        if total_operations > 0:
            error_rate = self.analytics['error_count'] / total_operations
            health_score -= min(error_rate * 2, 0.5)
        
        self.performance_metrics['system_health_score'] = max(0.0, health_score)
        
        return {
            'documents_count': len(self.documents),
            'chunks_count': len(self.all_chunks),
            'vector_store_count': self.vector_store.get_document_count(),
            'config': self.config.copy(),
            'analytics': self.analytics.copy(),
            'performance': self.performance_metrics.copy(),
            'health_score': health_score
        }


# Global instance
rag_state = RAGSystemState() 