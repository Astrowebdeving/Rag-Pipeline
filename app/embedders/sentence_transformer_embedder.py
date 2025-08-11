"""SentenceTransformer Embedder with Resource Cleanup

This module provides embedding functionality using SentenceTransformer models
with proper resource management to prevent memory leaks and semaphore issues.
"""

import os
import logging
import multiprocessing
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Global model instance and device tracking
_model = None
_model_device = None

def cleanup_model() -> None:
    """Properly cleanup the SentenceTransformer model to prevent resource leaks."""
    global _model, _model_device
    if _model is not None:
        try:
            logger.info("Cleaning up SentenceTransformer model resources...")
            
            # Move model to CPU if possible
            if hasattr(_model, '_modules'):
                try:
                    _model.to('cpu')
                    logger.debug("Moved model to CPU")
                except Exception as e:
                    logger.debug(f"Could not move model to CPU: {e}")
            
            # Clear model cache if available
            if hasattr(_model, '_cache'):
                try:
                    _model._cache.clear()
                    logger.debug("Cleared model cache")
                except Exception as e:
                    logger.debug(f"Could not clear model cache: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if using GPU
            if _model_device in ['cuda', 'mps']:
                try:
                    import torch
                    if _model_device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("Cleared CUDA cache")
                    elif _model_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        logger.debug("Cleared MPS cache")
                except Exception as e:
                    logger.debug(f"Could not clear GPU cache: {e}")
            
            logger.info("SentenceTransformer model cleanup completed")
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
        finally:
            _model = None
            _model_device = None

def _initialize_model(model_name: str = 'all-MiniLM-L6-v2') -> bool:
    """Initialize the SentenceTransformer model with proper resource management.
    
    Args:
        model_name: Name of the SentenceTransformer model to load
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _model, _model_device
    
    try:
        # Set environment variables to prevent multiprocessing issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Set multiprocessing start method to 'spawn' to prevent semaphore leaks
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                logger.debug("Set multiprocessing start method to 'spawn'")
            except RuntimeError:
                logger.debug("Could not set multiprocessing start method (already set)")
        
        # Import SentenceTransformer
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        _model = SentenceTransformer(model_name)
        
        # Determine device
        import torch
        if torch.cuda.is_available():
            _model_device = 'cuda'
            _model = _model.to('cuda')
            logger.info("Using CUDA for SentenceTransformer")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _model_device = 'mps'
            _model = _model.to('mps')
            logger.info("Using MPS for SentenceTransformer")
        else:
            _model_device = 'cpu'
            logger.info("Using CPU for SentenceTransformer")
        
        logger.info(f"SentenceTransformer model loaded successfully on {_model_device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformer model: {e}")
        _model = None
        _model_device = None
        return False

def reset_model() -> None:
    """Reset the model (cleanup and reinitialize)."""
    cleanup_model()
    _initialize_model()

class SentenceTransformerEmbedder:
    """SentenceTransformer-based embedder with resource management."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model
        """
        self.model_name = model_name
        self.dimension = None
        
        # Initialize model if not already done
        if _model is None:
            if not _initialize_model(model_name):
                raise RuntimeError(f"Failed to initialize SentenceTransformer model: {model_name}")
        
        # Get embedding dimension
        try:
            test_embedding = self.embed("test")
            if test_embedding is not None:
                self.dimension = len(test_embedding)
                logger.info(f"SentenceTransformer embedding dimension: {self.dimension}")
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
    
    def embed(self, text: Union[str, List[str]]) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """Create embeddings for text(s).
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Embedding(s) as numpy array(s) or None if failed
        """
        global _model
        
        if _model is None:
            logger.error("SentenceTransformer model not initialized")
            return None
        
        try:
            # Handle single text vs list of texts
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Create embeddings
            embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            # Return single embedding or list of numpy arrays
            if is_single:
                return embeddings[0] if len(embeddings) > 0 else None
            else:
                # Convert to list of individual numpy arrays
                return [np.array(emb) for emb in embeddings]
                
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> Optional[List[np.ndarray]]:
        """Create embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings or None if failed
        """
        if not texts:
            return []
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embed(batch)
                
                if batch_embeddings is None:
                    logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}")
                    return None
                
                all_embeddings.extend(batch_embeddings)
                
                # Log progress for large batches
                if len(texts) > 100 and (i + batch_size) % 100 == 0:
                    logger.info(f"Processed {i + batch_size}/{len(texts)} texts")
            
            logger.info(f"Created embeddings for {len(all_embeddings)} texts")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return None
    
    def get_dimension(self) -> Optional[int]:
        """Get embedding dimension.
        
        Returns:
            int: Embedding dimension or None if unknown
        """
        return self.dimension
    
    def is_available(self) -> bool:
        """Check if the embedder is available and working.
        
        Returns:
            bool: True if embedder is working
        """
        global _model
        return _model is not None

# Factory function for creating embedders
def create_embedder(model_name: str = 'all-MiniLM-L6-v2') -> Optional[SentenceTransformerEmbedder]:
    """Create a SentenceTransformer embedder.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        SentenceTransformerEmbedder instance or None if failed
    """
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as e:
        logger.error(f"Failed to create SentenceTransformer embedder: {e}")
        return None

# Utility functions for direct embedding
def create_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> Optional[List[np.ndarray]]:
    """Create embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model_name: SentenceTransformer model name
        
    Returns:
        List of embeddings or None if failed
    """
    embedder = create_embedder(model_name)
    if embedder is None:
        return None
    return embedder.embed_batch(texts)

def create_single_embedding(text: str, model_name: str = 'all-MiniLM-L6-v2') -> Optional[np.ndarray]:
    """Create embedding for a single text.
    
    Args:
        text: Text to embed
        model_name: SentenceTransformer model name
        
    Returns:
        Embedding as numpy array or None if failed
    """
    embedder = create_embedder(model_name)
    if embedder is None:
        return None
    return embedder.embed(text)