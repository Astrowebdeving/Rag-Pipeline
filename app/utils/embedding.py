"""
Unified Embedding Interface
Provides a consistent interface for different embedding methods
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

def create_embeddings(texts: List[str], method: str = 'sentence_transformer') -> Optional[List[np.ndarray]]:
    """Create embeddings for a list of texts using the specified method.
    
    Args:
        texts: List of texts to embed
        method: Embedding method ('sentence_transformer', 'tfidf', 'huggingface')
        
    Returns:
        List of embeddings or None if failed
    """
    if not texts:
        return []
    
    try:
        if method == 'sentence_transformer':
            from app.embedders.sentence_transformer_embedder import create_embeddings as st_create
            return st_create(texts)
            
        elif method == 'tfidf':
            from app.embedders.tfidf_embedder import TFIDFEmbedder
            embedder = TFIDFEmbedder()
            return embedder.embed_batch(texts)
            
        elif method == 'huggingface':
            from app.embedders.huggingface_embedder import HuggingFaceEmbedder
            embedder = HuggingFaceEmbedder()
            return embedder.embed_batch(texts)
            
        else:
            logger.error(f"Unknown embedding method: {method}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating embeddings with method {method}: {e}")
        return None

def create_single_embedding(text: str, method: str = 'sentence_transformer') -> Optional[np.ndarray]:
    """Create embedding for a single text using the specified method.
    
    Args:
        text: Text to embed
        method: Embedding method ('sentence_transformer', 'tfidf', 'huggingface')
        
    Returns:
        Embedding as numpy array or None if failed
    """
    if not text or not text.strip():
        return None
    
    try:
        if method == 'sentence_transformer':
            from app.embedders.sentence_transformer_embedder import create_single_embedding as st_create
            return st_create(text)
            
        elif method == 'tfidf':
            from app.embedders.tfidf_embedder import TFIDFEmbedder
            embedder = TFIDFEmbedder()
            return embedder.embed(text)
            
        elif method == 'huggingface':
            from app.embedders.huggingface_embedder import HuggingFaceEmbedder
            embedder = HuggingFaceEmbedder()
            return embedder.embed(text)
            
        else:
            logger.error(f"Unknown embedding method: {method}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating single embedding with method {method}: {e}")
        return None

def get_embedding_dimension(method: str = 'sentence_transformer') -> Optional[int]:
    """Get the embedding dimension for a specific method.
    
    Args:
        method: Embedding method
        
    Returns:
        Embedding dimension or None if unknown
    """
    try:
        if method == 'sentence_transformer':
            from app.embedders.sentence_transformer_embedder import create_embedder
            embedder = create_embedder()
            return embedder.get_dimension() if embedder else None
            
        elif method == 'tfidf':
            # TF-IDF dimension depends on vocabulary size, return None
            return None
            
        elif method == 'huggingface':
            from app.embedders.huggingface_embedder import HuggingFaceEmbedder
            embedder = HuggingFaceEmbedder()
            return embedder.get_dimension()
            
        else:
            logger.error(f"Unknown embedding method: {method}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting embedding dimension for method {method}: {e}")
        return None

def is_method_available(method: str) -> bool:
    """Check if an embedding method is available.
    
    Args:
        method: Embedding method to check
        
    Returns:
        True if method is available, False otherwise
    """
    try:
        if method == 'sentence_transformer':
            from app.embedders.sentence_transformer_embedder import create_embedder
            embedder = create_embedder()
            return embedder is not None and embedder.is_available()
            
        elif method == 'tfidf':
            from app.embedders.tfidf_embedder import TFIDFEmbedder
            embedder = TFIDFEmbedder()
            return embedder.is_available()
            
        elif method == 'huggingface':
            from app.embedders.huggingface_embedder import HuggingFaceEmbedder
            embedder = HuggingFaceEmbedder()
            return embedder.is_available()
            
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error checking availability of method {method}: {e}")
        return False

# Additional functions for compatibility with main.py
def create_query_embedding(query: str, method: str) -> Optional[np.ndarray]:
    """Create query embedding using selected method (for main.py compatibility).
    
    Args:
        query: Query text
        method: Embedding method
        
    Returns:
        np.ndarray: Query embedding or None if failed
    """
    return create_single_embedding(query, method)

def get_embedding_method() -> str:
    """Interactive method selection for embeddings (for main.py compatibility).
    
    Returns:
        str: Selected embedding method
    """
    print("\n=== Embedding Method Selection ===")
    print("1. SentenceTransformer (Recommended)")
    print("2. TF-IDF (Fast)")
    print("3. HuggingFace")
    
    while True:
        choice = input("Select embedding method (1-3): ").strip()
        if choice == '1':
            return 'sentence_transformer'
        elif choice == '2':
            return 'tfidf'
        elif choice == '3':
            return 'huggingface'
        else:
            print("Invalid choice. Please select 1, 2, or 3.")