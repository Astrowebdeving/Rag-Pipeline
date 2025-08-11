"""Chunking Utilities

Provides unified interface for different text chunking strategies.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# New unified interface functions (for new code)
def create_chunks(text: str, method: str = 'semantic', **kwargs) -> List[str]:
    """Create text chunks using the specified method.
    
    Args:
        text: Text to chunk
        method: Chunking method ('semantic', 'fixed_length', 'adaptive')
        **kwargs: Additional parameters for chunking methods
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    try:
        if method == 'semantic':
            from app.chunkers.semantic_chunker import chunk_by_semantics
            min_chunk_size = kwargs.get('min_chunk_size', 100)
            return chunk_by_semantics(text, min_chunk_size=min_chunk_size)
            
        elif method == 'fixed_length':
            from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
            chunk_size = kwargs.get('chunk_size', 500)
            return chunk_by_fixed_length(text, chunk_size=chunk_size)
            
        elif method == 'adaptive':
            from app.chunkers.adaptive_chunker import chunk_adaptively
            min_chunk_size = kwargs.get('min_chunk_size', 100)
            max_chunk_size = kwargs.get('max_chunk_size', 2000)
            target_chunk_size = kwargs.get('chunk_size', 800)
            return chunk_adaptively(
                text,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                target_chunk_size=target_chunk_size
            )
            
        else:
            logger.warning(f"Unknown chunking method: {method}, using fallback")
            return simple_chunk(text)
            
    except Exception as e:
        logger.error(f"Error creating chunks with method {method}: {e}")
        logger.info("Falling back to simple chunking")
        return simple_chunk(text)

# Original interface functions (for backward compatibility)
def create_chunks_with_method(text: str, method: str, **config) -> List[str]:
    """Create chunks using the specified chunking method.
    
    Args:
        text: Text to chunk
        method: Chunking method ('fixed_length', 'semantic', 'adaptive')
        **config: Configuration parameters for chunking
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If method is not supported
    """
    # Get configuration with defaults
    chunk_size = config.get('chunk_size', 800)
    min_chunk_size = config.get('min_chunk_size', 100)
    max_chunk_size = config.get('max_chunk_size', 2000)
    
    logger.info(f"Chunking text using method: {method}")
    logger.debug(f"Text length: {len(text)} characters")
    
    try:
        # Apply chunking method based on configuration
        if method == 'semantic':
            from app.chunkers.semantic_chunker import chunk_by_semantics
            chunks = chunk_by_semantics(text, min_chunk_size=min_chunk_size)
        elif method == 'adaptive':
            # Use adaptive chunking with configuration parameters
            from app.chunkers.adaptive_chunker import chunk_adaptively
            chunks = chunk_adaptively(
                text,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                target_chunk_size=chunk_size
            )
        elif method == 'fixed_length':
            from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
            chunks = chunk_by_fixed_length(text, chunk_size=chunk_size)
        else:
            # Default to fixed_length for unknown methods
            logger.warning(f"Unknown chunking method '{method}', defaulting to fixed_length")
            from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
            chunks = chunk_by_fixed_length(text, chunk_size=chunk_size)
        
        logger.info(f"Created {len(chunks)} chunks using {method} method")
        
        # Log chunk statistics
        if chunks:
            chunk_lengths = [len(chunk) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.debug(f"Average chunk length: {avg_length:.1f} characters")
            logger.debug(f"Chunk length range: {min(chunk_lengths)} - {max(chunk_lengths)}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Chunking failed with method {method}: {str(e)}")
        # Fallback to simple fixed-length chunking
        logger.info("Falling back to fixed-length chunking")
        from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
        return chunk_by_fixed_length(text, chunk_size=500)

def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple fallback chunking method.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Try to break at sentence or word boundary
        if end < text_length:
            # Look for sentence boundary
            sentence_break = text.rfind('.', start, end)
            if sentence_break > start:
                end = sentence_break + 1
            else:
                # Look for word boundary
                word_break = text.rfind(' ', start, end)
                if word_break > start:
                    end = word_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)
        
        # Avoid infinite loop
        if start >= end:
            start = end
    
    return chunks

def get_available_methods() -> List[str]:
    """Get list of available chunking methods.
    
    Returns:
        List of available chunking method names
    """
    return ['fixed_length', 'semantic', 'adaptive']

def validate_chunking_config(config: dict) -> bool:
    """Validate chunking configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check method is valid
        method = config.get('chunking_method')
        if method and method not in get_available_methods():
            logger.error(f"Invalid chunking method: {method}")
            return False
        
        # Check numeric parameters
        chunk_size = config.get('chunk_size', 800)
        min_chunk_size = config.get('min_chunk_size', 100)
        max_chunk_size = config.get('max_chunk_size', 2000)
        
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            logger.error(f"Invalid chunk_size: {chunk_size}")
            return False
            
        if not isinstance(min_chunk_size, int) or min_chunk_size <= 0:
            logger.error(f"Invalid min_chunk_size: {min_chunk_size}")
            return False
            
        if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
            logger.error(f"Invalid max_chunk_size: {max_chunk_size}")
            return False
        
        # Check logical relationships
        if min_chunk_size >= max_chunk_size:
            logger.error(f"min_chunk_size ({min_chunk_size}) must be less than max_chunk_size ({max_chunk_size})")
            return False
            
        if chunk_size < min_chunk_size or chunk_size > max_chunk_size:
            logger.error(f"chunk_size ({chunk_size}) must be between min_chunk_size ({min_chunk_size}) and max_chunk_size ({max_chunk_size})")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating chunking config: {str(e)}")
        return False

def get_chunking_stats(chunks: List[str]) -> dict:
    """Get statistics about a list of chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary containing chunk statistics
    """
    if not chunks:
        return {
            'count': 0,
            'total_length': 0,
            'average_length': 0,
            'min_length': 0,
            'max_length': 0
        }
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    return {
        'count': len(chunks),
        'total_length': sum(chunk_lengths),
        'average_length': sum(chunk_lengths) / len(chunk_lengths),
        'min_length': min(chunk_lengths),
        'max_length': max(chunk_lengths)
    }

def is_method_available(method: str) -> bool:
    """Check if a chunking method is available.
    
    Args:
        method: Chunking method to check
        
    Returns:
        True if method is available, False otherwise
    """
    try:
        if method == 'semantic':
            from app.chunkers.semantic_chunker import chunk_by_semantics
            return True
            
        elif method == 'fixed_length':
            from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
            return True
            
        elif method == 'adaptive':
            from app.chunkers.adaptive_chunker import chunk_adaptively
            return True
            
        else:
            return False
            
    except ImportError:
        return False
    except Exception as e:
        logger.error(f"Error checking availability of chunking method {method}: {e}")
        return False

# Additional functions for compatibility with main.py
def get_chunking_method() -> str:
    """Interactive method selection for chunking (for main.py compatibility).
    
    Returns:
        str: Selected chunking method
    """
    print("\n=== Chunking Method Selection ===")
    print("1. Semantic (Recommended)")
    print("2. Fixed Length")
    print("3. Adaptive")
    
    while True:
        choice = input("Select chunking method (1-3): ").strip()
        if choice == '1':
            return 'semantic'
        elif choice == '2':
            return 'fixed_length'
        elif choice == '3':
            return 'adaptive'
        else:
            print("Invalid choice. Please select 1, 2, or 3.")