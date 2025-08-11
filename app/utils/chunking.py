"""
Unified Chunking Interface
Provides a consistent interface for different chunking methods
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

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
            from app.chunkers.semantic_chunker import SemanticChunker
            chunker = SemanticChunker()
            return chunker.chunk_text(text)
            
        elif method == 'fixed_length':
            from app.chunkers.fixed_length_chunker import FixedLengthChunker
            chunk_size = kwargs.get('chunk_size', 500)
            overlap = kwargs.get('overlap', 50)
            chunker = FixedLengthChunker(chunk_size=chunk_size, overlap=overlap)
            return chunker.chunk_text(text)
            
        elif method == 'adaptive':
            from app.chunkers.adaptive_chunker import AdaptiveChunker
            chunker = AdaptiveChunker()
            return chunker.chunk_text(text)
            
        else:
            logger.warning(f"Unknown chunking method: {method}, using fallback")
            return simple_chunk(text)
            
    except Exception as e:
        logger.error(f"Error creating chunks with method {method}: {e}")
        logger.info("Falling back to simple chunking")
        return simple_chunk(text)

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

def is_method_available(method: str) -> bool:
    """Check if a chunking method is available.
    
    Args:
        method: Chunking method to check
        
    Returns:
        True if method is available, False otherwise
    """
    try:
        if method == 'semantic':
            from app.chunkers.semantic_chunker import SemanticChunker
            return True
            
        elif method == 'fixed_length':
            from app.chunkers.fixed_length_chunker import FixedLengthChunker
            return True
            
        elif method == 'adaptive':
            from app.chunkers.adaptive_chunker import AdaptiveChunker
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