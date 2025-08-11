"""
Ollama Generator for Local LLM Integration
Provides text generation using locally hosted Ollama models
"""

import logging
import requests
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Generator that uses Ollama for local LLM inference."""
    
    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 500,
        temperature: float = 0.7,
        timeout: int = 30
    ):
        """Initialize Ollama generator.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-2.0)
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        logger.info(f"Initializing Ollama generator with model: {model_name}")
        
        # Check if Ollama is available and model exists
        if not self._check_ollama_availability():
            raise ConnectionError("Ollama server is not available")
        
        if not self._check_model_availability():
            raise ValueError(f"Model '{model_name}' not found. Please run: ollama pull {model_name}")
        
        logger.info(f"Ollama generator initialized successfully with {model_name}")
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is running.
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            logger.error(f"Ollama server not available at {self.base_url}")
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            # Get list of available models from Ollama
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()  # Parse JSON response
                # Extract full model names from the response (including tags like :8b)
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                # Check exact match first, then try without tag if no exact match
                if self.model_name in available_models:
                    return True
                
                # If no exact match, try matching base name (e.g., "deepseek-r1" matches "deepseek-r1:8b")
                base_model_name = self.model_name.split(':')[0]
                for available_model in available_models:
                    if available_model.split(':')[0] == base_model_name:
                        return True
                
                return False
            return False  # Return False if request failed
        except requests.exceptions.RequestException:
            return False  # Return False if connection fails
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[str],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using Ollama.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved document chunks for context
            system_prompt: Optional system prompt
            
        Returns:
            Dict with 'success', 'response', and optional 'error' keys
        """
        try:
            # Build context from retrieved chunks
            context = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""
            
            # Create the prompt
            if system_prompt:
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            else:
                if context:
                    prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                else:
                    prompt = f"Question: {query}\n\nAnswer:"
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "stop": ["\n\n", "Question:", "Context:"]
                }
            }
            
            logger.debug(f"Sending request to Ollama with model: {self.model_name}")
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    logger.info(f"Generated response with {len(generated_text)} characters")
                    return {
                        'success': True,
                        'response': generated_text,
                        'model': self.model_name
                    }
                else:
                    logger.warning("Ollama returned empty response")
                    return {
                        'success': False,
                        'error': 'Empty response from Ollama',
                        'response': ''
                    }
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'response': ''
                }
                
        except requests.exceptions.Timeout:
            error_msg = f"Ollama request timeout after {self.timeout}s"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'response': ''
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama connection error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'response': ''
            }
        except Exception as e:
            error_msg = f"Unexpected error in Ollama generation: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'response': ''
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dict containing model information
        """
        return {
            'model_name': self.model_name,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'available': self._check_ollama_availability() and self._check_model_availability()
        }
    
    def is_available(self) -> bool:
        """Check if the generator is available.
        
        Returns:
            bool: True if Ollama and model are available
        """
        return self._check_ollama_availability() and self._check_model_availability()


def create_ollama_generator(
    model_name: str = "deepseek-r1:8b",
    **kwargs
) -> Optional[OllamaGenerator]:
    """Create an Ollama generator instance.
    
    Args:
        model_name: Name of the Ollama model
        **kwargs: Additional arguments for OllamaGenerator
        
    Returns:
        OllamaGenerator instance or None if creation failed
    """
    try:
        return OllamaGenerator(model_name=model_name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create Ollama generator: {e}")
        return None