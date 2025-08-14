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
        max_tokens: int = 3000,
        temperature: float = 0.3,
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
        system_prompt: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
        suppress_info_log: bool = False,
        allow_partial_on_timeout: bool = False,
        extra_metadata: Optional[List[Dict[str, Any]]] = None,
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
            
            # Create the prompt (support system prompt)
            if system_prompt:
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\n"
            else:
                if context:
                    prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\n"
                else:
                    prompt = ""

            # Optionally add metadata summaries (if provided)
            if extra_metadata:
                try:
                    # Compact metadata block
                    meta_lines = []
                    for md in extra_metadata[:5]:
                        kws = ", ".join(md.get('lex_keywords', [])[:6])
                        desc = ", ".join(md.get('lex_descriptors', [])[:4])
                        tone = ", ".join(md.get('lex_tone', [])[:3])
                        meta_lines.append(f"Keywords: {kws}\nDescriptors: {desc}\nTone: {tone}")
                    meta_block = "\n\nMetadata (keywords/descriptors/tone):\n" + "\n---\n".join(meta_lines)
                    prompt += meta_block
                except Exception:
                    pass

            prompt += f"\n\nQuestion: {query}\n\nAnswer:"
            
            # Prepare request payload
            # For DeepSeek-R1, we normally use higher token limits to allow thinking + final answer
            # but honor per-call overrides for short tasks (query enhancement/metadata).
            is_deepseek = "deepseek" in self.model_name.lower()
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": bool(allow_partial_on_timeout),
                "options": {
                    "num_predict": (
                        int(max_tokens_override)
                        if max_tokens_override is not None
                        else (max(self.max_tokens, 3000) if is_deepseek else self.max_tokens)
                    ),
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    # Remove stop tokens for DeepSeek-R1 to allow complete thinking process
                    "stop": [] if is_deepseek else ["\n\n", "Question:", "Context:"]
                }
            }
            
            logger.debug(f"Sending request to Ollama with model: {self.model_name}")
            
            # Make request to Ollama (streaming if partials allowed)
            if allow_partial_on_timeout:
                partial_text = ""
                try:
                    with requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                        stream=True,
                    ) as response:
                        if response.status_code != 200:
                            error_msg = f"Ollama API error: {response.status_code}"
                            logger.error(error_msg)
                            return { 'success': False, 'error': error_msg, 'response': '' }
                        for line in response.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            chunk = data.get('response', '')
                            if chunk:
                                partial_text += chunk
                            if data.get('done'):
                                break
                except requests.exceptions.ReadTimeout:
                    logger.warning(f"Ollama stream read timeout after {self.timeout}s; returning partial")
                except requests.exceptions.Timeout:
                    logger.warning(f"Ollama request timeout after {self.timeout}s; returning partial")

                partial_text = partial_text.strip()
                if partial_text:
                    final_answer = self._extract_final_answer(partial_text)
                    log_msg = (
                        f"Generated (partial-capable) response with {len(partial_text)} characters, final answer: {len(final_answer)} characters"
                    )
                    if suppress_info_log:
                        logger.debug(log_msg)
                    else:
                        logger.info(log_msg)
                    return { 'success': True, 'response': final_answer, 'model': self.model_name }
                else:
                    return { 'success': False, 'error': 'Empty response from Ollama', 'response': '' }

            # Non-streaming path (no partials)
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()

                if generated_text:
                    # Special handling for DeepSeek-R1 models that use thinking process
                    final_answer = self._extract_final_answer(generated_text)
                    log_msg = (
                        f"Generated response with {len(generated_text)} characters, final answer: {len(final_answer)} characters"
                    )
                    if suppress_info_log:
                        logger.debug(log_msg)
                    else:
                        logger.info(log_msg)
                    return {
                        'success': True,
                        'response': final_answer,
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
            # Best-effort partial extraction: try to return any partial text if available
            # Note: requests timeout aborts before body is read; if server streamed, partial text would be lost.
            # For true partials, switch to streaming mode and accumulate chunks.
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
    
    def _extract_final_answer(self, generated_text: str) -> str:
        """Extract the final answer from DeepSeek-R1 response, removing thinking process.
        
        DeepSeek-R1 models generate responses in this format:
        <think>
        [internal reasoning process]
        </think>
        
        [final answer]
        
        This method extracts only the final answer part.
        """
        try:
            # Check if this is a DeepSeek-R1 response with thinking tags
            if '<think>' in generated_text:
                if '</think>' in generated_text:
                    # Complete thinking process - extract final answer
                    think_end = generated_text.find('</think>')
                    final_answer = generated_text[think_end + len('</think>'):].strip()
                    
                    if final_answer and len(final_answer) > 10:  # Ensure meaningful content
                        logger.debug(f"Extracted final answer from DeepSeek-R1: {len(final_answer)} chars")
                        return final_answer
                    else:
                        logger.warning("DeepSeek-R1 completed thinking but no substantial final answer found")
                        # Try to extract the last meaningful sentence from thinking process
                        thinking_content = generated_text[generated_text.find('<think>') + 7:think_end].strip()
                        sentences = thinking_content.split('.')
                        if sentences and len(sentences) > 1:
                            last_sentence = sentences[-2].strip() + '.'
                            if len(last_sentence) > 20:
                                logger.info("Using last meaningful sentence from thinking process as answer")
                                return last_sentence
                        return "I need to process this further. Please increase the token limit for a complete answer."
                else:
                    # Incomplete thinking process - response was cut off
                    logger.warning("DeepSeek-R1 response cut off during thinking process")
                    
                    # Try to extract the most relevant part of the thinking process
                    thinking_start = generated_text.find('<think>') + 7
                    thinking_content = generated_text[thinking_start:].strip()
                    
                    # Look for conclusion-like phrases in the thinking
                    conclusion_phrases = ["In conclusion", "So the answer", "Therefore", "The result is", "Final answer"]
                    for phrase in conclusion_phrases:
                        if phrase.lower() in thinking_content.lower():
                            start_idx = thinking_content.lower().find(phrase.lower())
                            conclusion = thinking_content[start_idx:start_idx + 200].strip()
                            if conclusion:
                                logger.info("Extracted conclusion from incomplete thinking process")
                                return conclusion
                    
                    return "The response was cut off during processing. Please increase the token limit (current: {}) or timeout for a complete answer.".format(self.max_tokens)
            
            # For non-DeepSeek models or responses without thinking tags, return as-is
            return generated_text
            
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            # Return original text if extraction fails
            return generated_text


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