"""
Ollama Backend Implementation

This module implements the OllamaBackend class that provides local AI processing
capabilities using Ollama with the embeddinggemma:latest model. It allows users
to run AI-powered document Q&A entirely on their local machine without requiring
internet connectivity or external API keys.
"""

import requests
import json
import logging
import time
from typing import Dict, Any, Optional
from requests.exceptions import ConnectionError, Timeout, RequestException
from ai_backend import AIBackend
from ai_backend_errors import (
    ServiceUnavailableError, ModelNotFoundError, InvalidModelError,
    ProcessingTimeoutError, NetworkError, ErrorMessageGenerator, RetryManager
)


class OllamaBackend(AIBackend):
    """
    Backend implementation for local Ollama AI processing.
    
    This backend communicates with a local Ollama instance running on localhost:11434
    and uses the embeddinggemma:latest model for text processing and question answering.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2:1b"):
        """
        Initialize the Ollama backend.
        
        Args:
            base_url (str): The base URL for the Ollama API (default: http://localhost:11434)
            model_name (str): The model to use for processing (default: gemma2:2b)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
        # Timeout configuration for different operations
        self.connection_timeout = 5   # Quick timeout for availability checks
        self.processing_timeout = 300  # Longer timeout for actual processing (5 minutes)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize retry manager for transient failures
        self.retry_manager = RetryManager(max_retries=1, base_delay=1.0)
        
        # List of preferred models in order of preference
        self.preferred_models = [
            "llama3.2:1b",    # Fast, good quality
            "gemma2:2b",      # Fast and efficient, good quality
            "phi3:mini",      # Very small and fast
            "gemma2:9b",      # Larger, better quality
            "llama3.2:3b",    # Good balance
            "qwen2.5:3b",     # Alternative option
        ]
        
        # Track last availability check to avoid excessive polling
        self._last_availability_check = 0
        self._availability_cache = False
        self._availability_cache_duration = 30  # Cache for 30 seconds

    def is_available(self) -> bool:
        """
        Check if Ollama service is available and a suitable model is installed.
        Uses caching to avoid excessive network calls.
        
        Returns:
            bool: True if Ollama is running and a suitable model is available
        """
        current_time = time.time()
        
        # Use cached result if recent
        if (current_time - self._last_availability_check) < self._availability_cache_duration:
            return self._availability_cache
        
        try:
            # First check if Ollama service is running
            if not self._check_ollama_service():
                self.logger.debug("Ollama service is not running on %s", self.base_url)
                self._update_availability_cache(False)
                return False
            
            # Then check if a suitable model is available
            if not self._check_model_availability():
                self.logger.debug("No suitable model available in Ollama")
                self._update_availability_cache(False)
                return False
            
            self._update_availability_cache(True)
            return True
            
        except Exception as e:
            self.logger.error("Error checking Ollama availability: %s", str(e))
            self._update_availability_cache(False)
            return False
    
    def _update_availability_cache(self, available: bool) -> None:
        """Update availability cache with timestamp."""
        self._last_availability_check = time.time()
        self._availability_cache = available

    def _check_ollama_service(self) -> bool:
        """
        Check if Ollama service is running by making a request to the base endpoint.
        
        Returns:
            bool: True if service is running, False otherwise
            
        Raises:
            ServiceUnavailableError: If service check fails with specific error details
        """
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=self.connection_timeout)
            if response.status_code == 200:
                return True
            else:
                self.logger.warning(f"Ollama service returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            self.logger.debug(f"Connection error checking Ollama service: {e}")
            return False
        except requests.exceptions.Timeout as e:
            self.logger.debug(f"Timeout checking Ollama service: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Unexpected error checking Ollama service: {e}")
            return False

    def _check_model_availability(self) -> bool:
        """
        Check if a suitable text generation model is available in Ollama.
        
        Returns:
            bool: True if a suitable model is available, False otherwise
            
        Raises:
            ModelNotFoundError: If no suitable models are found
            InvalidModelError: If only embedding models are available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.connection_timeout)
            if response.status_code != 200:
                self.logger.warning(f"Failed to get model list, status: {response.status_code}")
                return False
            
            models_data = response.json()
            available_models = [model.get('name', '') for model in models_data.get('models', [])]
            
            if not available_models:
                self.logger.info("No models installed in Ollama")
                return False
            
            # First check if the specified model is available
            if self.model_name in available_models:
                # Verify it's not an embedding model
                if "embedding" in self.model_name.lower():
                    self.logger.warning(f"Model {self.model_name} is an embedding model, looking for alternatives")
                else:
                    return True
            
            # Look for suitable alternatives from our preferred list
            for preferred_model in self.preferred_models:
                if preferred_model in available_models:
                    self.logger.info(f"Using alternative model {preferred_model} instead of {self.model_name}")
                    self.model_name = preferred_model
                    return True
            
            # Check if any text generation models are available (not embedding models)
            text_gen_models = [model for model in available_models 
                             if not any(keyword in model.lower() 
                                      for keyword in ['embedding', 'embed', 'nomic-embed'])]
            
            if text_gen_models:
                # Use the first available text generation model
                selected_model = text_gen_models[0]
                self.logger.info(f"Using available model {selected_model} instead of {self.model_name}")
                self.model_name = selected_model
                return True
            
            # Only embedding models available
            embedding_models = [model for model in available_models 
                              if any(keyword in model.lower() 
                                   for keyword in ['embedding', 'embed', 'nomic-embed'])]
            
            if embedding_models:
                self.logger.warning(f"Only embedding models available: {embedding_models}")
            
            return False
            
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Network error checking models: {e}")
            return False
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON response from Ollama: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking model availability: {e}")
            return False

    def process_question(self, text: str, question: str) -> str:
        """
        Process a question against the given text context using Ollama with retry logic.
        
        Args:
            text (str): The context text to analyze
            question (str): The user's question
            
        Returns:
            str: The AI-generated answer
            
        Raises:
            ValueError: If inputs are invalid
            ServiceUnavailableError: If Ollama service is unavailable
            ModelNotFoundError: If no suitable model is available
            ProcessingTimeoutError: If processing times out
            NetworkError: If network communication fails
        """
        # Validate inputs using the parent class method
        self.validate_inputs(text, question)
        
        # Check if service is available before processing
        if not self.is_available():
            # Provide specific error based on what's wrong
            if not self._check_ollama_service():
                raise ServiceUnavailableError("Ollama", {
                    "base_url": self.base_url,
                    "message": ErrorMessageGenerator.get_ollama_service_unavailable_message()
                })
            else:
                # Service is running but no suitable model
                available_models = self._get_available_models()
                if any("embedding" in model.lower() for model in available_models):
                    raise InvalidModelError(
                        "embeddinggemma:latest", 
                        "This is an embedding model, not suitable for text generation",
                        self.preferred_models
                    )
                else:
                    raise ModelNotFoundError(
                        self.model_name, 
                        "Ollama", 
                        f"ollama pull {self.preferred_models[0]}"
                    )
        
        # Format the prompt
        prompt = self._format_prompt(text, question)
        
        # Attempt processing with retry logic
        last_error = None
        for attempt in range(self.retry_manager.max_retries + 1):
            try:
                # Make request to Ollama API
                response = self._make_ollama_request(prompt)
                return self._extract_answer(response)
                
            except requests.exceptions.Timeout as e:
                last_error = ProcessingTimeoutError(
                    self.processing_timeout, 
                    "Ollama text generation"
                )
                if not self.retry_manager.should_retry(last_error, attempt):
                    break
                    
            except requests.exceptions.ConnectionError as e:
                last_error = NetworkError("Ollama", e)
                if not self.retry_manager.should_retry(last_error, attempt):
                    break
                    
            except Exception as e:
                # For other errors, don't retry
                raise RuntimeError(f"Error processing question with Ollama: {str(e)}")
            
            # If we should retry, wait and try again
            if attempt < self.retry_manager.max_retries:
                delay = self.retry_manager.get_delay(attempt)
                self.retry_manager.log_retry(last_error, attempt, delay)
                time.sleep(delay)
        
        # If we get here, all retries failed
        if isinstance(last_error, ProcessingTimeoutError):
            raise last_error
        elif isinstance(last_error, NetworkError):
            raise last_error
        else:
            raise RuntimeError(f"Failed to process question after {self.retry_manager.max_retries + 1} attempts")

    def _format_prompt(self, text: str, question: str) -> str:
        """
        Format the prompt for the embeddinggemma model.
        
        Args:
            text (str): The context text
            question (str): The user's question
            
        Returns:
            str: Formatted prompt
        """
        # Create a structured prompt that works well with embeddinggemma
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{text}

Question: {question}

Answer:"""
        
        return prompt

    def _make_ollama_request(self, prompt: str) -> Dict[str, Any]:
        """
        Make a request to the Ollama API.
        
        Args:
            prompt (str): The formatted prompt
            
        Returns:
            Dict[str, Any]: The API response
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,  # We want the complete response, not streaming
            "options": {
                "temperature": 0.7,  # Balanced creativity vs accuracy
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            headers=headers,
            timeout=self.processing_timeout
        )
        
        if response.status_code != 200:
            error_msg = f"Ollama API returned status {response.status_code}"
            try:
                error_detail = response.json().get('error', 'Unknown error')
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {response.text}"
            raise RuntimeError(error_msg)
        
        return response.json()

    def _extract_answer(self, response: Dict[str, Any]) -> str:
        """
        Extract the answer from Ollama API response.
        
        Args:
            response (Dict[str, Any]): The API response
            
        Returns:
            str: The extracted answer
            
        Raises:
            RuntimeError: If response format is unexpected
        """
        try:
            answer = response.get('response', '').strip()
            
            if not answer:
                raise RuntimeError("Ollama returned an empty response")
            
            return answer
            
        except (KeyError, AttributeError) as e:
            raise RuntimeError(f"Unexpected response format from Ollama: {str(e)}")

    def get_backend_name(self) -> str:
        """
        Return the display name of this backend.
        
        Returns:
            str: Human-readable name for UI display
        """
        return "Local Ollama"

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dict[str, str]: Model information including name and base URL
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "backend_type": self.get_backend_type()
        }

    def get_timeout_info(self) -> Dict[str, int]:
        """
        Get timeout configuration information.
        
        Returns:
            Dict[str, int]: Timeout values in seconds
        """
        return {
            "connection_timeout": self.connection_timeout,
            "processing_timeout": self.processing_timeout
        }

    def _get_available_models(self) -> list:
        """
        Get list of available models from Ollama.
        
        Returns:
            list: List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.connection_timeout)
            if response.status_code == 200:
                models_data = response.json()
                return [model.get('name', '') for model in models_data.get('models', [])]
        except Exception:
            pass
        return []
    
    def get_installation_instructions(self) -> str:
        """
        Get installation instructions for users who don't have Ollama set up.
        
        Returns:
            str: Installation instructions
        """
        return (
            "To use Local Ollama processing:\n\n"
            "1. Install Ollama from https://ollama.ai\n"
            "2. Start Ollama service\n"
            "3. Install a text generation model (choose one):\n"
            "   - ollama pull gemma2:2b (recommended, fast and efficient)\n"
            "   - ollama pull phi3:mini (smaller, faster)\n"
            "   - ollama pull llama3.2:1b (very fast, good quality)\n"
            "4. Restart this application\n\n"
            "Note: embeddinggemma:latest is an embedding model and cannot be used for text generation."
        )
    
    def get_detailed_error_info(self) -> Dict[str, Any]:
        """
        Get detailed error information for troubleshooting.
        
        Returns:
            Dict[str, Any]: Detailed error information
        """
        info = {
            "service_url": self.base_url,
            "target_model": self.model_name,
            "preferred_models": self.preferred_models,
            "service_running": False,
            "available_models": [],
            "embedding_models_found": [],
            "text_gen_models_found": []
        }
        
        try:
            # Check service
            info["service_running"] = self._check_ollama_service()
            
            if info["service_running"]:
                # Get available models
                available_models = self._get_available_models()
                info["available_models"] = available_models
                
                # Categorize models
                for model in available_models:
                    if any(keyword in model.lower() for keyword in ['embedding', 'embed', 'nomic-embed']):
                        info["embedding_models_found"].append(model)
                    else:
                        info["text_gen_models_found"].append(model)
        
        except Exception as e:
            info["error"] = str(e)
        
        return info