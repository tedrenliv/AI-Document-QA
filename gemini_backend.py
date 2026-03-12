"""
Google Gemini AI Backend Implementation

This module implements the AIBackend interface for Google's Gemini API,
providing cloud-based AI processing capabilities for the Q&A application.
"""

import google.generativeai as genai
import time
from typing import Optional
from ai_backend import AIBackend
from ai_backend_errors import (
    AuthenticationError, ProcessingTimeoutError, NetworkError,
    ErrorMessageGenerator, RetryManager, RateLimitError
)


class GeminiBackend(AIBackend):
    """
    Google Gemini API backend implementation.
    
    This backend uses Google's Gemini API for processing questions against text context.
    It requires a valid Google API key and internet connectivity to function.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini backend with an API key.
        
        Args:
            api_key (Optional[str]): Google API key for Gemini access.
                                   If None, the backend will not be available.
        """
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash"
        self._model = None
        
        # Initialize retry manager for transient failures
        self.retry_manager = RetryManager(max_retries=2, base_delay=2.0)
        
        # Timeout configuration
        self.request_timeout = 60  # 60 seconds for API requests
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                # If configuration fails, model remains None
                self._model = None

    def process_question(self, text: str, question: str) -> str:
        """
        Process a question against the given text context using Gemini API with retry logic.
        
        Args:
            text (str): The context text to analyze
            question (str): The user's question
            
        Returns:
            str: The AI-generated answer from Gemini
            
        Raises:
            AuthenticationError: If API key is invalid or authentication fails
            ProcessingTimeoutError: If request times out
            NetworkError: If network communication fails
            ValueError: If inputs are invalid
        """
        # Validate inputs using the base class method
        self.validate_inputs(text, question)
        
        if not self.is_available():
            raise AuthenticationError("Google Gemini", {
                "message": ErrorMessageGenerator.get_gemini_auth_error_message()
            })
        
        # Gemini 2.0 Flash supports ~1M tokens; cap at 500,000 chars as a safety margin
        context_text = text[:500_000] if len(text) > 500_000 else text
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
        
        # Attempt processing with retry logic
        last_error = None
        for attempt in range(self.retry_manager.max_retries + 1):
            try:
                response = self._model.generate_content(prompt)
                
                if not response.text:
                    raise RuntimeError("Gemini API returned empty response")
                    
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for authentication errors (don't retry these)
                if any(keyword in error_str for keyword in ['api key', 'authentication', 'unauthorized', 'forbidden']):
                    raise AuthenticationError("Google Gemini", {"original_error": str(e)})
                
                # Check for timeout errors
                if any(keyword in error_str for keyword in ['timeout', 'deadline']):
                    last_error = ProcessingTimeoutError(self.request_timeout, "Gemini API request")
                    if not self.retry_manager.should_retry(last_error, attempt):
                        break
                
                # Check for network errors
                elif any(keyword in error_str for keyword in ['connection', 'network', 'dns']):
                    last_error = NetworkError("Google Gemini", e)
                    if not self.retry_manager.should_retry(last_error, attempt):
                        break
                
                # Check for rate limiting (should retry with longer delay)
                elif any(keyword in error_str for keyword in ['rate limit', 'quota', 'too many requests', '429']):
                    retry_after = self._parse_retry_after(e)
                    last_error = RateLimitError("Google Gemini", retry_after=retry_after, original_error=e)
                    if not self.retry_manager.should_retry(last_error, attempt):
                        break
                
                else:
                    # For other errors, don't retry
                    raise RuntimeError(f"Failed to process question with Gemini: {str(e)}")
            
            # If we should retry, wait and try again
            if attempt < self.retry_manager.max_retries:
                if isinstance(last_error, RateLimitError) and last_error.retry_after:
                    delay = float(last_error.retry_after)
                else:
                    delay = self.retry_manager.get_delay(attempt)
                self.retry_manager.log_retry(last_error, attempt, delay)
                time.sleep(delay)

        # If we get here, all retries failed
        if isinstance(last_error, (ProcessingTimeoutError, NetworkError, RateLimitError)):
            raise last_error
        else:
            raise RuntimeError(f"Failed to process question after {self.retry_manager.max_retries + 1} attempts")

    def is_available(self) -> bool:
        """
        Check if Gemini backend is available and ready to use.
        
        Returns:
            bool: True if API key is provided and model is configured, False otherwise
        """
        if not self.api_key or not self.api_key.strip():
            return False
            
        if self._model is None:
            # Try to reconfigure if model is not set
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
                return True
            except Exception:
                return False
                
        return True

    def get_backend_name(self) -> str:
        """
        Return the display name of this backend.
        
        Returns:
            str: Human-readable name for UI display
        """
        return "Google Gemini"

    def update_api_key(self, api_key: str) -> bool:
        """
        Update the API key and reconfigure the backend.
        
        Args:
            api_key (str): New Google API key
            
        Returns:
            bool: True if the key was successfully updated and configured, False otherwise
        """
        self.api_key = api_key
        
        if not api_key or not api_key.strip():
            self._model = None
            return False
            
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            return True
        except Exception:
            self._model = None
            return False

    def _parse_retry_after(self, error: Exception) -> Optional[int]:
        """Extract Retry-After seconds from a rate limit error if present."""
        error_str = str(error)
        import re
        match = re.search(r'retry.?after[:\s]+(\d+)', error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def get_model_name(self) -> str:
        """
        Get the name of the Gemini model being used.
        
        Returns:
            str: The model name (e.g., "gemini-2.5-flash")
        """
        return self.model_name