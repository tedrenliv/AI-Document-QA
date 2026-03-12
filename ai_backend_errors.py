"""
AI Backend Error Handling

This module defines custom exception classes and error handling utilities
for AI backend operations. It provides specific error types for different
failure scenarios and user-friendly error messages.
"""

from typing import Optional, Dict, Any
import logging


class AIBackendError(Exception):
    """Base exception class for AI backend errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        Initialize AI backend error.
        
        Args:
            message (str): Human-readable error message
            error_code (str, optional): Machine-readable error code
            details (Dict[str, Any], optional): Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}


class ServiceUnavailableError(AIBackendError):
    """Raised when an AI service is not available or not running."""
    
    def __init__(self, service_name: str, details: Dict[str, Any] = None):
        message = f"{service_name} service is not available"
        super().__init__(message, "SERVICE_UNAVAILABLE", details)
        self.service_name = service_name


class ModelNotFoundError(AIBackendError):
    """Raised when a required AI model is not found or not installed."""
    
    def __init__(self, model_name: str, service_name: str = None, installation_cmd: str = None):
        if service_name:
            message = f"Model '{model_name}' not found in {service_name}"
        else:
            message = f"Model '{model_name}' not found"
        
        details = {}
        if installation_cmd:
            details["installation_command"] = installation_cmd
            
        super().__init__(message, "MODEL_NOT_FOUND", details)
        self.model_name = model_name
        self.service_name = service_name
        self.installation_cmd = installation_cmd


class InvalidModelError(AIBackendError):
    """Raised when a model exists but is not suitable for the requested operation."""
    
    def __init__(self, model_name: str, reason: str, suggested_models: list = None):
        message = f"Model '{model_name}' is not suitable: {reason}"
        details = {}
        if suggested_models:
            details["suggested_models"] = suggested_models
            
        super().__init__(message, "INVALID_MODEL", details)
        self.model_name = model_name
        self.reason = reason
        self.suggested_models = suggested_models or []


class AuthenticationError(AIBackendError):
    """Raised when authentication fails (e.g., invalid API key)."""
    
    def __init__(self, service_name: str, details: Dict[str, Any] = None):
        message = f"Authentication failed for {service_name}"
        super().__init__(message, "AUTHENTICATION_FAILED", details)
        self.service_name = service_name


class ProcessingTimeoutError(AIBackendError):
    """Raised when processing takes too long and times out."""
    
    def __init__(self, timeout_seconds: int, operation: str = "processing"):
        message = f"Operation timed out after {timeout_seconds} seconds during {operation}"
        details = {"timeout_seconds": timeout_seconds, "operation": operation}
        super().__init__(message, "PROCESSING_TIMEOUT", details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class NetworkError(AIBackendError):
    """Raised when network communication fails."""

    def __init__(self, service_name: str, original_error: Exception = None):
        message = f"Network error communicating with {service_name}"
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__

        super().__init__(message, "NETWORK_ERROR", details)
        self.service_name = service_name
        self.original_error = original_error


class RateLimitError(AIBackendError):
    """Raised when the API rate limit is exceeded."""

    def __init__(self, service_name: str, retry_after: int = None, original_error: Exception = None):
        message = f"Rate limit exceeded for {service_name}"
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)
        self.service_name = service_name
        self.retry_after = retry_after
        self.original_error = original_error


class ErrorMessageGenerator:
    """Generates user-friendly error messages for different error scenarios."""
    
    @staticmethod
    def get_ollama_service_unavailable_message() -> str:
        """Get user-friendly message for Ollama service unavailable."""
        return (
            "Ollama service is not running or not accessible.\n\n"
            "To fix this:\n"
            "1. Make sure Ollama is installed (visit https://ollama.ai)\n"
            "2. Start the Ollama service\n"
            "3. Verify it's running on localhost:11434\n"
            "4. Check that no firewall is blocking the connection\n\n"
            "You can also switch to Google Gemini if you have an API key."
        )
    
    @staticmethod
    def get_ollama_model_missing_message(model_name: str) -> str:
        """Get user-friendly message for missing Ollama model."""
        # Provide specific installation commands for different models
        if "embeddinggemma" in model_name.lower():
            return (
                f"The model '{model_name}' is an embedding model and cannot be used for text generation.\n\n"
                "Please install a text generation model instead:\n"
                "• ollama pull gemma2:2b (recommended, fast and efficient)\n"
                "• ollama pull phi3:mini (smaller, faster)\n"
                "• ollama pull llama3.2:1b (very fast, good quality)\n\n"
                "After installation, restart this application."
            )
        else:
            return (
                f"The model '{model_name}' is not installed in Ollama.\n\n"
                "To install it, run:\n"
                f"ollama pull {model_name}\n\n"
                "Or install a recommended alternative:\n"
                "• ollama pull gemma2:2b (recommended)\n"
                "• ollama pull phi3:mini (smaller, faster)\n"
                "• ollama pull llama3.2:1b (very fast)\n\n"
                "After installation, restart this application."
            )
    
    @staticmethod
    def get_gemini_auth_error_message() -> str:
        """Get user-friendly message for Gemini authentication errors."""
        return (
            "Google Gemini API authentication failed.\n\n"
            "Please check:\n"
            "1. Your API key is correct and complete\n"
            "2. The API key has not expired\n"
            "3. You have sufficient quota/credits\n"
            "4. The Gemini API is enabled for your project\n\n"
            "You can get an API key at: https://makersuite.google.com/app/apikey"
        )
    
    @staticmethod
    def get_network_error_message(service_name: str) -> str:
        """Get user-friendly message for network errors."""
        if "ollama" in service_name.lower():
            return (
                "Cannot connect to Ollama service.\n\n"
                "Please check:\n"
                "• Ollama is running and accessible\n"
                "• No firewall is blocking localhost:11434\n"
                "• The Ollama service hasn't crashed\n\n"
                "Try restarting Ollama or switch to Google Gemini."
            )
        else:
            return (
                f"Network error connecting to {service_name}.\n\n"
                "Please check:\n"
                "• Your internet connection is working\n"
                "• The service is not experiencing outages\n"
                "• No firewall is blocking the connection\n\n"
                "Try again in a moment or switch to a different backend."
            )

    @staticmethod
    def get_rate_limit_error_message(service_name: str, retry_after: int = None) -> str:
        """Get user-friendly message for rate limit errors."""
        wait_hint = f"Wait at least {retry_after} seconds before trying again." if retry_after else "Wait 60 seconds before trying again."
        return (
            f"{service_name} rate limit reached.\n\n"
            f"{wait_hint}\n\n"
            "To avoid rate limits:\n"
            "• Space out your requests\n"
            "• Use a smaller document\n"
            "• Switch to Local Ollama for unlimited offline processing\n\n"
            "Free tier limits: ~15 requests/minute, 1,500 requests/day."
        )
    
    @staticmethod
    def get_timeout_error_message(backend_name: str, timeout_seconds: int) -> str:
        """Get user-friendly message for timeout errors."""
        if "ollama" in backend_name.lower():
            return (
                f"Local processing timed out after {timeout_seconds} seconds.\n\n"
                "This can happen when:\n"
                "• Processing very large documents\n"
                "• The model is loading for the first time\n"
                "• Your system is under heavy load\n\n"
                "Try:\n"
                "• Using a smaller document or text chunk\n"
                "• Waiting a moment and trying again\n"
                "• Switching to Google Gemini for faster processing"
            )
        else:
            return (
                f"Request timed out after {timeout_seconds} seconds.\n\n"
                "This might be due to:\n"
                "• Slow internet connection\n"
                "• Service overload\n"
                "• Very large document processing\n\n"
                "Please try again with a smaller document or check your connection."
            )


class RetryManager:
    """Manages retry logic for transient failures."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry manager.
        
        Args:
            max_retries (int): Maximum number of retry attempts
            base_delay (float): Base delay between retries in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if an error should be retried.
        
        Args:
            error (Exception): The error that occurred
            attempt (int): Current attempt number (0-based)
            
        Returns:
            bool: True if should retry, False otherwise
        """
        if attempt >= self.max_retries:
            return False
        
        # Retry on transient errors
        if isinstance(error, (NetworkError, ProcessingTimeoutError, RateLimitError)):
            return True

        # Don't retry on permanent errors
        if isinstance(error, (AuthenticationError, ModelNotFoundError, InvalidModelError)):
            return False
        
        # For generic connection errors, check the error message
        if "connection" in str(error).lower() or "timeout" in str(error).lower():
            return True
        
        return False
    
    def get_delay(self, attempt: int) -> float:
        """
        Get delay before next retry attempt (exponential backoff).
        
        Args:
            attempt (int): Current attempt number (0-based)
            
        Returns:
            float: Delay in seconds
        """
        return self.base_delay * (2 ** attempt)
    
    def log_retry(self, error: Exception, attempt: int, delay: float) -> None:
        """
        Log retry attempt.
        
        Args:
            error (Exception): The error that caused the retry
            attempt (int): Current attempt number
            delay (float): Delay before next attempt
        """
        self.logger.warning(
            f"Retry attempt {attempt + 1}/{self.max_retries} after error: {error}. "
            f"Waiting {delay:.1f}s before next attempt."
        )


class FallbackManager:
    """Manages fallback logic when preferred backends are unavailable."""
    
    def __init__(self, backend_factory):
        """
        Initialize fallback manager.
        
        Args:
            backend_factory: The AI backend factory instance
        """
        self.backend_factory = backend_factory
        self.logger = logging.getLogger(__name__)
    
    def get_fallback_backend(self, preferred_backend_type: str):
        """
        Get a fallback backend when the preferred one is unavailable.
        
        Args:
            preferred_backend_type (str): The preferred backend type
            
        Returns:
            tuple: (backend_instance, fallback_message) or (None, error_message)
        """
        # First try to get the preferred backend
        preferred_backend = self.backend_factory.get_backend(preferred_backend_type)
        if preferred_backend and preferred_backend.is_available():
            return preferred_backend, None
        
        # Log the fallback attempt
        self.logger.info(f"Preferred backend '{preferred_backend_type}' unavailable, looking for fallback")
        
        # Try to find any available backend
        available_backends = self.backend_factory.get_available_backends()
        
        if not available_backends:
            return None, self._get_no_backends_message()
        
        # Prefer Gemini over Ollama for fallback (usually more reliable)
        fallback_priority = ["gemini", "ollama"]
        
        for backend_type in fallback_priority:
            if backend_type != preferred_backend_type and backend_type in available_backends:
                fallback_backend = available_backends[backend_type]
                fallback_message = self._get_fallback_message(preferred_backend_type, backend_type)
                return fallback_backend, fallback_message
        
        # If we get here, only the same type is available but it's not working
        return None, self._get_backend_specific_error_message(preferred_backend_type)
    
    def _get_fallback_message(self, preferred_type: str, fallback_type: str) -> str:
        """Generate fallback notification message."""
        preferred_name = "Google Gemini" if preferred_type == "gemini" else "Local Ollama"
        fallback_name = "Google Gemini" if fallback_type == "gemini" else "Local Ollama"
        
        return (
            f"{preferred_name} is not available. "
            f"Automatically switched to {fallback_name} for this request."
        )
    
    def _get_no_backends_message(self) -> str:
        """Generate message when no backends are available."""
        return (
            "No AI backends are currently available.\n\n"
            "Please ensure either:\n"
            "• Google Gemini: Valid API key is provided\n"
            "• Local Ollama: Service is running with a text generation model\n\n"
            "Check the status indicators and try again."
        )
    
    def _get_backend_specific_error_message(self, backend_type: str) -> str:
        """Generate backend-specific error message."""
        if backend_type == "gemini":
            return ErrorMessageGenerator.get_gemini_auth_error_message()
        elif backend_type == "ollama":
            return ErrorMessageGenerator.get_ollama_service_unavailable_message()
        else:
            return f"Backend '{backend_type}' is not available. Please check its configuration."