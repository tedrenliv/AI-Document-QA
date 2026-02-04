"""
AI Backend Factory

This module provides a factory class for creating and managing AI backend instances
based on configuration settings. It implements the factory pattern to abstract
backend instantiation and provide a clean interface for backend selection.
"""

from typing import Optional, Dict, Type, Tuple, Any
import logging
from ai_backend import AIBackend
from gemini_backend import GeminiBackend
from ollama_backend import OllamaBackend
from backend_config import BackendConfig
from ai_backend_errors import FallbackManager, ErrorMessageGenerator


class AIBackendFactory:
    """
    Factory class for creating and managing AI backend instances.
    
    This factory handles the instantiation of appropriate backend classes based on
    configuration settings and provides methods for backend availability checking
    and runtime switching.
    """
    
    # Registry of available backend classes
    _backend_classes: Dict[str, Type[AIBackend]] = {
        "gemini": GeminiBackend,
        "ollama": OllamaBackend
    }
    
    def __init__(self, config: BackendConfig):
        """
        Initialize the factory with configuration.
        
        Args:
            config (BackendConfig): Configuration object containing backend preferences
        """
        self.config = config
        self._backend_instances: Dict[str, AIBackend] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize fallback manager
        self.fallback_manager = FallbackManager(self)
        
        self._initialize_backends()
    
    def _initialize_backends(self) -> None:
        """Initialize all available backend instances."""
        # Initialize Gemini backend
        self._backend_instances["gemini"] = GeminiBackend(self.config.api_key)
        
        # Initialize Ollama backend
        self._backend_instances["ollama"] = OllamaBackend()
    
    def get_current_backend(self) -> Optional[AIBackend]:
        """
        Get the currently selected backend instance.
        
        Returns:
            Optional[AIBackend]: The current backend instance, or None if not available
        """
        backend_type = self.config.backend_type
        
        if backend_type not in self._backend_instances:
            return None
        
        backend = self._backend_instances[backend_type]
        
        # Return the backend only if it's available
        if backend.is_available():
            return backend
        
        return None
    
    def get_current_backend_with_fallback(self) -> Tuple[Optional[AIBackend], Optional[str]]:
        """
        Get the current backend with automatic fallback if unavailable.
        
        Returns:
            Tuple[Optional[AIBackend], Optional[str]]: (backend_instance, fallback_message)
            If fallback_message is not None, it means a fallback was used.
        """
        return self.fallback_manager.get_fallback_backend(self.config.backend_type)
    
    def get_backend(self, backend_type: str) -> Optional[AIBackend]:
        """
        Get a specific backend instance by type.
        
        Args:
            backend_type (str): The backend type ("gemini" or "ollama")
            
        Returns:
            Optional[AIBackend]: The requested backend instance, or None if not found
        """
        return self._backend_instances.get(backend_type)
    
    def get_all_backends(self) -> Dict[str, AIBackend]:
        """
        Get all available backend instances.
        
        Returns:
            Dict[str, AIBackend]: Dictionary mapping backend types to instances
        """
        return self._backend_instances.copy()
    
    def switch_backend(self, backend_type: str) -> bool:
        """
        Switch to a different backend type.
        
        Args:
            backend_type (str): The backend type to switch to
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if backend_type not in self._backend_instances:
            return False
        
        # Update configuration
        self.config.backend_type = backend_type
        self.config.save_to_config()
        
        return True
    
    def update_api_key(self, api_key: str) -> bool:
        """
        Update the API key for Gemini backend.
        
        Args:
            api_key (str): New API key
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Update configuration
        self.config.api_key = api_key
        self.config.save_to_config()
        
        # Update Gemini backend instance
        gemini_backend = self._backend_instances.get("gemini")
        if isinstance(gemini_backend, GeminiBackend):
            return gemini_backend.update_api_key(api_key)
        
        return False
    
    def is_backend_available(self, backend_type: str) -> bool:
        """
        Check if a specific backend is available.
        
        Args:
            backend_type (str): The backend type to check
            
        Returns:
            bool: True if backend is available, False otherwise
        """
        backend = self._backend_instances.get(backend_type)
        return backend.is_available() if backend else False
    
    def get_available_backends(self) -> Dict[str, AIBackend]:
        """
        Get all currently available backend instances.
        
        Returns:
            Dict[str, AIBackend]: Dictionary of available backends
        """
        available = {}
        for backend_type, backend in self._backend_instances.items():
            if backend.is_available():
                available[backend_type] = backend
        return available
    
    def get_backend_status(self, backend_type: str) -> Dict[str, Any]:
        """
        Get detailed status information for a backend.

        Args:
            backend_type (str): The backend type to check

        Returns:
            Dict[str, Any]: Status information including availability and details
        """
        backend = self._backend_instances.get(backend_type)

        if not backend:
            return {
                "available": False,
                "status": "not_found",
                "message": f"Backend type '{backend_type}' not found"
            }

        is_available = backend.is_available()

        status_info = {
            "available": is_available,
            "backend_name": backend.get_backend_name(),
            "backend_type": backend_type
        }
        
        if not is_available:
            if backend_type == "gemini":
                if not self.config.api_key.strip():
                    status_info["status"] = "api_key_required"
                    status_info["message"] = "API Key Required"
                else:
                    status_info["status"] = "api_key_invalid"
                    status_info["message"] = "Invalid API Key"
            elif backend_type == "ollama":
                status_info["status"] = "service_unavailable"
                status_info["message"] = "Ollama Service Not Available"
        else:
            status_info["status"] = "ready"
            status_info["message"] = "Ready"
        
        return status_info
    
    def get_fallback_backend(self) -> Optional[AIBackend]:
        """
        Get a fallback backend if the current one is not available.
        
        Returns:
            Optional[AIBackend]: An available backend, or None if none are available
        """
        current_backend = self.get_current_backend()
        if current_backend:
            return current_backend
        
        # Try to find any available backend
        for backend in self._backend_instances.values():
            if backend.is_available():
                return backend
        
        return None
    
    @classmethod
    def register_backend(cls, backend_type: str, backend_class: Type[AIBackend]) -> None:
        """
        Register a new backend class with the factory.
        
        Args:
            backend_type (str): The backend type identifier
            backend_class (Type[AIBackend]): The backend class to register
        """
        cls._backend_classes[backend_type] = backend_class
    
    def get_backend_error_info(self, backend_type: str) -> Dict[str, Any]:
        """
        Get detailed error information for a backend.
        
        Args:
            backend_type (str): The backend type to check
            
        Returns:
            Dict[str, Any]: Detailed error information
        """
        backend = self._backend_instances.get(backend_type)
        
        if not backend:
            return {"error": f"Backend type '{backend_type}' not found"}
        
        if backend.is_available():
            return {"status": "available", "backend_name": backend.get_backend_name()}
        
        # Get backend-specific error information
        if backend_type == "ollama" and hasattr(backend, 'get_detailed_error_info'):
            return backend.get_detailed_error_info()
        elif backend_type == "gemini":
            return {
                "status": "unavailable",
                "reason": "authentication_failed" if not self.config.api_key.strip() else "api_key_invalid",
                "message": ErrorMessageGenerator.get_gemini_auth_error_message()
            }
        else:
            return {"status": "unavailable", "reason": "unknown"}
    
    def get_installation_help(self, backend_type: str) -> str:
        """
        Get installation help for a specific backend.
        
        Args:
            backend_type (str): The backend type
            
        Returns:
            str: Installation instructions
        """
        backend = self._backend_instances.get(backend_type)
        
        if backend_type == "ollama" and hasattr(backend, 'get_installation_instructions'):
            return backend.get_installation_instructions()
        elif backend_type == "gemini":
            return (
                "To use Google Gemini:\n\n"
                "1. Get an API key from https://makersuite.google.com/app/apikey\n"
                "2. Enter the API key in the application\n"
                "3. Ensure you have internet connectivity\n"
                "4. Check that your API key has sufficient quota"
            )
        else:
            return f"No installation instructions available for {backend_type}"
    
    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """
        Get list of supported backend types.
        
        Returns:
            list[str]: List of supported backend type identifiers
        """
        return list(cls._backend_classes.keys())