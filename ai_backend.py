"""
AI Backend Abstraction Layer

This module defines the abstract base class for AI backends used in the Q&A application.
It provides a common interface that allows different AI processing backends (like Google Gemini
and local Ollama) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Optional


class AIBackend(ABC):
    """
    Abstract base class for AI backends that process questions against text context.
    
    This class defines the interface that all AI backends must implement to be compatible
    with the Q&A application. It ensures consistent behavior across different AI processing
    services while allowing for backend-specific implementations.
    """

    @abstractmethod
    def process_question(self, text: str, question: str) -> str:
        """
        Process a question against the given text context and return an answer.
        
        Args:
            text (str): The context text to analyze (document content, chunks, etc.)
            question (str): The user's question to be answered based on the context
            
        Returns:
            str: The AI-generated answer to the question based on the provided context
            
        Raises:
            Exception: Implementation-specific exceptions for processing failures
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available and ready to process questions.
        
        This method should verify that all necessary components are available:
        - API keys (for cloud services)
        - Local services running (for local backends)
        - Required models installed and accessible
        
        Returns:
            bool: True if the backend is ready to use, False otherwise
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """
        Return the display name of this backend for UI purposes.
        
        Returns:
            str: Human-readable name of the backend (e.g., "Google Gemini", "Local Ollama")
        """
        pass

    def get_backend_type(self) -> str:
        """
        Return the backend type identifier for configuration purposes.
        
        This is a concrete method that can be overridden if needed, but provides
        a default implementation based on the class name.
        
        Returns:
            str: Backend type identifier (e.g., "gemini", "ollama")
        """
        return self.__class__.__name__.lower().replace('backend', '')

    def validate_inputs(self, text: str, question: str) -> None:
        """
        Validate input parameters before processing.
        
        This is a concrete helper method that performs common validation
        that all backends should enforce.
        
        Args:
            text (str): The context text to validate
            question (str): The question to validate
            
        Raises:
            ValueError: If inputs are invalid (empty, None, etc.)
        """
        if not text or not text.strip():
            raise ValueError("Context text cannot be empty")
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        if len(text.strip()) < 10:
            raise ValueError("Context text is too short to provide meaningful answers")
        
        if len(question.strip()) < 3:
            raise ValueError("Question is too short to be meaningful")