"""
Unit Tests for AI Backend Implementations

This module contains comprehensive unit tests for all AI backend implementations,
including interface compliance, mock API communication, and availability checking.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import google.generativeai as genai

# Import the backend classes and related modules
from ai_backend import AIBackend
from ollama_backend import OllamaBackend
from gemini_backend import GeminiBackend
from ai_backend_factory import AIBackendFactory
from backend_config import BackendConfig
from ai_backend_errors import (
    ServiceUnavailableError, ModelNotFoundError, InvalidModelError,
    ProcessingTimeoutError, NetworkError, AuthenticationError
)


class TestAIBackendInterface(unittest.TestCase):
    """Test AIBackend abstract base class interface compliance."""
    
    def test_abstract_methods_defined(self):
        """Test that AIBackend defines all required abstract methods."""
        # Verify that AIBackend cannot be instantiated directly
        with self.assertRaises(TypeError):
            AIBackend()
    
    def test_concrete_methods_available(self):
        """Test that concrete helper methods are available."""
        # Create a mock implementation to test concrete methods
        class MockBackend(AIBackend):
            def process_question(self, text: str, question: str) -> str:
                return "mock answer"
            
            def is_available(self) -> bool:
                return True
            
            def get_backend_name(self) -> str:
                return "Mock Backend"
        
        backend = MockBackend()
        
        # Test get_backend_type method
        self.assertEqual(backend.get_backend_type(), "mock")
        
        # Test validate_inputs method with valid inputs
        backend.validate_inputs("This is a valid context text.", "What is this about?")
        
        # Test validate_inputs with invalid inputs
        with self.assertRaises(ValueError):
            backend.validate_inputs("", "question")
        
        with self.assertRaises(ValueError):
            backend.validate_inputs("context", "")
        
        with self.assertRaises(ValueError):
            backend.validate_inputs("short", "question")
        
        with self.assertRaises(ValueError):
            backend.validate_inputs("valid context text", "hi")


class TestOllamaBackend(unittest.TestCase):
    """Test OllamaBackend implementation with mocked API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = OllamaBackend()
        self.sample_text = "This is a sample document about artificial intelligence and machine learning."
        self.sample_question = "What is this document about?"
    
    def test_interface_compliance(self):
        """Test that OllamaBackend implements AIBackend interface correctly."""
        self.assertIsInstance(self.backend, AIBackend)
        self.assertTrue(hasattr(self.backend, 'process_question'))
        self.assertTrue(hasattr(self.backend, 'is_available'))
        self.assertTrue(hasattr(self.backend, 'get_backend_name'))
    
    def test_get_backend_name(self):
        """Test backend name method."""
        self.assertEqual(self.backend.get_backend_name(), "Local Ollama")
    
    def test_get_backend_type(self):
        """Test backend type method."""
        self.assertEqual(self.backend.get_backend_type(), "ollama")
    
    @patch('requests.get')
    def test_check_ollama_service_available(self, mock_get):
        """Test Ollama service availability check when service is running."""
        # Mock successful version check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response
        
        result = self.backend._check_ollama_service()
        self.assertTrue(result)
        mock_get.assert_called_with(f"{self.backend.base_url}/api/version", timeout=5)
    
    @patch('requests.get')
    def test_check_ollama_service_unavailable(self, mock_get):
        """Test Ollama service availability check when service is not running."""
        # Mock connection error
        mock_get.side_effect = ConnectionError("Connection refused")
        
        result = self.backend._check_ollama_service()
        self.assertFalse(result)
    
    @patch('requests.get')
    def test_check_ollama_service_timeout(self, mock_get):
        """Test Ollama service availability check with timeout."""
        # Mock timeout error
        mock_get.side_effect = Timeout("Request timed out")
        
        result = self.backend._check_ollama_service()
        self.assertFalse(result)
    
    @patch('requests.get')
    def test_check_model_availability_success(self, mock_get):
        """Test model availability check when suitable model is available."""
        # Mock successful models list with text generation model
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "gemma2:2b"},
                {"name": "llama3.2:1b"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.backend._check_model_availability()
        self.assertTrue(result)
    
    @patch('requests.get')
    def test_check_model_availability_only_embedding_models(self, mock_get):
        """Test model availability check when only embedding models are available."""
        # Mock models list with only embedding models
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "nomic-embed-text"},
                {"name": "embeddinggemma:latest"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.backend._check_model_availability()
        self.assertFalse(result)
    
    @patch('requests.get')
    def test_check_model_availability_no_models(self, mock_get):
        """Test model availability check when no models are installed."""
        # Mock empty models list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        result = self.backend._check_model_availability()
        self.assertFalse(result)
    
    @patch.object(OllamaBackend, '_check_model_availability')
    @patch.object(OllamaBackend, '_check_ollama_service')
    def test_is_available_success(self, mock_service_check, mock_model_check):
        """Test is_available method when both service and model are available."""
        mock_service_check.return_value = True
        mock_model_check.return_value = True
        
        result = self.backend.is_available()
        self.assertTrue(result)
    
    @patch.object(OllamaBackend, '_check_ollama_service')
    def test_is_available_service_unavailable(self, mock_service_check):
        """Test is_available method when service is not running."""
        mock_service_check.return_value = False
        
        result = self.backend.is_available()
        self.assertFalse(result)
    
    @patch.object(OllamaBackend, '_check_model_availability')
    @patch.object(OllamaBackend, '_check_ollama_service')
    def test_is_available_model_unavailable(self, mock_service_check, mock_model_check):
        """Test is_available method when service is running but no suitable model."""
        mock_service_check.return_value = True
        mock_model_check.return_value = False
        
        result = self.backend.is_available()
        self.assertFalse(result)
    
    @patch('requests.post')
    @patch.object(OllamaBackend, 'is_available')
    def test_process_question_success(self, mock_is_available, mock_post):
        """Test successful question processing."""
        # Mock availability check
        mock_is_available.return_value = True
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This document is about artificial intelligence and machine learning."
        }
        mock_post.return_value = mock_response
        
        result = self.backend.process_question(self.sample_text, self.sample_question)
        
        self.assertEqual(result, "This document is about artificial intelligence and machine learning.")
        
        # Verify API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['model'], self.backend.model_name)
        self.assertIn(self.sample_text, call_args[1]['json']['prompt'])
        self.assertIn(self.sample_question, call_args[1]['json']['prompt'])
    
    @patch.object(OllamaBackend, '_check_model_availability')
    @patch.object(OllamaBackend, '_check_ollama_service')
    def test_process_question_service_unavailable(self, mock_service_check, mock_model_check):
        """Test process_question when service is unavailable."""
        mock_service_check.return_value = False
        mock_model_check.return_value = False
        
        with self.assertRaises(ServiceUnavailableError):
            self.backend.process_question(self.sample_text, self.sample_question)
    
    @patch('requests.post')
    @patch.object(OllamaBackend, 'is_available')
    def test_process_question_timeout(self, mock_is_available, mock_post):
        """Test process_question with timeout error."""
        mock_is_available.return_value = True
        mock_post.side_effect = Timeout("Request timed out")
        
        with self.assertRaises(ProcessingTimeoutError):
            self.backend.process_question(self.sample_text, self.sample_question)
    
    @patch('requests.post')
    @patch.object(OllamaBackend, 'is_available')
    def test_process_question_connection_error(self, mock_is_available, mock_post):
        """Test process_question with connection error."""
        mock_is_available.return_value = True
        mock_post.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(NetworkError):
            self.backend.process_question(self.sample_text, self.sample_question)
    
    def test_process_question_invalid_inputs(self):
        """Test process_question with invalid inputs."""
        with self.assertRaises(ValueError):
            self.backend.process_question("", self.sample_question)
        
        with self.assertRaises(ValueError):
            self.backend.process_question(self.sample_text, "")
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        prompt = self.backend._format_prompt(self.sample_text, self.sample_question)
        
        self.assertIn(self.sample_text, prompt)
        self.assertIn(self.sample_question, prompt)
        self.assertIn("Context:", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("Answer:", prompt)
    
    @patch('requests.post')
    def test_make_ollama_request_success(self, mock_post):
        """Test successful Ollama API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test response"}
        mock_post.return_value = mock_response
        
        result = self.backend._make_ollama_request("test prompt")
        
        self.assertEqual(result, {"response": "test response"})
    
    @patch('requests.post')
    def test_make_ollama_request_error_status(self, mock_post):
        """Test Ollama API request with error status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_response.text = "Server error"
        mock_post.return_value = mock_response
        
        with self.assertRaises(RuntimeError):
            self.backend._make_ollama_request("test prompt")
    
    def test_extract_answer_success(self):
        """Test successful answer extraction."""
        response = {"response": "This is the answer"}
        result = self.backend._extract_answer(response)
        self.assertEqual(result, "This is the answer")
    
    def test_extract_answer_empty_response(self):
        """Test answer extraction with empty response."""
        response = {"response": ""}
        
        with self.assertRaises(RuntimeError):
            self.backend._extract_answer(response)
    
    def test_extract_answer_missing_key(self):
        """Test answer extraction with missing response key."""
        response = {"other_key": "value"}
        
        with self.assertRaises(RuntimeError):
            self.backend._extract_answer(response)


class TestGeminiBackend(unittest.TestCase):
    """Test GeminiBackend implementation with mocked API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key_123"
        self.backend = GeminiBackend(self.api_key)
        self.sample_text = "This is a sample document about artificial intelligence and machine learning."
        self.sample_question = "What is this document about?"
    
    def test_interface_compliance(self):
        """Test that GeminiBackend implements AIBackend interface correctly."""
        self.assertIsInstance(self.backend, AIBackend)
        self.assertTrue(hasattr(self.backend, 'process_question'))
        self.assertTrue(hasattr(self.backend, 'is_available'))
        self.assertTrue(hasattr(self.backend, 'get_backend_name'))
    
    def test_get_backend_name(self):
        """Test backend name method."""
        self.assertEqual(self.backend.get_backend_name(), "Google Gemini")
    
    def test_get_backend_type(self):
        """Test backend type method."""
        self.assertEqual(self.backend.get_backend_type(), "gemini")
    
    def test_initialization_with_api_key(self):
        """Test backend initialization with valid API key."""
        backend = GeminiBackend("valid_key")
        self.assertEqual(backend.api_key, "valid_key")
    
    def test_initialization_without_api_key(self):
        """Test backend initialization without API key."""
        backend = GeminiBackend()
        self.assertIsNone(backend.api_key)
        self.assertFalse(backend.is_available())
    
    def test_is_available_with_api_key(self):
        """Test is_available method with API key provided."""
        # With valid setup, should return True
        self.assertTrue(self.backend.is_available())
    
    def test_is_available_without_api_key(self):
        """Test is_available method without API key."""
        backend = GeminiBackend()
        self.assertFalse(backend.is_available())
    
    def test_is_available_with_empty_api_key(self):
        """Test is_available method with empty API key."""
        backend = GeminiBackend("")
        self.assertFalse(backend.is_available())
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_process_question_success(self, mock_configure, mock_model_class):
        """Test successful question processing."""
        # Mock the model and its response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This document is about artificial intelligence and machine learning."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Create backend and set the mocked model
        backend = GeminiBackend(self.api_key)
        backend._model = mock_model
        
        result = backend.process_question(self.sample_text, self.sample_question)
        
        self.assertEqual(result, "This document is about artificial intelligence and machine learning.")
        mock_model.generate_content.assert_called_once()
    
    @patch('google.generativeai.GenerativeModel')
    def test_process_question_authentication_error(self, mock_model_class):
        """Test process_question with authentication error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API key invalid")
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(self.api_key)
        backend._model = mock_model
        
        with self.assertRaises(AuthenticationError):
            backend.process_question(self.sample_text, self.sample_question)
    
    def test_process_question_no_api_key(self):
        """Test process_question without API key."""
        backend = GeminiBackend()
        
        with self.assertRaises(AuthenticationError):
            backend.process_question(self.sample_text, self.sample_question)
    
    def test_process_question_invalid_inputs(self):
        """Test process_question with invalid inputs."""
        with self.assertRaises(ValueError):
            self.backend.process_question("", self.sample_question)
        
        with self.assertRaises(ValueError):
            self.backend.process_question(self.sample_text, "")
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_update_api_key_success(self, mock_configure, mock_model_class):
        """Test successful API key update."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend()
        result = backend.update_api_key("new_api_key")
        
        self.assertTrue(result)
        self.assertEqual(backend.api_key, "new_api_key")
        mock_configure.assert_called_with(api_key="new_api_key")
    
    @patch('google.generativeai.configure')
    def test_update_api_key_failure(self, mock_configure):
        """Test API key update failure."""
        mock_configure.side_effect = Exception("Configuration failed")
        
        backend = GeminiBackend()
        result = backend.update_api_key("invalid_key")
        
        self.assertFalse(result)
        self.assertIsNone(backend._model)
    
    def test_get_model_name(self):
        """Test get_model_name method."""
        expected_model = "gemini-2.0-flash-exp"
        self.assertEqual(self.backend.get_model_name(), expected_model)


class TestAIBackendFactory(unittest.TestCase):
    """Test AIBackendFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BackendConfig(backend_type="gemini", api_key="test_key")
        self.factory = AIBackendFactory(self.config)
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        self.assertIsInstance(self.factory, AIBackendFactory)
        self.assertEqual(self.factory.config, self.config)
        self.assertIn("gemini", self.factory._backend_instances)
        self.assertIn("ollama", self.factory._backend_instances)
    
    def test_get_backend_by_type(self):
        """Test getting backend by type."""
        gemini_backend = self.factory.get_backend("gemini")
        ollama_backend = self.factory.get_backend("ollama")
        
        self.assertIsInstance(gemini_backend, GeminiBackend)
        self.assertIsInstance(ollama_backend, OllamaBackend)
    
    def test_get_backend_invalid_type(self):
        """Test getting backend with invalid type."""
        result = self.factory.get_backend("invalid")
        self.assertIsNone(result)
    
    @patch.object(GeminiBackend, 'is_available')
    def test_get_current_backend_available(self, mock_is_available):
        """Test getting current backend when available."""
        mock_is_available.return_value = True
        
        backend = self.factory.get_current_backend()
        self.assertIsInstance(backend, GeminiBackend)
    
    @patch.object(GeminiBackend, 'is_available')
    def test_get_current_backend_unavailable(self, mock_is_available):
        """Test getting current backend when unavailable."""
        mock_is_available.return_value = False
        
        backend = self.factory.get_current_backend()
        self.assertIsNone(backend)
    
    def test_switch_backend(self):
        """Test switching backend type."""
        result = self.factory.switch_backend("ollama")
        self.assertTrue(result)
        self.assertEqual(self.factory.config.backend_type, "ollama")
    
    def test_switch_backend_invalid(self):
        """Test switching to invalid backend type."""
        result = self.factory.switch_backend("invalid")
        self.assertFalse(result)
    
    @patch.object(GeminiBackend, 'update_api_key')
    def test_update_api_key(self, mock_update):
        """Test updating API key."""
        mock_update.return_value = True
        
        result = self.factory.update_api_key("new_key")
        self.assertTrue(result)
        self.assertEqual(self.factory.config.api_key, "new_key")
    
    @patch.object(GeminiBackend, 'is_available')
    @patch.object(OllamaBackend, 'is_available')
    def test_get_available_backends(self, mock_ollama_available, mock_gemini_available):
        """Test getting available backends."""
        mock_gemini_available.return_value = True
        mock_ollama_available.return_value = False
        
        available = self.factory.get_available_backends()
        
        self.assertIn("gemini", available)
        self.assertNotIn("ollama", available)
    
    def test_get_supported_backends(self):
        """Test getting supported backend types."""
        supported = AIBackendFactory.get_supported_backends()
        self.assertIn("gemini", supported)
        self.assertIn("ollama", supported)
    
    @patch.object(GeminiBackend, 'is_available')
    def test_get_backend_status_available(self, mock_is_available):
        """Test getting backend status when available."""
        mock_is_available.return_value = True
        
        status = self.factory.get_backend_status("gemini")
        
        self.assertTrue(status["available"])
        self.assertEqual(status["status"], "ready")
    
    @patch.object(GeminiBackend, 'is_available')
    def test_get_backend_status_unavailable(self, mock_is_available):
        """Test getting backend status when unavailable."""
        mock_is_available.return_value = False
        
        status = self.factory.get_backend_status("gemini")
        
        self.assertFalse(status["available"])
        self.assertIn("status", status)


class TestBackendAvailabilityChecking(unittest.TestCase):
    """Test backend availability checking methods across all implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ollama_backend = OllamaBackend()
        self.gemini_backend = GeminiBackend("test_key")
    
    @patch('requests.get')
    def test_ollama_availability_caching(self, mock_get):
        """Test that Ollama availability checks are cached properly."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "gemma2:2b"}]
        }
        mock_get.return_value = mock_response
        
        # First call should make HTTP requests
        result1 = self.ollama_backend.is_available()
        call_count_after_first = mock_get.call_count
        
        # Second call within cache duration should use cache
        result2 = self.ollama_backend.is_available()
        call_count_after_second = mock_get.call_count
        
        self.assertEqual(result1, result2)
        self.assertEqual(call_count_after_first, call_count_after_second)
    
    def test_gemini_availability_with_different_key_states(self):
        """Test Gemini availability with different API key states."""
        # Test with no key
        backend_no_key = GeminiBackend()
        self.assertFalse(backend_no_key.is_available())
        
        # Test with empty key
        backend_empty_key = GeminiBackend("")
        self.assertFalse(backend_empty_key.is_available())
        
        # Test with whitespace key
        backend_whitespace_key = GeminiBackend("   ")
        self.assertFalse(backend_whitespace_key.is_available())
        
        # Test with valid key format
        backend_valid_key = GeminiBackend("valid_key_123")
        # Note: This will return True because we can't test actual API validation in unit tests
        self.assertTrue(backend_valid_key.is_available())
    
    @patch('requests.get')
    def test_ollama_model_preference_selection(self, mock_get):
        """Test that Ollama selects preferred models correctly."""
        # Mock response with multiple models including preferred ones
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "some-other-model"},
                {"name": "phi3:mini"},  # This is in preferred list
                {"name": "gemma2:2b"},  # This is first in preferred list
                {"name": "another-model"}
            ]
        }
        mock_get.return_value = mock_response
        
        # Check model availability - should select gemma2:2b as it's first in preferred list
        result = self.ollama_backend._check_model_availability()
        self.assertTrue(result)
        self.assertEqual(self.ollama_backend.model_name, "gemma2:2b")


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAIBackendInterface,
        TestOllamaBackend,
        TestGeminiBackend,
        TestAIBackendFactory,
        TestBackendAvailabilityChecking
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")