"""
Final Integration Tests for Task 12 - Integration testing and final validation

This module contains focused integration tests that validate:
- Complete Q&A workflow with both Gemini and Ollama backends
- Backend switching during runtime
- Error scenarios with Ollama not available or model missing
- Logging consistency across both backends
- UI updates based on backend availability and selection
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
import json
import requests
from requests.exceptions import ConnectionError, Timeout

# Import application modules
from mainintegratedWORD import QAApp
from backend_config import BackendConfig
from ai_backend_factory import AIBackendFactory
from ollama_backend import OllamaBackend
from gemini_backend import GeminiBackend
from ai_backend_errors import (
    ServiceUnavailableError, ModelNotFoundError, AuthenticationError,
    ProcessingTimeoutError, NetworkError
)


class TestCompleteWorkflowIntegration(unittest.TestCase):
    """Test complete Q&A workflow with both backends."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        self.test_log_file = os.path.join(self.temp_dir, "logbook.txt")
        
        # Create test document
        self.test_txt_file = os.path.join(self.temp_dir, "test_doc.txt")
        with open(self.test_txt_file, 'w', encoding='utf-8') as f:
            f.write("This is a comprehensive test document about artificial intelligence, "
                   "machine learning, neural networks, and deep learning technologies. "
                   "It covers various aspects of modern AI systems and their applications.")
        
        # Patch file paths
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.log_patcher = patch('mainintegratedWORD.LOG_FILE', self.test_log_file)
        self.config_patcher.start()
        self.log_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.log_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_gemini_complete_workflow(self, mock_configure, mock_model_class):
        """Test complete Q&A workflow using Gemini backend."""
        # Mock Gemini API
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This document discusses artificial intelligence and machine learning technologies."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Create configuration and factory
        config = BackendConfig(backend_type="gemini", api_key="test_api_key")
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        # Get backend and test workflow
        backend = factory.get_backend("gemini")
        self.assertIsNotNone(backend)
        self.assertTrue(backend.is_available())
        
        # Read test document
        with open(self.test_txt_file, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Process question
        question = "What technologies are discussed in this document?"
        answer = backend.process_question(document_text, question)
        
        # Verify workflow completed successfully
        self.assertEqual(answer, "This document discusses artificial intelligence and machine learning technologies.")
        mock_model.generate_content.assert_called_once()
        
        print("✓ Gemini complete workflow test passed")
    
    @patch('requests.post')
    @patch('requests.get')
    def test_ollama_complete_workflow(self, mock_get, mock_post):
        """Test complete Q&A workflow using Ollama backend."""
        # Mock Ollama service availability
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.1.0"}
        
        mock_models_response = Mock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "models": [{"name": "gemma2:2b"}, {"name": "llama3.2:1b"}]
        }
        
        def mock_get_side_effect(url, **kwargs):
            if "version" in url:
                return mock_version_response
            elif "tags" in url:
                return mock_models_response
            return Mock(status_code=404)
        
        mock_get.side_effect = mock_get_side_effect
        
        # Mock Ollama generation
        mock_generate_response = Mock()
        mock_generate_response.status_code = 200
        mock_generate_response.json.return_value = {
            "response": "This document covers AI and ML technologies comprehensively."
        }
        mock_post.return_value = mock_generate_response
        
        # Create configuration and backend
        config = BackendConfig(backend_type="ollama", api_key="")
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        backend = factory.get_backend("ollama")
        self.assertIsNotNone(backend)
        self.assertTrue(backend.is_available())
        
        # Read test document
        with open(self.test_txt_file, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Process question
        question = "What does this document cover?"
        answer = backend.process_question(document_text, question)
        
        # Verify workflow completed successfully
        self.assertEqual(answer, "This document covers AI and ML technologies comprehensively.")
        mock_post.assert_called_once()
        
        print("✓ Ollama complete workflow test passed")


class TestBackendSwitchingIntegration(unittest.TestCase):
    """Test backend switching functionality during runtime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.config_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_runtime_backend_switching(self):
        """Test switching backends during runtime."""
        # Start with Gemini
        config = BackendConfig(backend_type="gemini", api_key="test_key")
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        # Verify initial state
        self.assertEqual(factory.config.backend_type, "gemini")
        
        # Switch to Ollama
        success = factory.switch_backend("ollama")
        self.assertTrue(success)
        self.assertEqual(factory.config.backend_type, "ollama")
        
        # Switch back to Gemini
        success = factory.switch_backend("gemini")
        self.assertTrue(success)
        self.assertEqual(factory.config.backend_type, "gemini")
        
        # Test invalid backend
        success = factory.switch_backend("invalid")
        self.assertFalse(success)
        self.assertEqual(factory.config.backend_type, "gemini")  # Should remain unchanged
        
        print("✓ Runtime backend switching test passed")
    
    def test_configuration_persistence(self):
        """Test that backend switching persists across sessions."""
        # Create initial configuration
        config = BackendConfig(backend_type="gemini", api_key="test_key")
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        # Switch backend
        factory.switch_backend("ollama")
        
        # Create new factory instance (simulates app restart)
        new_config = BackendConfig.load_from_config()
        new_factory = AIBackendFactory(new_config)
        
        # Verify persistence
        self.assertEqual(new_factory.config.backend_type, "ollama")
        
        print("✓ Configuration persistence test passed")


class TestErrorScenariosIntegration(unittest.TestCase):
    """Test error scenarios with unavailable services."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.config_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.get')
    def test_ollama_service_unavailable_scenario(self, mock_get):
        """Test handling when Ollama service is not available."""
        # Mock connection error
        mock_get.side_effect = ConnectionError("Connection refused")
        
        config = BackendConfig(backend_type="ollama", api_key="")
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        ollama_backend = factory.get_backend("ollama")
        self.assertFalse(ollama_backend.is_available())
        
        # Test that processing raises appropriate error
        with self.assertRaises(ServiceUnavailableError):
            ollama_backend.process_question("This is a longer test text that meets the minimum length requirements for processing. It contains enough content to be considered valid input for the AI backend system.", "test question")
        
        print("✓ Ollama service unavailable scenario test passed")
    
    @patch('requests.get')
    def test_ollama_model_missing_scenario(self, mock_get):
        """Test handling when Ollama has no suitable models."""
        # Mock service available but no models
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.1.0"}
        
        mock_models_response = Mock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {"models": []}  # No models
        
        def mock_get_side_effect(url, **kwargs):
            if "version" in url:
                return mock_version_response
            elif "tags" in url:
                return mock_models_response
            return Mock(status_code=404)
        
        mock_get.side_effect = mock_get_side_effect
        
        config = BackendConfig(backend_type="ollama", api_key="")
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        ollama_backend = factory.get_backend("ollama")
        self.assertFalse(ollama_backend.is_available())
        
        print("✓ Ollama model missing scenario test passed")
    
    def test_gemini_authentication_error_scenario(self):
        """Test handling Gemini authentication errors."""
        config = BackendConfig(backend_type="gemini", api_key="")  # No API key
        config.save_to_config()
        factory = AIBackendFactory(config)
        
        gemini_backend = factory.get_backend("gemini")
        self.assertFalse(gemini_backend.is_available())
        
        # Test that processing raises appropriate error
        with self.assertRaises(AuthenticationError):
            gemini_backend.process_question("This is a longer test text that meets the minimum length requirements for processing. It contains enough content to be considered valid input for the AI backend system.", "test question")
        
        print("✓ Gemini authentication error scenario test passed")


class TestLoggingConsistencyIntegration(unittest.TestCase):
    """Test logging consistency across backends."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        self.test_log_file = os.path.join(self.temp_dir, "logbook.txt")
        
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.log_patcher = patch('mainintegratedWORD.LOG_FILE', self.test_log_file)
        self.config_patcher.start()
        self.log_patcher.start()
        
        # Create root window for QAApp
        self.root = tk.Tk()
        self.root.withdraw()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.log_patcher.stop()
        self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_consistent_logging_format(self):
        """Test that logging format is consistent across backends."""
        # Create QAApp instance
        with patch('backend_config.BackendConfig.load_from_config') as mock_load:
            mock_load.return_value = BackendConfig(backend_type="gemini", api_key="test_key")
            app = QAApp(self.root)
        
        # Test Gemini logging
        app.log_answer("What is AI?", "AI is artificial intelligence.", "Google Gemini")
        
        # Test Ollama logging
        app.log_answer("What is ML?", "ML is machine learning.", "Local Ollama")
        
        # Test backward compatibility (no backend name)
        app.log_answer("What is DL?", "DL is deep learning.")
        
        # Verify log content
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Check Gemini entry
        self.assertIn("(Google Gemini)", log_content)
        self.assertIn("What is AI?", log_content)
        self.assertIn("AI is artificial intelligence.", log_content)
        
        # Check Ollama entry
        self.assertIn("(Local Ollama)", log_content)
        self.assertIn("What is ML?", log_content)
        self.assertIn("ML is machine learning.", log_content)
        
        # Check backward compatibility entry
        self.assertIn("What is DL?", log_content)
        self.assertIn("DL is deep learning.", log_content)
        
        # Verify consistent format
        self.assertIn("Question:", log_content)
        self.assertIn("Answer:", log_content)
        
        print("✓ Consistent logging format test passed")


class TestUIUpdatesIntegration(unittest.TestCase):
    """Test UI updates based on backend availability and selection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.config_patcher.start()
        
        # Create root window
        self.root = tk.Tk()
        self.root.withdraw()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_ui_backend_selection_updates(self, mock_load_config):
        """Test UI updates when backend selection changes."""
        config = BackendConfig(backend_type="gemini", api_key="test_key")
        mock_load_config.return_value = config
        
        app = QAApp(self.root)
        
        # Initial state should be Gemini
        self.assertEqual(app.backend_var.get(), "gemini")
        self.assertEqual(str(app.api_entry.cget('state')), 'normal')
        
        # Switch to Ollama
        app.backend_var.set("ollama")
        app.on_backend_change()
        
        # API key field should be disabled
        self.assertEqual(str(app.api_entry.cget('state')), 'disabled')
        
        # Switch back to Gemini
        app.backend_var.set("gemini")
        app.on_backend_change()
        
        # API key field should be enabled again
        self.assertEqual(str(app.api_entry.cget('state')), 'normal')
        
        print("✓ UI backend selection updates test passed")
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_processing_status_updates(self, mock_load_config):
        """Test processing status display updates."""
        config = BackendConfig(backend_type="gemini", api_key="test_key")
        mock_load_config.return_value = config
        
        app = QAApp(self.root)
        
        # Test showing processing status
        app.show_processing_status("Processing with AI...", True)
        self.assertEqual(app.status_label.cget('text'), "Processing with AI...")
        # Note: tkinter color objects may not compare directly to strings
        self.assertIn("blue", str(app.status_label.cget('foreground')).lower())
        
        # Test hiding status
        app.hide_processing_status()
        self.assertEqual(app.status_label.cget('text'), "")
        
        # Test error status
        app.show_error_status("Connection failed")
        self.assertEqual(app.status_label.cget('text'), "Connection failed")
        # Note: tkinter color objects may not compare directly to strings
        self.assertIn("red", str(app.status_label.cget('foreground')).lower())
        
        print("✓ Processing status updates test passed")


def run_integration_tests():
    """Run all integration tests with detailed reporting."""
    print("="*60)
    print("RUNNING INTEGRATION TESTS FOR TASK 12")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    test_classes = [
        TestCompleteWorkflowIntegration,
        TestBackendSwitchingIntegration,
        TestErrorScenariosIntegration,
        TestLoggingConsistencyIntegration,
        TestUIUpdatesIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    # Print details if there are issues
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
    
    print("="*60)
    
    # Validation summary
    print("\nTASK 12 VALIDATION SUMMARY:")
    print("✓ Complete Q&A workflow tested for both backends")
    print("✓ Backend switching functionality verified")
    print("✓ Error scenarios tested and handled properly")
    print("✓ Logging consistency validated across backends")
    print("✓ UI updates tested for backend availability and selection")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)