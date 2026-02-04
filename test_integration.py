"""
Integration Tests for AI Backend System

This module contains comprehensive integration tests that validate the complete
Q&A workflow, backend switching, error scenarios, logging consistency, and UI updates.
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from tkinter import ttk
import time
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


class TestCompleteQAWorkflow(unittest.TestCase):
    """Test complete Q&A workflow with both Gemini and Ollama backends."""
    
    def setUp(self):
        """Set up test fixtures with temporary files and mock backends."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        self.test_log_file = os.path.join(self.temp_dir, "logbook.txt")
        
        # Create test document files
        self.test_txt_file = os.path.join(self.temp_dir, "test_document.txt")
        with open(self.test_txt_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document about artificial intelligence and machine learning. "
                   "It contains information about neural networks, deep learning algorithms, "
                   "and natural language processing techniques used in modern AI systems.")
        
        # Patch file paths to use temporary directory
        self.config_patcher = patch('mainintegratedWORD.CONFIG_FILE', self.test_config_file)
        self.log_patcher = patch('mainintegratedWORD.LOG_FILE', self.test_log_file)
        self.config_patcher.start()
        self.log_patcher.start()
        
        # Create test configuration
        self.test_config = BackendConfig(backend_type="gemini", api_key="test_api_key_123")
        self.test_config.save_to_config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.log_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)  
  
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_gemini_workflow(self, mock_configure, mock_model_class):
        """Test complete Q&A workflow using Gemini backend."""
        # Mock Gemini API response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This document discusses artificial intelligence and machine learning technologies."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Create backend factory and get Gemini backend
        factory = AIBackendFactory(self.test_config)
        gemini_backend = factory.get_backend("gemini")
        
        # Test the complete workflow
        question = "What is this document about?"
        with open(self.test_txt_file, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Process question
        answer = gemini_backend.process_question(document_text, question)
        
        # Verify response
        self.assertEqual(answer, "This document discusses artificial intelligence and machine learning technologies.")
        
        # Verify API was called correctly
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        self.assertIn(document_text, call_args)
        self.assertIn(question, call_args)
    
    @patch('requests.post')
    @patch('requests.get')
    def test_complete_ollama_workflow(self, mock_get, mock_post):
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
        
        # Configure mock responses based on URL
        def mock_get_side_effect(url, **kwargs):
            if "version" in url:
                return mock_version_response
            elif "tags" in url:
                return mock_models_response
            return Mock(status_code=404)
        
        mock_get.side_effect = mock_get_side_effect
        
        # Mock Ollama generation response
        mock_generate_response = Mock()
        mock_generate_response.status_code = 200
        mock_generate_response.json.return_value = {
            "response": "This document covers artificial intelligence and machine learning concepts."
        }
        mock_post.return_value = mock_generate_response
        
        # Create Ollama backend
        ollama_backend = OllamaBackend()
        
        # Test the complete workflow
        question = "What topics are covered in this document?"
        with open(self.test_txt_file, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Process question
        answer = ollama_backend.process_question(document_text, question)
        
        # Verify response
        self.assertEqual(answer, "This document covers artificial intelligence and machine learning concepts.")
        
        # Verify API calls were made correctly
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']
        self.assertEqual(call_data['model'], ollama_backend.model_name)
        self.assertIn(document_text, call_data['prompt'])
        self.assertIn(question, call_data['prompt'])


class TestBackendSwitching(unittest.TestCase):
    """Test backend switching functionality during runtime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        
        # Patch config file path
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.config_patcher.start()
        
        # Create initial configuration
        self.config = BackendConfig(backend_type="gemini", api_key="test_key")
        self.config.save_to_config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)  
  
    def test_backend_switching_success(self):
        """Test successful backend switching."""
        factory = AIBackendFactory(self.config)
        
        # Initially should be Gemini
        self.assertEqual(factory.config.backend_type, "gemini")
        
        # Switch to Ollama
        result = factory.switch_backend("ollama")
        self.assertTrue(result)
        self.assertEqual(factory.config.backend_type, "ollama")
        
        # Switch back to Gemini
        result = factory.switch_backend("gemini")
        self.assertTrue(result)
        self.assertEqual(factory.config.backend_type, "gemini")
    
    def test_backend_switching_invalid_type(self):
        """Test backend switching with invalid backend type."""
        factory = AIBackendFactory(self.config)
        original_type = factory.config.backend_type
        
        # Try to switch to invalid backend
        result = factory.switch_backend("invalid_backend")
        self.assertFalse(result)
        
        # Should remain unchanged
        self.assertEqual(factory.config.backend_type, original_type)
    
    def test_backend_switching_persistence(self):
        """Test that backend switching persists to configuration file."""
        factory = AIBackendFactory(self.config)
        
        # Switch backend
        factory.switch_backend("ollama")
        
        # Create new factory instance to test persistence
        new_config = BackendConfig.load_from_config()
        new_factory = AIBackendFactory(new_config)
        
        # Should have persisted the change
        self.assertEqual(new_factory.config.backend_type, "ollama")
    
    @patch.object(GeminiBackend, 'update_api_key')
    def test_api_key_update_during_runtime(self, mock_update):
        """Test API key updates during runtime."""
        mock_update.return_value = True
        
        factory = AIBackendFactory(self.config)
        
        # Update API key
        result = factory.update_api_key("new_test_key")
        self.assertTrue(result)
        self.assertEqual(factory.config.api_key, "new_test_key")
        
        # Verify backend was updated
        mock_update.assert_called_once_with("new_test_key")


class TestErrorScenarios(unittest.TestCase):
    """Test error scenarios with Ollama not available or model missing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        
        # Patch config file path
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.config_patcher.start()
        
        self.config = BackendConfig(backend_type="ollama", api_key="")
        self.config.save_to_config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)   
 
    @patch('requests.get')
    def test_ollama_service_unavailable(self, mock_get):
        """Test error handling when Ollama service is not available."""
        # Mock connection error
        mock_get.side_effect = ConnectionError("Connection refused")
        
        ollama_backend = OllamaBackend()
        
        # Service should not be available
        self.assertFalse(ollama_backend.is_available())
        
        # Processing should raise ServiceUnavailableError
        with self.assertRaises(ServiceUnavailableError):
            ollama_backend.process_question("test text", "test question")
    
    @patch('requests.get')
    def test_ollama_model_missing(self, mock_get):
        """Test error handling when Ollama model is missing."""
        # Mock service available but no suitable models
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
        
        ollama_backend = OllamaBackend()
        
        # Should not be available due to missing models
        self.assertFalse(ollama_backend.is_available())
        
        # Processing should raise ServiceUnavailableError
        with self.assertRaises(ServiceUnavailableError):
            ollama_backend.process_question("test text", "test question")
    
    @patch('requests.post')
    @patch('requests.get')
    def test_ollama_processing_timeout(self, mock_get, mock_post):
        """Test timeout handling during Ollama processing."""
        # Mock service and model availability
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.1.0"}
        
        mock_models_response = Mock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "models": [{"name": "gemma2:2b"}]
        }
        
        def mock_get_side_effect(url, **kwargs):
            if "version" in url:
                return mock_version_response
            elif "tags" in url:
                return mock_models_response
            return Mock(status_code=404)
        
        mock_get.side_effect = mock_get_side_effect
        
        # Mock timeout during processing
        mock_post.side_effect = Timeout("Request timed out")
        
        ollama_backend = OllamaBackend()
        
        # Should be available
        self.assertTrue(ollama_backend.is_available())
        
        # Processing should raise ProcessingTimeoutError
        with self.assertRaises(ProcessingTimeoutError):
            ollama_backend.process_question("test text", "test question")
    
    def test_gemini_authentication_error(self):
        """Test error handling for Gemini authentication failures."""
        # Create backend without API key
        gemini_backend = GeminiBackend()
        
        # Should not be available
        self.assertFalse(gemini_backend.is_available())
        
        # Processing should raise AuthenticationError
        with self.assertRaises(AuthenticationError):
            gemini_backend.process_question("test text", "test question")
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_gemini_api_error(self, mock_configure, mock_model_class):
        """Test error handling for Gemini API errors."""
        # Mock API error
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API key invalid")
        mock_model_class.return_value = mock_model
        
        gemini_backend = GeminiBackend("invalid_key")
        
        # Processing should raise AuthenticationError
        with self.assertRaises(AuthenticationError):
            gemini_backend.process_question("test text", "test question")

cla
ss TestLoggingConsistency(unittest.TestCase):
    """Test that logging works consistently across both backends."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, "logbook.txt")
        
        # Patch log file path
        self.log_patcher = patch('mainintegratedWORD.LOG_FILE', self.test_log_file)
        self.log_patcher.start()
        
        # Create root window for QAApp
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during testing
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.log_patcher.stop()
        self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gemini_logging_format(self):
        """Test logging format for Gemini backend responses."""
        # Create QAApp instance
        with patch('backend_config.BackendConfig.load_from_config') as mock_load:
            mock_load.return_value = BackendConfig(backend_type="gemini", api_key="test_key")
            app = QAApp(self.root)
        
        # Test logging
        question = "What is AI?"
        answer = "AI is artificial intelligence."
        app.log_answer(question, answer, "Google Gemini")
        
        # Verify log content
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        self.assertIn("(Google Gemini)", log_content)
        self.assertIn(question, log_content)
        self.assertIn(answer, log_content)
        self.assertIn("Question:", log_content)
        self.assertIn("Answer:", log_content)
    
    def test_ollama_logging_format(self):
        """Test logging format for Ollama backend responses."""
        # Create QAApp instance
        with patch('backend_config.BackendConfig.load_from_config') as mock_load:
            mock_load.return_value = BackendConfig(backend_type="ollama", api_key="")
            app = QAApp(self.root)
        
        # Test logging
        question = "What is machine learning?"
        answer = "Machine learning is a subset of AI."
        app.log_answer(question, answer, "Local Ollama")
        
        # Verify log content
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        self.assertIn("(Local Ollama)", log_content)
        self.assertIn(question, log_content)
        self.assertIn(answer, log_content)
        self.assertIn("Question:", log_content)
        self.assertIn("Answer:", log_content)
    
    def test_logging_backward_compatibility(self):
        """Test logging maintains backward compatibility."""
        # Create QAApp instance
        with patch('backend_config.BackendConfig.load_from_config') as mock_load:
            mock_load.return_value = BackendConfig(backend_type="gemini", api_key="test_key")
            app = QAApp(self.root)
        
        # Test logging without backend name (backward compatibility)
        question = "Test question"
        answer = "Test answer"
        app.log_answer(question, answer)  # No backend name provided
        
        # Verify log content
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Should not contain backend markers when not provided
        self.assertNotIn("(Google Gemini)", log_content)
        self.assertNotIn("(Local Ollama)", log_content)
        self.assertIn(question, log_content)
        self.assertIn(answer, log_content)
    
    def test_multiple_backend_logging(self):
        """Test logging entries from multiple backends in same session."""
        # Create QAApp instance
        with patch('backend_config.BackendConfig.load_from_config') as mock_load:
            mock_load.return_value = BackendConfig(backend_type="gemini", api_key="test_key")
            app = QAApp(self.root)
        
        # Log from Gemini
        app.log_answer("Gemini question", "Gemini answer", "Google Gemini")
        
        # Log from Ollama
        app.log_answer("Ollama question", "Ollama answer", "Local Ollama")
        
        # Verify both entries are logged correctly
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Check Gemini entry
        self.assertIn("(Google Gemini)", log_content)
        self.assertIn("Gemini question", log_content)
        self.assertIn("Gemini answer", log_content)
        
        # Check Ollama entry
        self.assertIn("(Local Ollama)", log_content)
        self.assertIn("Ollama question", log_content)
        self.assertIn("Ollama answer", log_content)
clas
s TestUIUpdates(unittest.TestCase):
    """Test UI updates based on backend availability and selection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "config.txt")
        
        # Patch config file path
        self.config_patcher = patch.object(BackendConfig, 'CONFIG_FILE', self.test_config_file)
        self.config_patcher.start()
        
        # Create test configuration
        self.config = BackendConfig(backend_type="gemini", api_key="test_key")
        self.config.save_to_config()
        
        # Create root window
        self.root = tk.Tk()
        self.root.withdraw()  # Hide during testing
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_initial_ui_state(self, mock_load_config):
        """Test initial UI state based on configuration."""
        mock_load_config.return_value = self.config
        
        app = QAApp(self.root)
        
        # Check initial backend selection
        self.assertEqual(app.backend_var.get(), "gemini")
        
        # Check API key field state (should be enabled for Gemini)
        self.assertEqual(str(app.api_entry.cget('state')), 'normal')
        
        # Check API key value
        self.assertEqual(app.api_entry.get(), "test_key")
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_backend_selection_ui_updates(self, mock_load_config):
        """Test UI updates when backend selection changes."""
        mock_load_config.return_value = self.config
        
        app = QAApp(self.root)
        
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
    
    @patch('backend_config.BackendConfig.load_from_config')
    @patch.object(GeminiBackend, 'is_available')
    @patch.object(OllamaBackend, 'is_available')
    def test_backend_status_indicators(self, mock_ollama_available, mock_gemini_available, mock_load_config):
        """Test backend status indicator updates."""
        mock_load_config.return_value = self.config
        mock_gemini_available.return_value = True
        mock_ollama_available.return_value = False
        
        app = QAApp(self.root)
        app.update_backend_status()
        
        # Check Gemini status (should be ready)
        gemini_status_text = app.gemini_status.cget('text')
        self.assertIn("Ready", gemini_status_text)
        
        # Check Ollama status (should be not available)
        ollama_status_text = app.ollama_status.cget('text')
        self.assertIn("Not Available", ollama_status_text)
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_api_key_change_updates(self, mock_load_config):
        """Test UI updates when API key changes."""
        mock_load_config.return_value = self.config
        
        app = QAApp(self.root)
        
        # Simulate API key change
        app.api_entry.delete(0, tk.END)
        app.api_entry.insert(0, "new_test_key")
        
        # Trigger change event
        app.on_api_key_change()
        
        # Verify configuration was updated
        self.assertEqual(app.backend_factory.config.api_key, "new_test_key")
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_processing_status_display(self, mock_load_config):
        """Test processing status display during operations."""
        mock_load_config.return_value = self.config
        
        app = QAApp(self.root)
        
        # Test showing processing status
        app.show_processing_status("Processing with AI...", True)
        
        # Check status label
        self.assertEqual(app.status_label.cget('text'), "Processing with AI...")
        self.assertEqual(app.status_label.cget('foreground'), "blue")
        
        # Check progress bar is visible
        progress_info = app.progress_bar.grid_info()
        self.assertTrue(len(progress_info) > 0)  # Grid info exists when widget is visible
        
        # Test hiding status
        app.hide_processing_status()
        
        # Check status is cleared
        self.assertEqual(app.status_label.cget('text'), "")
        
        # Check progress bar is hidden
        progress_info = app.progress_bar.grid_info()
        self.assertEqual(len(progress_info), 0)  # No grid info when hidden
    
    @patch('backend_config.BackendConfig.load_from_config')
    def test_error_status_display(self, mock_load_config):
        """Test error status display."""
        mock_load_config.return_value = self.config
        
        app = QAApp(self.root)
        
        # Test showing error status
        app.show_error_status("Connection failed")
        
        # Check status label
        self.assertEqual(app.status_label.cget('text'), "Connection failed")
        self.assertEqual(app.status_label.cget('foreground'), "red")
        
        # Progress bar should be hidden during error
        progress_info = app.progress_bar.grid_info()
        self.assertEqual(len(progress_info), 0)


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite with all integration test classes
    test_suite = unittest.TestSuite()
    
    test_classes = [
        TestCompleteQAWorkflow,
        TestBackendSwitching,
        TestErrorScenarios,
        TestLoggingConsistency,
        TestUIUpdates
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run integration tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    # Print failure and error details
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
            print(f"   {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
            print(f"   {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See details above'}")
    
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"Integration tests {'PASSED' if exit_code == 0 else 'FAILED'}")