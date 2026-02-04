# Implementation Plan

- [x] 1. Create backend abstraction layer





  - Create AIBackend abstract base class with required interface methods
  - Define common interface for process_question, is_available, and get_backend_name methods
  - Add type hints and docstrings for clear interface definition
  - _Requirements: 1.1, 3.1_

- [x] 2. Implement OllamaBackend class





  - Create OllamaBackend class inheriting from AIBackend
  - Implement Ollama service detection via HTTP requests to localhost:11434
  - Add model availability checking for embeddinggemma:latest model
  - Implement process_question method using Ollama's /api/generate endpoint
  - Add proper error handling for connection failures and model issues
  - _Requirements: 2.1, 2.2, 2.4, 3.1, 3.3_

- [x] 3. Refactor existing Gemini code into GeminiBackend class





  - Extract current Gemini API logic from QAApp into separate GeminiBackend class
  - Implement AIBackend interface methods for Gemini processing
  - Maintain existing functionality while conforming to new backend interface
  - Add API key validation in is_available method
  - _Requirements: 1.3, 3.1_

- [x] 4. Create backend configuration management





  - Implement BackendConfig dataclass for storing backend preferences
  - Add methods to save and load backend configuration from config.txt
  - Extend existing config file format to include backend selection
  - Ensure backward compatibility with existing config.txt files
  - _Requirements: 1.1, 1.4_

- [x] 5. Add backend selection UI controls





  - Add radio buttons to GUI for selecting between "Google Gemini" and "Local Ollama"
  - Implement dynamic API key field enabling/disabling based on backend selection
  - Add backend availability status indicators in the UI
  - Position new controls appropriately in existing layout
  - _Requirements: 1.1, 1.2, 1.3, 2.2_

- [x] 6. Implement backend factory and integration





  - Create AIBackendFactory class to instantiate appropriate backend based on configuration
  - Integrate backend selection logic into QAApp class
  - Replace direct Gemini API calls with backend abstraction calls
  - Add backend switching functionality during runtime
  - _Requirements: 1.1, 1.4, 3.1_

- [x] 7. Add processing status feedback





  - Implement progress indicators for local Ollama processing
  - Add status messages to inform users about processing state
  - Disable Ask button during processing to prevent multiple simultaneous requests
  - Add timeout handling for potentially slower local processing
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 8. Update logging functionality





  - Modify log_answer method to include backend information in log entries
  - Ensure consistent logging format regardless of backend used
  - Add backend name markers to distinguish between different AI responses
  - Maintain backward compatibility with existing log format
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 9. Implement comprehensive error handling





  - Add specific error messages for Ollama service unavailable scenarios
  - Create user-friendly error messages for missing embeddinggemma model
  - Implement graceful fallback when preferred backend is unavailable
  - Add error recovery and retry mechanisms for transient failures
  - _Requirements: 2.2, 2.4, 3.3_

- [x] 10. Add dependency management and imports





  - Add requests library import for Ollama HTTP API communication
  - Update existing imports to support new backend classes
  - Add proper exception handling imports for new error scenarios
  - Ensure all required dependencies are properly imported
  - _Requirements: 3.1, 3.3_

- [x] 11. Create unit tests for backend implementations





  - Write tests for AIBackend interface compliance
  - Create mock tests for OllamaBackend API communication
  - Test GeminiBackend refactored functionality
  - Add tests for backend availability checking methods
  - _Requirements: 2.1, 2.2, 3.1, 3.3_

- [x] 12. Integration testing and final validation





  - Test complete Q&A workflow with both Gemini and Ollama backends
  - Verify backend switching works correctly during runtime
  - Test error scenarios with Ollama not available or model missing
  - Validate that logging works consistently across both backends
  - Ensure UI updates correctly based on backend availability and selection
  - _Requirements: 1.1, 1.4, 2.2, 4.1, 5.1_