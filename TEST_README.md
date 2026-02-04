# AI Backend Unit Tests

This document describes the comprehensive unit test suite for the AI backend implementations in the Ollama Local Integration feature.

## Overview

The test suite covers all backend implementations and ensures they comply with the AIBackend interface while providing robust error handling and functionality testing.

## Test Files

### `test_backends.py`
Main test file containing all unit tests for the backend implementations.

### `run_tests.py`
Test runner script that provides convenient options for running tests with different configurations.

## Test Coverage

### 1. AIBackend Interface Compliance Tests (`TestAIBackendInterface`)

Tests that verify the abstract base class works correctly:

- **Abstract Method Enforcement**: Ensures AIBackend cannot be instantiated directly
- **Concrete Method Functionality**: Tests helper methods like `validate_inputs()` and `get_backend_type()`
- **Input Validation**: Verifies proper validation of text and question inputs

### 2. OllamaBackend Tests (`TestOllamaBackend`)

Comprehensive tests for the local Ollama backend:

#### Interface Compliance
- Verifies implementation of all required AIBackend methods
- Tests backend name and type identification

#### Service Availability Checking
- **Service Running**: Tests detection when Ollama service is available
- **Service Unavailable**: Tests handling when Ollama is not running
- **Timeout Handling**: Tests behavior when service checks timeout
- **Caching**: Verifies availability checks are cached to avoid excessive network calls

#### Model Availability Checking
- **Suitable Models**: Tests detection of text generation models
- **Embedding Models Only**: Tests handling when only embedding models are available
- **No Models**: Tests behavior when no models are installed
- **Model Preference**: Tests selection of preferred models from available options

#### Question Processing
- **Successful Processing**: Tests complete question-answer workflow with mocked API
- **Service Unavailable**: Tests error handling when service is not available
- **Network Errors**: Tests retry logic for connection failures
- **Timeout Errors**: Tests handling of processing timeouts
- **Input Validation**: Tests rejection of invalid inputs

#### API Communication
- **Request Formation**: Tests proper formatting of Ollama API requests
- **Response Parsing**: Tests extraction of answers from API responses
- **Error Status Codes**: Tests handling of API error responses
- **Prompt Formatting**: Tests proper context and question formatting

### 3. GeminiBackend Tests (`TestGeminiBackend`)

Tests for the Google Gemini API backend:

#### Interface Compliance
- Verifies implementation of all required AIBackend methods
- Tests backend identification and model information

#### Authentication and Availability
- **API Key Validation**: Tests availability checking with different key states
- **No API Key**: Tests behavior when no key is provided
- **Empty/Whitespace Keys**: Tests handling of invalid key formats
- **Key Updates**: Tests dynamic API key updating functionality

#### Question Processing
- **Successful Processing**: Tests complete workflow with mocked Gemini API
- **Authentication Errors**: Tests handling of invalid API keys
- **Network Issues**: Tests retry logic for network failures
- **Input Validation**: Tests proper input validation

### 4. AIBackendFactory Tests (`TestAIBackendFactory`)

Tests for the backend factory and management:

#### Factory Functionality
- **Initialization**: Tests proper setup of all backend instances
- **Backend Retrieval**: Tests getting backends by type
- **Current Backend**: Tests getting the currently configured backend
- **Backend Switching**: Tests runtime switching between backends

#### Configuration Management
- **API Key Updates**: Tests updating Gemini API keys
- **Backend Selection**: Tests changing backend preferences
- **Status Reporting**: Tests getting detailed backend status information

#### Availability Management
- **Available Backends**: Tests getting list of currently available backends
- **Fallback Logic**: Tests automatic fallback when preferred backend is unavailable
- **Error Information**: Tests detailed error reporting for troubleshooting

### 5. Backend Availability Checking Tests (`TestBackendAvailabilityChecking`)

Specialized tests for availability checking across all backends:

#### Caching Behavior
- **Ollama Caching**: Tests that availability checks are cached appropriately
- **Cache Duration**: Verifies cache expiration and refresh logic

#### State Management
- **Gemini Key States**: Tests availability with various API key configurations
- **Model Selection**: Tests Ollama's preferred model selection logic

## Requirements Coverage

The tests address all requirements specified in the task:

### Requirement 2.1 (Ollama Service Detection)
- ✅ Tests for service availability checking via HTTP requests
- ✅ Tests for proper error handling when service is unavailable
- ✅ Tests for timeout handling during service checks

### Requirement 2.2 (Model Availability)
- ✅ Tests for embeddinggemma:latest model detection
- ✅ Tests for handling when only embedding models are available
- ✅ Tests for alternative model selection from preferred list
- ✅ Tests for proper error messages when no suitable models found

### Requirement 3.1 (Backend Interface)
- ✅ Tests for AIBackend interface compliance across all implementations
- ✅ Tests for consistent method signatures and behavior
- ✅ Tests for proper error handling and exception types

### Requirement 3.3 (Error Handling)
- ✅ Tests for connection failure handling with retry logic
- ✅ Tests for timeout handling with appropriate error messages
- ✅ Tests for authentication error handling in Gemini backend
- ✅ Tests for graceful degradation when backends are unavailable

## Running the Tests

### Run All Tests
```bash
python test_backends.py
```

### Using the Test Runner

#### Run all tests with summary
```bash
python run_tests.py
```

#### Run specific test class
```bash
python run_tests.py --class TestOllamaBackend
```

#### Run with verbose output
```bash
python run_tests.py --verbose
```

#### Run quietly (summary only)
```bash
python run_tests.py --quiet
```

#### Enable debug logging
```bash
python run_tests.py --debug
```

#### List available test classes
```bash
python run_tests.py --list-classes
```

## Test Results

The test suite includes 55 comprehensive tests covering:

- **Interface compliance**: 2 tests
- **OllamaBackend functionality**: 23 tests  
- **GeminiBackend functionality**: 15 tests
- **AIBackendFactory functionality**: 12 tests
- **Availability checking**: 3 tests

All tests use proper mocking to avoid dependencies on external services and ensure consistent, reliable test execution.

## Mock Strategy

The tests use `unittest.mock` to:

- **Mock HTTP requests** to Ollama API endpoints
- **Mock Google Generative AI** library calls
- **Mock network failures** and timeouts for error testing
- **Mock service availability** for different scenarios
- **Isolate unit tests** from external dependencies

This ensures tests run quickly and consistently regardless of whether actual services are available.

## Error Scenarios Tested

1. **Network Issues**: Connection failures, timeouts, DNS errors
2. **Service Unavailable**: Ollama not running, API endpoints unreachable
3. **Authentication Failures**: Invalid API keys, authorization errors
4. **Model Issues**: Missing models, embedding-only models, model loading failures
5. **Input Validation**: Empty inputs, malformed data, edge cases
6. **API Errors**: HTTP error codes, malformed responses, rate limiting

## Future Enhancements

The test suite is designed to be easily extensible for:

- Additional backend implementations
- New error scenarios and edge cases
- Performance and load testing
- Integration testing with real services
- Automated CI/CD pipeline integration