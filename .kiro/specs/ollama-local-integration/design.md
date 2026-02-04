# Design Document

## Overview

This design extends the existing mainintegratedWORD.py application to support local AI processing through Ollama with embeddinggemma:latest model. The solution implements a backend abstraction pattern that allows seamless switching between Google Gemini API and local Ollama processing while maintaining the existing user interface and functionality.

## Architecture

The design follows a strategy pattern where different AI backends can be plugged in without changing the core application logic. The main components are:

```
QAApp (Main GUI)
    ├── AIBackendFactory (Creates appropriate backend)
    ├── GeminiBackend (Existing Google API implementation)
    └── OllamaBackend (New local Ollama implementation)
```

### Backend Abstraction

All AI backends implement a common interface:
- `process_question(text: str, question: str) -> str`
- `is_available() -> bool`
- `get_backend_name() -> str`

## Components and Interfaces

### 1. AIBackend Abstract Base Class

```python
from abc import ABC, abstractmethod

class AIBackend(ABC):
    @abstractmethod
    def process_question(self, text: str, question: str) -> str:
        """Process a question against the given text context."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and ready to use."""
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the display name of this backend."""
        pass
```

### 2. OllamaBackend Implementation

The OllamaBackend class handles:
- Ollama service detection via HTTP requests to localhost:11434
- Model availability checking for embeddinggemma:latest
- API communication using requests library
- Error handling for connection and model issues

```python
import requests
import json

class OllamaBackend(AIBackend):
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "embeddinggemma:latest"
    
    def is_available(self) -> bool:
        # Check Ollama service and model availability
        pass
    
    def process_question(self, text: str, question: str) -> str:
        # Format prompt and send to Ollama API
        pass
```

### 3. GeminiBackend Refactoring

Extract existing Gemini logic into a dedicated backend class:

```python
class GeminiBackend(AIBackend):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def is_available(self) -> bool:
        # Check if API key is provided and valid
        pass
    
    def process_question(self, text: str, question: str) -> str:
        # Existing Gemini processing logic
        pass
```

### 4. UI Enhancements

Add backend selection controls to the existing GUI:
- Radio buttons for "Google Gemini" and "Local Ollama"
- Dynamic API key field enabling/disabling
- Status indicators for backend availability
- Progress feedback during processing

## Data Models

### Backend Configuration
```python
@dataclass
class BackendConfig:
    backend_type: str  # "gemini" or "ollama"
    api_key: str = ""  # Only used for Gemini
    
    def save_to_config(self):
        """Save configuration to config.txt"""
        pass
    
    @classmethod
    def load_from_config(cls):
        """Load configuration from config.txt"""
        pass
```

### Processing Status
```python
from enum import Enum

class ProcessingStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
```

## Error Handling

### Ollama-Specific Errors
1. **Service Unavailable**: Ollama not running on localhost:11434
   - Display: "Ollama service not found. Please ensure Ollama is running."
   - Action: Disable Ollama option, fallback to Gemini if available

2. **Model Not Found**: embeddinggemma:latest not installed
   - Display: "embeddinggemma:latest model not found. Run: ollama pull embeddinggemma:latest"
   - Action: Provide installation instructions

3. **API Communication Errors**: Network timeouts, malformed responses
   - Display: "Failed to communicate with Ollama. Please check the service."
   - Action: Log error details, allow retry

### Graceful Degradation
- If Ollama is unavailable, automatically select Gemini backend if API key exists
- If both backends fail, display clear error message with troubleshooting steps
- Maintain application stability regardless of backend failures

## Testing Strategy

### Unit Tests
1. **Backend Interface Compliance**
   - Test that both backends implement the AIBackend interface correctly
   - Verify is_available() methods work as expected
   - Test error handling for various failure scenarios

2. **Ollama Integration Tests**
   - Mock Ollama API responses for testing
   - Test model availability checking
   - Verify prompt formatting and response parsing

3. **Configuration Management**
   - Test saving/loading backend preferences
   - Verify API key handling and security
   - Test configuration migration from existing format

### Integration Tests
1. **End-to-End Workflow**
   - Test complete Q&A flow with both backends
   - Verify logging works consistently across backends
   - Test backend switching during runtime

2. **UI Behavior**
   - Test radio button state management
   - Verify API key field enabling/disabling
   - Test progress indicators and status messages

### Manual Testing Scenarios
1. **Ollama Setup Verification**
   - Test with Ollama not installed
   - Test with Ollama running but model missing
   - Test with complete Ollama + embeddinggemma setup

2. **Backend Switching**
   - Switch between backends during session
   - Test with different document types (TXT, PDF, DOCX)
   - Verify consistent behavior across backends

## Implementation Notes

### Dependencies
- Add `requests` library for Ollama HTTP API communication
- No additional GUI libraries needed (using existing tkinter)
- Maintain compatibility with existing dependencies

### Configuration Storage
Extend existing config.txt format to include backend preference:
```
api_key=your_google_api_key_here
backend=ollama
```

### Ollama API Integration
Use Ollama's REST API endpoint `/api/generate` for text generation:
```json
{
  "model": "embeddinggemma:latest",
  "prompt": "Context: ...\n\nQuestion: ...",
  "stream": false
}
```

### Performance Considerations
- Local Ollama processing may be slower than cloud API
- Implement timeout handling for Ollama requests
- Consider chunking strategy for large documents with local processing
- Cache Ollama availability checks to avoid repeated network calls