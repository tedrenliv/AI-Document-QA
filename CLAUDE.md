# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python AI Q&A application that supports multiple backend AI services. It enables users to ask questions about text documents, PDFs, or Word files using either Google Gemini (cloud) or local Ollama (on-device) as the processing backend. The application uses Chroma DB for vector storage and semantic search with embeddings.

## Architecture

### Core Components

**Backend Abstraction Layer** (`ai_backend.py`):
- Abstract base class `AIBackend` defining the interface all backends must implement
- Three core methods: `process_question()`, `is_available()`, `get_backend_name()`
- Helper methods: `validate_inputs()`, `get_backend_type()`
- All backends must follow this interface for interchangeability

**Backend Implementations**:
- `GeminiBackend` (`gemini_backend.py`): Google Generative AI client with API key validation
- `OllamaBackend` (`ollama_backend.py`): Local Ollama service at localhost:11434 using embeddinggemma:latest model

**Factory Pattern** (`ai_backend_factory.py`):
- `AIBackendFactory`: Centralized creation and management of backend instances
- Handles backend initialization, availability checking, and runtime switching
- Integrates with `FallbackManager` for graceful degradation when primary backend is unavailable

**Configuration** (`backend_config.py`):
- `BackendConfig`: Dataclass managing backend selection and API keys
- Persists preferences to `config.txt` (supports both old single-line and new key=value formats)
- Methods: `save_to_config()`, `load_from_config()`, `is_gemini_configured()`, `is_ollama_configured()`

**Error Handling** (`ai_backend_errors.py`):
- Custom exception hierarchy: `AIBackendError`, `ServiceUnavailableError`, `ModelNotFoundError`, `AuthenticationError`, `ProcessingTimeoutError`, `NetworkError`
- `ErrorMessageGenerator`: Produces user-friendly error messages
- `FallbackManager`: Implements fallback logic when primary backend unavailable
- `RetryManager`: Handles retry logic with exponential backoff for transient failures

**Document Processing** (`chunk.py`):
- `read_data()`: Reads text files with multiple encoding fallbacks (UTF-8, Latin1, CP1252, UTF-16)
- `get_chunks()`: Splits documents into semantic chunks while preserving header context

### User Applications

- `main.py`: CLI-based Q&A using Chroma DB vector search + LLM
- `mainintegratedWORD.py`: tkinter GUI app supporting TXT, PDF, and DOCX files with backend selection UI
- `mainintegratedPDF.py`: Focused PDF Q&A application

## Development Commands

### Run Tests

```bash
# Run all unit tests
python test_backends.py

# Use test runner with options
python run_tests.py                          # Run all tests with summary
python run_tests.py --class TestOllamaBackend  # Run specific test class
python run_tests.py --verbose                # Detailed output
python run_tests.py --quiet                  # Summary only
python run_tests.py --debug                  # Debug logging enabled
python run_tests.py --list-classes           # List available test classes
```

### Run Application

```bash
# GUI application with Word/PDF/TXT support
python mainintegratedWORD.py

# CLI interface with vector search
python main.py --build                       # Index data.txt into Chroma DB
python main.py --ask "your question"         # Query the indexed documents
```

## Configuration

**config.txt** format (loaded on startup):
```
api_key=your_google_api_key
backend=gemini    # or "ollama"
```

**Environment Variables**:
- Backends may check environment variables as fallback for API keys
- Ollama assumes localhost:11434 (default)

## Key Directories

- `./chroma_db/`: Persistent vector database storage
- `.idea/`: PyCharm IDE configuration
- `.kiro/`: Project specs and design documentation
- `.venv/`: Python virtual environment

## Testing Strategy

**Test Files**:
- `test_backends.py`: Unit tests for all backend implementations (55 comprehensive tests)
- `test_integration.py`, `test_integration_final.py`: Integration tests

**Coverage**:
- Interface compliance, service availability checking, model detection
- Question processing, error handling, network failures
- API communication, authentication, input validation
- Fallback logic and backend switching

Uses `unittest.mock` to isolate from external services.

## Common Patterns

**Backend Usage**:
```python
from backend_config import BackendConfig
from ai_backend_factory import AIBackendFactory

config = BackendConfig.load_from_config()
factory = AIBackendFactory(config)
backend, fallback_msg = factory.get_current_backend_with_fallback()
if backend:
    answer = backend.process_question(context_text, user_question)
```

**Error Handling**:
- All backends raise custom exceptions inheriting from `AIBackendError`
- Use `ErrorMessageGenerator.get_user_friendly_message()` for UI display
- Input validation via `backend.validate_inputs(text, question)` before processing

**Document Processing**:
```python
from chunk import read_data, get_chunks
text = read_data(file_path)
chunks = get_chunks(text)  # Returns list of semantic chunks with preserved headers
```

## Data Files

- `data.txt`, `data10.txt`, `data11.txt`, `data12.txt`: Sample documents for testing
- `logbook.txt`: Activity log from application usage
