# AI Document Q&A

A Python application for asking natural-language questions about text documents, PDFs, and Word files. Supports **Google Gemini** (cloud) and **Ollama** (local, on-device) as interchangeable AI backends.

## Features

- **Multi-format support** — TXT, PDF, and DOCX files
- **Dual backend** — Google Gemini (cloud API) or Ollama (local inference)
- **Automatic fallback** — switches to an available backend if the preferred one is down
- **Vector search (RAG)** — CLI mode embeds documents into ChromaDB and retrieves relevant chunks before answering
- **Threaded GUI** — UI stays responsive during long-running AI calls
- **Retry & error handling** — exponential backoff, custom exception hierarchy, user-friendly error messages

## Architecture

```
┌─────────────────────────────────────────┐
│            Applications                 │
│  mainintegratedWORD.py  (GUI, all docs) │
│  mainintegratedPDF.py   (GUI, PDF)      │
│  main.py                (CLI + RAG)     │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼────────┐
       │ AIBackendFactory│  ← factory + fallback
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │   AIBackend     │  ← abstract interface
       │  ├─ Gemini      │     (gemini_backend.py)
       │  └─ Ollama      │     (ollama_backend.py)
       └────────────────┘
               │
       ┌───────▼────────┐
       │  Shared Utils   │
       │  chunk.py       │  ← read files, chunking
       │  backend_config │  ← persistent config
       │  ai_backend_errors │ ← exceptions, retry, fallback
       └────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai) installed locally for on-device inference

### Installation

```bash
pip install -r requirements.txt
```

### Configure a backend

**Google Gemini** — set your API key:

```bash
# Option 1: environment variable
export GOOGLE_API_KEY="your-key-here"

# Option 2: enter it in the GUI when prompted

# Option 3: config.txt
echo "api_key=your-key-here" > config.txt
echo "backend=gemini" >> config.txt
```

**Ollama** — install a text generation model:

```bash
ollama run llama3.2:1b       # recommended (fast, good quality)
# or: ollama pull gemma2:2b / phi3:mini
```

### Run

```bash
# GUI (supports TXT, PDF, DOCX)
python mainintegratedWORD.py

# PDF-only GUI
python mainintegratedPDF.py

# CLI with vector search (RAG)
python main.py --build               # index data.txt into ChromaDB
python main.py --ask "your question"  # query indexed documents
```

## Configuration

Settings are stored in `config.txt` (auto-created on first run):

```
api_key=your_google_api_key
backend=gemini
```

Accepted `backend` values: `gemini`, `ollama`.

## Project Structure

```
.
├── ai_backend.py            # Abstract backend interface
├── ai_backend_errors.py     # Custom exceptions, retry/fallback managers
├── ai_backend_factory.py    # Factory for creating & switching backends
├── backend_config.py        # Persistent configuration (config.txt)
├── chunk.py                 # Document reading (TXT/PDF/DOCX) & chunking
├── gemini_backend.py        # Google Gemini backend
├── ollama_backend.py        # Local Ollama backend
├── main.py                  # CLI app with ChromaDB vector search
├── mainintegratedWORD.py    # Tkinter GUI (TXT, PDF, DOCX)
├── mainintegratedPDF.py     # Tkinter GUI (PDF focused)
├── test_backends.py         # Unit tests (55 tests)
├── test_integration.py      # Integration tests
├── run_tests.py             # Test runner CLI
├── requirements.txt         # Python dependencies
├── config.txt               # Runtime configuration (auto-generated)
└── logbook.txt              # Q&A activity log (auto-generated)
```

## Testing

```bash
# Run all tests
python run_tests.py

# Unit tests only
python run_tests.py --unit-only

# Integration tests only
python run_tests.py --integration-only

# Specific test class
python run_tests.py --class TestOllamaBackend

# Verbose / quiet / debug
python run_tests.py --verbose
python run_tests.py --quiet
python run_tests.py --debug

# List available test classes
python run_tests.py --list-classes
```

## Backend Usage (programmatic)

```python
from backend_config import BackendConfig
from ai_backend_factory import AIBackendFactory

config = BackendConfig.load_from_config()
factory = AIBackendFactory(config)
backend, fallback_msg = factory.get_current_backend_with_fallback()

if backend:
    answer = backend.process_question(context_text, user_question)
```

## Error Handling

All backends raise exceptions from `ai_backend_errors.py`:

| Exception | When |
|-----------|------|
| `ServiceUnavailableError` | Backend service is not running |
| `ModelNotFoundError` | Required model is not installed |
| `InvalidModelError` | Model exists but wrong type (e.g. embedding-only) |
| `AuthenticationError` | Invalid or missing API key |
| `ProcessingTimeoutError` | Request exceeded timeout |
| `NetworkError` | Connection / DNS failure |

Use `ErrorMessageGenerator` to convert any of these into user-friendly strings.
