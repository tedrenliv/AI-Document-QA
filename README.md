# Skill-Enhanced RAG Expert System V3.0

**by Prof. Jun Ren**

A Python application for asking natural-language questions about text documents, PDFs, and Word files. Supports **Google Gemini** (cloud) and **Ollama** (local, on-device) as interchangeable AI backends, with a modern browser-based web interface.

---

## Features

- **Modern web UI** — Flask-powered single-page application with a responsive 3-panel layout
- **Multi-file support** — upload and query multiple TXT, PDF, and DOCX files simultaneously
- **Document preview panel** — browse the extracted text of each uploaded file directly in the UI, with tabbed navigation between files and a collapse toggle
- **Dual backend** — Google Gemini (cloud API) or Ollama (local inference), switchable at runtime
- **Ollama model picker** — dropdown in the sidebar lists all installed text-generation models; select one without restarting
- **Automatic fallback** — switches to an available backend if the preferred one is down
- **Live backend status** — green/orange/red indicators poll availability every 10 seconds
- **Large context support** — Gemini 2.0 Flash handles up to ~500,000 chars; Ollama up to ~100,000 chars
- **Vector search (RAG)** — CLI mode embeds documents into ChromaDB and retrieves relevant chunks before answering
- **Retry & error handling** — exponential backoff, rate-limit detection (with `Retry-After` header support), custom exception hierarchy, user-friendly error messages
- **Q&A logging** — every question and answer is appended to `logbook.txt` with timestamp and backend name

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Applications                        │
│  web_app.py             (Flask web server + REST API)    │
│  mainintegratedWORD.py  (Tkinter GUI, all formats)       │
│  mainintegratedPDF.py   (Tkinter GUI, PDF)               │
│  main.py                (CLI + ChromaDB RAG)             │
└──────────────────┬───────────────────────────────────────┘
                   │
          ┌────────▼─────────┐
          │ AIBackendFactory  │  ← factory + fallback manager
          └────────┬─────────┘
                   │
          ┌────────▼─────────┐
          │    AIBackend      │  ← abstract interface
          │  ├─ GeminiBackend │     gemini_backend.py
          │  └─ OllamaBackend │     ollama_backend.py
          └────────┬─────────┘
                   │
          ┌────────▼─────────┐
          │   Shared Utils    │
          │  chunk.py         │  ← read files, chunking
          │  backend_config   │  ← persistent config
          │  ai_backend_errors│  ← exceptions, retry, fallback
          └──────────────────┘
```

### Web API endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/api/config` | Return saved API key and backend preference |
| `GET` | `/api/backend-status` | Check Gemini & Ollama availability |
| `POST` | `/api/switch-backend` | Switch active backend |
| `POST` | `/api/update-api-key` | Update Gemini API key |
| `GET` | `/api/files` | List uploaded filenames |
| `POST` | `/api/upload` | Upload one or more documents |
| `POST` | `/api/remove-file` | Remove a specific uploaded file |
| `POST` | `/api/file-content` | Retrieve extracted text for preview |
| `POST` | `/api/ask` | Ask a question across all uploaded files |
| `GET` | `/api/ollama-models` | List available Ollama text-generation models |
| `POST` | `/api/set-ollama-model` | Set the active Ollama model |

---

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
# Option 1: config.txt
echo "api_key=your-key-here" > config.txt
echo "backend=gemini"       >> config.txt

# Option 2: enter it directly in the web UI
```

**Ollama** — install a text generation model:

```bash
ollama pull llama3.2:1b   # recommended (fast, good quality)
# or: ollama pull gemma2:2b / phi3:mini / llama3.2:3b
```

### Run the web app

```bash
python web_app.py
# then open http://localhost:5000 in your browser
```

### Run the legacy Tkinter GUI

```bash
python mainintegratedWORD.py
```

### Run the CLI (vector search / RAG)

```bash
python main.py --build               # index data.txt into ChromaDB (resumable — safe to re-run daily)
python main.py --ask "your question" # query indexed documents
```

> **Note:** The `--build` command is **resumable**. It uses deterministic chunk IDs so re-running skips already-embedded chunks. If you hit the free-tier daily quota (1,000 embeddings/day), just run `--build` again the next day until the index is complete.

---

## Web Interface

The browser UI has three panels:

| Panel | Contents |
|-------|----------|
| **Left sidebar** | AI backend selector (Gemini / Ollama) with live status badges; Ollama model dropdown (appears when Ollama is selected); Google API key field; multi-file upload zone with per-file remove |
| **Document Preview** | Tabbed viewer showing the extracted text of each uploaded file; click a tab to switch files; collapse to a slim strip when not needed |
| **Q&A** | Question input (Enter to submit), processing status bar, answer display with copy button and source metadata |

---

## Configuration

Settings are stored in `config.txt` (auto-created on first run):

```
api_key=your_google_api_key
backend=gemini
```

Accepted `backend` values: `gemini`, `ollama`.

---

## Project Structure

```
.
├── web_app.py               # Flask web server & REST API
├── templates/
│   └── index.html           # Single-page web UI
├── ai_backend.py            # Abstract backend interface
├── ai_backend_errors.py     # Custom exceptions, retry/fallback managers
├── ai_backend_factory.py    # Factory for creating & switching backends
├── backend_config.py        # Persistent configuration (config.txt)
├── chunk.py                 # Document reading (TXT/PDF/DOCX) & chunking
├── gemini_backend.py        # Google Gemini backend (2.0 Flash)
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

---

## Testing

```bash
# Run all tests
python run_tests.py

# Specific test class
python run_tests.py --class TestOllamaBackend

# Verbose / quiet / debug
python run_tests.py --verbose
python run_tests.py --quiet
python run_tests.py --debug

# List available test classes
python run_tests.py --list-classes
```

---

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

---

## Error Handling

All backends raise exceptions from `ai_backend_errors.py`:

| Exception | When |
|-----------|------|
| `ServiceUnavailableError` | Backend service is not running |
| `ModelNotFoundError` | Required model is not installed |
| `InvalidModelError` | Model exists but wrong type (e.g. embedding-only) |
| `AuthenticationError` | Invalid or missing API key |
| `ProcessingTimeoutError` | Request exceeded timeout |
| `RateLimitError` | API quota exceeded (429); respects `Retry-After` header |
| `NetworkError` | Connection / DNS failure |

Use `ErrorMessageGenerator` to convert any of these into user-friendly strings.
