import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ai_backend_errors import (
    AIBackendError,
    AuthenticationError,
    ErrorMessageGenerator,
    InvalidModelError,
    ModelNotFoundError,
    NetworkError,
    ProcessingTimeoutError,
    ServiceUnavailableError,
)
from ai_backend_factory import AIBackendFactory
from backend_config import BackendConfig
from chunk import read_data

app = Flask(__name__)

LOG_FILE = "logbook.txt"

# Global application state (single-user local app)
_config = BackendConfig.load_from_config()
_factory = AIBackendFactory(_config)
_uploaded_files: dict[str, str] = {}  # display_name → temp_path
_state_lock = threading.Lock()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "api_key": _config.api_key,
        "backend": _config.backend_type,
    })


@app.route("/api/backend-status")
def backend_status():
    gemini = _factory.get_backend_status("gemini")
    ollama = _factory.get_backend_status("ollama")
    return jsonify({
        "gemini": gemini,
        "ollama": ollama,
        "current": _config.backend_type,
    })


@app.route("/api/switch-backend", methods=["POST"])
def switch_backend():
    data = request.get_json()
    backend_type = data.get("backend")
    if backend_type not in ("gemini", "ollama"):
        return jsonify({"success": False, "error": "Invalid backend type"}), 400
    success = _factory.switch_backend(backend_type)
    return jsonify({"success": success})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    _factory.update_api_key(api_key)
    return jsonify({"success": True})


@app.route("/api/files")
def list_files():
    with _state_lock:
        return jsonify({"files": list(_uploaded_files.keys())})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    global _uploaded_files

    files = request.files.getlist("file")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "No files provided"}), 400

    added, skipped = [], []
    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in {".txt", ".pdf", ".docx"}:
            skipped.append(file.filename)
            continue

        with _state_lock:
            # Clean up old temp file if name already exists
            if file.filename in _uploaded_files:
                old_path = _uploaded_files[file.filename]
                if os.path.exists(old_path):
                    try:
                        os.unlink(old_path)
                    except OSError:
                        pass

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                file.save(tmp.name)
                _uploaded_files[file.filename] = tmp.name

        added.append(file.filename)

    if not added:
        return jsonify({"error": f"No supported files. Skipped: {', '.join(skipped)}"}), 400

    return jsonify({"added": added, "skipped": skipped})


@app.route("/api/remove-file", methods=["POST"])
def remove_file():
    global _uploaded_files

    data = request.get_json()
    filename = data.get("filename", "")

    with _state_lock:
        if filename not in _uploaded_files:
            return jsonify({"error": "File not found"}), 404
        path = _uploaded_files.pop(filename)

    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass

    return jsonify({"success": True})


@app.route("/api/file-content", methods=["POST"])
def get_file_content():
    data = request.get_json()
    filename = data.get("filename", "")

    with _state_lock:
        if filename not in _uploaded_files:
            return jsonify({"error": "File not found"}), 404
        path = _uploaded_files[filename]

    if not os.path.exists(path):
        return jsonify({"error": "File no longer available"}), 404

    try:
        text = read_data(Path(path))
        limit = 50_000
        truncated = len(text) > limit
        return jsonify({
            "filename": filename,
            "content": text[:limit] if truncated else text,
            "truncated": truncated,
            "total_chars": len(text),
        })
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    with _state_lock:
        files_snapshot = dict(_uploaded_files)

    if not files_snapshot:
        return jsonify({"error": "Please upload at least one document file first."}), 400

    current_backend, fallback_message = _factory.get_current_backend_with_fallback()
    if not current_backend:
        return jsonify({"error": "No AI backend is available. Please configure a Gemini API key or start Ollama."}), 503

    try:
        parts = []
        for name, path in files_snapshot.items():
            if os.path.exists(path):
                parts.append(f"[Document: {name}]\n{read_data(Path(path))}")

        if not parts:
            return jsonify({"error": "Uploaded files could not be read."}), 400

        combined_text = "\n\n---\n\n".join(parts)
        answer = current_backend.process_question(combined_text, question)
        backend_name = current_backend.get_backend_name()
        _log_answer(question, answer, backend_name)
        return jsonify({
            "answer": answer,
            "backend": backend_name,
            "file_count": len(parts),
            "fallback_message": fallback_message,
        })

    except ServiceUnavailableError as e:
        return jsonify({"error": e.details.get("message", str(e))}), 503
    except ModelNotFoundError as e:
        msg = e.message
        if e.installation_cmd:
            msg += f"\n\nTo install: {e.installation_cmd}"
        return jsonify({"error": msg}), 404
    except InvalidModelError as e:
        suggestions = "\n".join(f"• ollama pull {m}" for m in e.suggested_models[:3])
        return jsonify({"error": f"{e.message}\n\nSuggested alternatives:\n{suggestions}"}), 400
    except AuthenticationError as e:
        return jsonify({"error": e.details.get("message", ErrorMessageGenerator.get_gemini_auth_error_message())}), 401
    except ProcessingTimeoutError as e:
        return jsonify({"error": ErrorMessageGenerator.get_timeout_error_message(current_backend.get_backend_name(), e.timeout_seconds)}), 408
    except NetworkError as e:
        return jsonify({"error": ErrorMessageGenerator.get_network_error_message(e.service_name)}), 503
    except AIBackendError as e:
        return jsonify({"error": f"{e.message}\n\nError Code: {e.error_code}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _log_answer(question: str, answer: str, backend_name: str | None = None) -> None:
    if backend_name:
        if "Gemini" in backend_name:
            marker = " (Google Gemini)"
        elif "Ollama" in backend_name:
            marker = " (Local Ollama)"
        else:
            marker = f" ({backend_name})"
    else:
        marker = ""
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now()}]{marker}\nQuestion: {question}\nAnswer: {answer}\n{'-' * 40}\n")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting AI Q&A Web App...")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=False, host="localhost", port=5000, threaded=True)
