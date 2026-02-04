import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import threading

from chunk import read_data, get_chunks
from backend_config import BackendConfig
from ai_backend import AIBackend
from ai_backend_factory import AIBackendFactory
from ai_backend_errors import (
    AIBackendError, ServiceUnavailableError, ModelNotFoundError,
    InvalidModelError, AuthenticationError, ProcessingTimeoutError,
    NetworkError, ErrorMessageGenerator
)

LOG_FILE = "logbook.txt"


class PDFQAApp:
    """Focused PDF Q&A application using the backend abstraction layer."""

    def __init__(self, root):
        self.root = root
        self.root.title("AI Q&A (PDF)")
        self.selected_file = None

        self.config = BackendConfig.load_from_config()
        self.backend_factory = AIBackendFactory(self.config)

        self.build_gui()

    def build_gui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        # API Key
        ttk.Label(frame, text="Google API Key:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.api_entry = ttk.Entry(frame, width=60)
        self.api_entry.insert(0, self.config.api_key)
        self.api_entry.grid(row=0, column=1, padx=10, pady=5)
        self.api_entry.bind("<KeyRelease>", self._on_api_key_change)

        # File selection
        ttk.Label(frame, text="Select PDF file:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w", pady=10)
        self.file_label = ttk.Label(frame, text="No file selected", foreground="blue")
        self.file_label.grid(row=1, column=1, sticky="w")
        ttk.Button(frame, text="Select File", command=self.select_file).grid(row=1, column=2, padx=10, pady=10)

        # Question
        ttk.Label(frame, text="Your Question:", font=("Arial", 11)).grid(row=2, column=0, sticky="w", pady=10)
        self.question_entry = ttk.Entry(frame, width=80)
        self.question_entry.grid(row=2, column=1, columnspan=2, pady=5)

        # Answer
        ttk.Label(frame, text="Answer:", font=("Arial", 11)).grid(row=3, column=0, sticky="nw", pady=10)
        self.answer_text = tk.Text(frame, height=15, width=90, wrap="word")
        self.answer_text.grid(row=3, column=1, columnspan=2, pady=5)

        # Status
        self.status_label = ttk.Label(frame, text="", foreground="blue")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=(5, 0))
        self.progress_bar = ttk.Progressbar(frame, mode="indeterminate")
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(5, 10))
        self.progress_bar.grid_remove()

        # Buttons
        self.ask_btn = ttk.Button(frame, text="Ask", command=self.ask_question)
        self.ask_btn.grid(row=6, column=1, pady=10, sticky="e", ipadx=10)
        ttk.Button(frame, text="Exit", command=self.root.quit).grid(row=6, column=2, pady=10, sticky="w", ipadx=10)

    def _on_api_key_change(self, event=None):
        new_key = self.api_entry.get().strip()
        if new_key != self.config.api_key:
            self.backend_factory.update_api_key(new_key)

    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.selected_file = Path(filename)
            self.file_label.config(text=self.selected_file.name)

    def ask_question(self):
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a PDF file.")
            return

        question = self.question_entry.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question.")
            return

        backend, fallback_msg = self.backend_factory.get_current_backend_with_fallback()
        if not backend:
            messagebox.showerror("No Backend", "No AI backend is available. Check your API key or Ollama service.")
            return

        if fallback_msg:
            self._show_status(fallback_msg, False)

        self.ask_btn.config(state="disabled", text="Processing...")
        self._show_status("Processing...", True)

        def _worker():
            try:
                text = read_data(self.selected_file)
                answer = backend.process_question(text, question)
                self.root.after(0, lambda: self._on_done(question, answer, backend.get_backend_name()))
            except ServiceUnavailableError as e:
                self.root.after(0, lambda: self._on_error("Service Unavailable", e.details.get("message", str(e))))
            except ModelNotFoundError as e:
                msg = (f"{e.message}\n\nTo install: {e.installation_cmd}" if e.installation_cmd
                       else ErrorMessageGenerator.get_ollama_model_missing_message(e.model_name))
                self.root.after(0, lambda: self._on_error("Model Not Found", msg))
            except InvalidModelError as e:
                msg = f"{e.message}\n\nSuggested alternatives:\n"
                for m in e.suggested_models[:3]:
                    msg += f"  ollama pull {m}\n"
                self.root.after(0, lambda: self._on_error("Invalid Model", msg))
            except AuthenticationError as e:
                self.root.after(0, lambda: self._on_error(
                    "Authentication Error", e.details.get("message", ErrorMessageGenerator.get_gemini_auth_error_message())))
            except ProcessingTimeoutError as e:
                msg = ErrorMessageGenerator.get_timeout_error_message(backend.get_backend_name(), e.timeout_seconds)
                self.root.after(0, lambda: self._on_error("Timeout Error", msg))
            except NetworkError as e:
                msg = ErrorMessageGenerator.get_network_error_message(e.service_name)
                self.root.after(0, lambda: self._on_error("Network Error", msg))
            except AIBackendError as e:
                self.root.after(0, lambda: self._on_error("AI Error", f"{e.message}\n\nError Code: {e.error_code}"))
            except Exception as e:
                self.root.after(0, lambda: self._on_error("Error", f"Q&A failed: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, question: str, answer: str, backend_name: str):
        self._hide_status()
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, answer)
        self.ask_btn.config(state="normal", text="Ask")
        self._log(question, answer, backend_name)

    def _on_error(self, title: str, detail: str):
        self._hide_status()
        self.ask_btn.config(state="normal", text="Ask")
        messagebox.showerror(title, detail)

    def _show_status(self, message: str, show_progress: bool):
        self.status_label.config(text=message, foreground="blue")
        if show_progress:
            self.progress_bar.grid()
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
        self.root.update()

    def _hide_status(self):
        self.status_label.config(text="")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()

    def _log(self, question: str, answer: str, backend_name: str):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] ({backend_name})\nQuestion: {question}\nAnswer: {answer}\n{'-'*40}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = PDFQAApp(root)
    root.mainloop()
