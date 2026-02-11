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

# ---------- Main App ----------
class QAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Q&A with Chroma DB (TXT, DOC & PDF)")
        self.selected_file = None
        
        # Load configuration (includes API key and backend preference)
        self.config = BackendConfig.load_from_config()
        
        # Initialize backend factory
        self.backend_factory = AIBackendFactory(self.config)
        
        self.build_gui()
        self.update_backend_selection()
        
        # Schedule periodic status updates
        self.schedule_status_update()

    def build_gui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        # Backend Selection
        ttk.Label(frame, text="AI Backend:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        backend_frame = ttk.Frame(frame)
        backend_frame.grid(row=0, column=1, columnspan=2, sticky="w", pady=(0, 5))
        
        self.backend_var = tk.StringVar(value=self.config.backend_type)
        
        self.gemini_radio = ttk.Radiobutton(
            backend_frame, 
            text="Google Gemini", 
            variable=self.backend_var, 
            value="gemini",
            command=self.on_backend_change
        )
        self.gemini_radio.pack(side="left", padx=(0, 20))
        
        self.ollama_radio = ttk.Radiobutton(
            backend_frame, 
            text="Local Ollama", 
            variable=self.backend_var, 
            value="ollama",
            command=self.on_backend_change
        )
        self.ollama_radio.pack(side="left")
        
        # Backend status indicators
        self.status_frame = ttk.Frame(backend_frame)
        self.status_frame.pack(side="left", padx=(20, 0))
        
        self.gemini_status = ttk.Label(self.status_frame, text="", foreground="gray")
        self.gemini_status.pack(side="left", padx=(0, 10))
        
        self.ollama_status = ttk.Label(self.status_frame, text="", foreground="gray")
        self.ollama_status.pack(side="left")

        # API Key
        self.api_label = ttk.Label(frame, text="Google API Key:", font=("Arial", 11, "bold"))
        self.api_label.grid(row=1, column=0, sticky="w", pady=(10, 5))
        
        self.api_entry = ttk.Entry(frame, width=60)
        self.api_entry.insert(0, self.config.api_key)
        self.api_entry.grid(row=1, column=1, padx=10, pady=(10, 5))
        self.api_entry.bind('<KeyRelease>', self.on_api_key_change)

        # File selection
        ttk.Label(frame, text="Select .txt or .pdf file:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", pady=10)
        self.file_label = ttk.Label(frame, text="No file selected", foreground="blue")
        self.file_label.grid(row=2, column=1, sticky="w")

        self.select_btn = ttk.Button(frame, text="📂 Select File", command=self.select_file)
        self.select_btn.grid(row=2, column=2, padx=10, pady=10, ipadx=10, ipady=5)

        # Question
        ttk.Label(frame, text="Your Question:", font=("Arial", 11)).grid(row=3, column=0, sticky="w", pady=10)
        self.question_entry = ttk.Entry(frame, width=80)
        self.question_entry.grid(row=3, column=1, columnspan=2, pady=5)

        # Answer display
        ttk.Label(frame, text="Answer:", font=("Arial", 11)).grid(row=4, column=0, sticky="nw", pady=10)
        self.answer_text = tk.Text(frame, height=15, width=90, wrap="word")
        self.answer_text.grid(row=4, column=1, columnspan=2, pady=5)

        # Processing status
        self.status_label = ttk.Label(frame, text="", foreground="blue")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=(5, 0))
        
        # Progress bar for processing feedback
        self.progress_bar = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(5, 10))
        self.progress_bar.grid_remove()  # Hide initially

        # Buttons
        self.ask_btn = ttk.Button(frame, text="🔎 Ask", command=self.ask_question)
        self.ask_btn.grid(row=7, column=1, pady=10, sticky="e", ipadx=10)

        self.exit_btn = ttk.Button(frame, text="❌ Exit", command=self.root.quit)
        self.exit_btn.grid(row=7, column=2, pady=10, sticky="w", ipadx=10)

    def on_backend_change(self):
        """Handle backend selection change."""
        new_backend_type = self.backend_var.get()
        if self.backend_factory.switch_backend(new_backend_type):
            self.update_backend_selection()
        else:
            # Revert to previous selection if switch failed
            self.backend_var.set(self.config.backend_type)
    
    def on_api_key_change(self, event=None):
        """Handle API key field changes."""
        new_api_key = self.api_entry.get().strip()
        if new_api_key != self.config.api_key:
            # Update API key through factory
            self.backend_factory.update_api_key(new_api_key)
            self.update_backend_status()
    
    def update_backend_selection(self):
        """Update UI based on selected backend."""
        selected_backend = self.backend_var.get()
        
        # Enable/disable API key field based on backend selection
        if selected_backend == "gemini":
            self.api_entry.config(state="normal")
            self.api_label.config(foreground="black")
        else:  # ollama
            self.api_entry.config(state="disabled")
            self.api_label.config(foreground="gray")
        
        self.update_backend_status()
    
    def update_backend_status(self):
        """Update backend availability status indicators."""
        # Get status for Gemini backend
        gemini_status = self.backend_factory.get_backend_status("gemini")
        if gemini_status["available"]:
            self.gemini_status.config(text="✓ Ready", foreground="green")
        elif gemini_status["status"] == "api_key_required":
            self.gemini_status.config(text="⚠ API Key Required", foreground="orange")
        else:
            self.gemini_status.config(text="✗ Not Available", foreground="red")

        ollama_status = self.backend_factory.get_backend_status("ollama")
        if ollama_status["available"]:
            self.ollama_status.config(text="✓ Ready", foreground="green")
        else:
            self.ollama_status.config(text="✗ Not Available", foreground="red")

    def select_file(self):
        filetypes = [("Text and Document files", "*.txt *.pdf *.docx"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select data file", filetypes=filetypes)
        if filename:
            self.selected_file = Path(filename)
            self.file_label.config(text=self.selected_file.name)

    def ask_question(self):
        # Validate inputs
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a .txt .DOC or .pdf file.")
            return
        
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question.")
            return
        
        # Try to get current backend with fallback
        current_backend, fallback_message = self.backend_factory.get_current_backend_with_fallback()
        
        # Check if any backend is available
        if not current_backend:
            # No backends available - show comprehensive error
            self._show_no_backends_error()
            return
        
        # If we're using a fallback, inform the user
        if fallback_message:
            self.show_processing_status(fallback_message, False)
            # Show the message for 3 seconds before proceeding
            self.root.after(3000, lambda: self._proceed_with_processing(current_backend, question))
        else:
            self._proceed_with_processing(current_backend, question)
    
    def _show_no_backends_error(self):
        """Show comprehensive error when no backends are available."""
        selected_backend_type = self.backend_var.get()
        error_info = self.backend_factory.get_backend_error_info(selected_backend_type)
        installation_help = self.backend_factory.get_installation_help(selected_backend_type)
        
        if selected_backend_type == "gemini":
            messagebox.showerror("Google Gemini Not Available", 
                ErrorMessageGenerator.get_gemini_auth_error_message())
        else:  # ollama
            if "embedding" in str(error_info.get("embedding_models_found", [])):
                messagebox.showerror("Ollama Model Issue", 
                    ErrorMessageGenerator.get_ollama_model_missing_message("embeddinggemma:latest"))
            else:
                messagebox.showerror("Ollama Not Available", 
                    ErrorMessageGenerator.get_ollama_service_unavailable_message())
    
    def _proceed_with_processing(self, backend: AIBackend, question: str):
        """Start processing in a background thread to keep the UI responsive."""
        self.ask_btn.config(state="disabled", text="Processing...")

        backend_name = backend.get_backend_name()
        if "Ollama" in backend_name:
            self.show_processing_status("Processing with Local Ollama... This may take a moment.", True)
        else:
            self.show_processing_status("Processing with Google Gemini...", True)

        def _worker():
            try:
                text = read_data(self.selected_file)
                self.root.after(0, lambda: self.show_processing_status(
                    "AI is analyzing your question..." +
                    (" Local processing may take longer." if "Ollama" in backend_name else ""),
                    True))

                answer = backend.process_question(text, question)
                self.root.after(0, lambda: self._on_processing_done(question, answer, backend_name))

            except ServiceUnavailableError as e:
                msg = e.details.get("message", str(e))
                self.root.after(0, lambda: self._on_processing_error(
                    "Service unavailable", "Service Unavailable", msg))
            except ModelNotFoundError as e:
                msg = (f"{e.message}\n\nTo install: {e.installation_cmd}" if e.installation_cmd
                       else ErrorMessageGenerator.get_ollama_model_missing_message(e.model_name))
                self.root.after(0, lambda: self._on_processing_error("Model not found", "Model Not Found", msg))
            except InvalidModelError as e:
                msg = f"{e.message}\n\nSuggested alternatives:\n"
                for model in e.suggested_models[:3]:
                    msg += f"• ollama pull {model}\n"
                self.root.after(0, lambda: self._on_processing_error("Invalid model", "Invalid Model", msg))
            except AuthenticationError as e:
                msg = e.details.get("message", ErrorMessageGenerator.get_gemini_auth_error_message())
                self.root.after(0, lambda: self._on_processing_error(
                    "Authentication failed", "Authentication Error", msg))
            except ProcessingTimeoutError as e:
                msg = ErrorMessageGenerator.get_timeout_error_message(backend_name, e.timeout_seconds)
                self.root.after(0, lambda: self._on_processing_error("Processing timed out", "Timeout Error", msg))
            except NetworkError as e:
                msg = ErrorMessageGenerator.get_network_error_message(e.service_name)
                self.root.after(0, lambda: self._on_processing_error("Network error", "Network Error", msg))
            except AIBackendError as e:
                msg = f"{e.message}\n\nError Code: {e.error_code}"
                self.root.after(0, lambda: self._on_processing_error(
                    "AI processing error", "AI Error", msg))
            except Exception as e:
                msg = f"An unexpected error occurred: {e}\n\nPlease try again or switch to a different backend."
                self.root.after(0, lambda: self._on_processing_error(
                    "Unexpected error occurred", "Unexpected Error", msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_processing_done(self, question: str, answer: str, backend_name: str):
        """Handle successful processing result on the main thread."""
        self.show_processing_status("Processing complete!", False)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, answer)
        self.log_answer(question, answer, backend_name)
        self.ask_btn.config(state="normal", text="🔎 Ask")
        self.root.after(1000, self.hide_processing_status)

    def _on_processing_error(self, status_msg: str, title: str, detail: str):
        """Handle processing error on the main thread."""
        self.show_error_status(status_msg)
        self.ask_btn.config(state="normal", text="🔎 Ask")
        messagebox.showerror("Q&A Failed", f"[{title}]\n\n{detail}")

    def schedule_status_update(self):
        """Schedule periodic backend status updates in a background thread."""
        def _check():
            gemini_status = self.backend_factory.get_backend_status("gemini")
            ollama_status = self.backend_factory.get_backend_status("ollama")
            self.root.after(0, lambda: self._apply_status(gemini_status, ollama_status))
        threading.Thread(target=_check, daemon=True).start()
        self.root.after(10000, self.schedule_status_update)

    def _apply_status(self, gemini_status, ollama_status):
        """Apply backend status to UI (must run on main thread)."""
        if gemini_status["available"]:
            self.gemini_status.config(text="✓ Ready", foreground="green")
        elif gemini_status.get("status") == "api_key_required":
            self.gemini_status.config(text="⚠ API Key Required", foreground="orange")
        else:
            self.gemini_status.config(text="✗ Not Available", foreground="red")

        if ollama_status["available"]:
            self.ollama_status.config(text="✓ Ready", foreground="green")
        else:
            self.ollama_status.config(text="✗ Not Available", foreground="red")
    
    def show_processing_status(self, message: str, show_progress: bool = True):
        """
        Show processing status with optional progress indicator.
        
        Args:
            message (str): Status message to display
            show_progress (bool): Whether to show the progress bar
        """
        self.status_label.config(text=message, foreground="blue")
        
        if show_progress:
            self.progress_bar.grid()
            self.progress_bar.start(10)  # Start animation with 10ms interval
        else:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
        
        self.root.update()
    
    def hide_processing_status(self):
        """Hide processing status and progress indicator."""
        self.status_label.config(text="")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.root.update()
    
    def show_error_status(self, message: str):
        """
        Show error status message.
        
        Args:
            message (str): Error message to display
        """
        self.status_label.config(text=message, foreground="red")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.root.update()
        
        # Clear error message after 5 seconds
        self.root.after(5000, self.hide_processing_status)

    def log_answer(self, question, answer, backend_name=None):
        """
        Log the question and answer with backend information.
        
        Args:
            question (str): The user's question
            answer (str): The AI's response
            backend_name (str, optional): Name of the backend used for processing
        """
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            # Format backend information consistently
            if backend_name:
                # Normalize backend name for consistent logging
                if "Gemini" in backend_name:
                    backend_marker = " (Google Gemini)"
                elif "Ollama" in backend_name:
                    backend_marker = " (Local Ollama)"
                else:
                    # Fallback for any other backend names
                    backend_marker = f" ({backend_name})"
            else:
                # Maintain backward compatibility - no backend info if not provided
                backend_marker = ""
            
            # Write log entry with consistent format
            log.write(f"\n[{datetime.now()}]{backend_marker}\nQuestion: {question}\nAnswer: {answer}\n{'-'*40}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = QAApp(root)
    root.mainloop()
