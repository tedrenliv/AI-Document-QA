"""
Backend configuration management for AI Q&A application.
Handles saving and loading backend preferences with backward compatibility.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BackendConfig:
    """Configuration for AI backend selection and settings."""
    backend_type: str = "gemini"  # "gemini" or "ollama"
    api_key: str = ""  # Only used for Gemini backend
    
    CONFIG_FILE = "config.txt"
    
    def save_to_config(self) -> None:
        """Save configuration to config.txt file."""
        config_path = Path(self.CONFIG_FILE)
        
        # Create config content in key=value format
        config_lines = []
        if self.api_key:
            config_lines.append(f"api_key={self.api_key}")
        config_lines.append(f"backend={self.backend_type}")
        
        config_content = "\n".join(config_lines)
        config_path.write_text(config_content, encoding="utf-8")
    
    @classmethod
    def load_from_config(cls) -> 'BackendConfig':
        """Load configuration from config.txt file with backward compatibility."""
        config_path = Path(cls.CONFIG_FILE)
        
        if not config_path.exists():
            # No config file exists, return default configuration
            return cls()
        
        try:
            content = config_path.read_text(encoding="utf-8").strip()
            
            if not content:
                # Empty config file, return default
                return cls()
            
            # Check if this is the old format (single line API key)
            if "\n" not in content and "=" not in content:
                # Old format: single line with just the API key
                return cls(backend_type="gemini", api_key=content)
            
            # New format: key=value pairs
            config = cls()
            
            for line in content.split("\n"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "api_key":
                    config.api_key = value
                elif key == "backend":
                    if value in ["gemini", "ollama"]:
                        config.backend_type = value
            
            return config
            
        except Exception as e:
            # If there's any error reading the config, return default
            print(f"Warning: Error reading config file: {e}")
            return cls()
    
    def is_gemini_configured(self) -> bool:
        """Check if Gemini backend is properly configured."""
        return self.backend_type == "gemini" and bool(self.api_key.strip())
    
    def is_ollama_configured(self) -> bool:
        """Check if Ollama backend is configured."""
        return self.backend_type == "ollama"
    
    def get_display_name(self) -> str:
        """Get user-friendly display name for the current backend."""
        if self.backend_type == "gemini":
            return "Google Gemini"
        elif self.backend_type == "ollama":
            return "Local Ollama"
        else:
            return "Unknown Backend"