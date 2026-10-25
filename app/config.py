import os
from pathlib import Path
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application-wide settings and limits."""
    app_name: str = "WhatsApp Style Assistant"
    data_dir: Path = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
    db_file: str = "assistant.db"
    default_ollama_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    default_ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB limit for text exports
    max_context_chars: int = 15000
    max_intent_chars: int = 2000
    default_timeout_seconds: float = 60.0

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / self.db_file


config = AppConfig()
