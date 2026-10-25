import os
from pathlib import Path
from pydantic import BaseModel, Field


# Auto-load local .env file if present
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            _v = _v.strip().strip("'\"")
            if _k and _k not in os.environ:
                os.environ[_k] = _v


class AppConfig(BaseModel):
    """Application-wide settings and limits."""
    app_name: str = "WhatsApp Style Assistant"
    data_dir: Path = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
    db_file: str = "assistant.db"
    default_ollama_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    default_ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB limit for text exports
    max_context_chars: int = 15000
    max_intent_chars: int = 2000
    default_timeout_seconds: float = 120.0
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / self.db_file


config = AppConfig()
