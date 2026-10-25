import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from app.config import config
from app.models import StyleProfile, PrivacyFilterConfig, OllamaConfig


class Storage:
    """Local SQLite repository for profiles, configuration, and data management."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Table for style profiles
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS style_profiles (
                    id TEXT PRIMARY KEY,
                    profile_name TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                )
                """
            )
            # 2. Table for application settings
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            # 3. Table for optional imported messages (only if user explicitly opts in)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS imported_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp_str TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_profile(self, profile: StyleProfile):
        """Saves or updates a style profile in SQLite."""
        data_json = profile.model_dump_json()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO style_profiles (id, profile_name, sender_name, created_at, data_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (profile.id, profile.profile_name, profile.sender_name, profile.created_at, data_json),
            )
            conn.commit()

    def get_profile(self, profile_id: str = "default") -> Optional[StyleProfile]:
        """Retrieves a style profile by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data_json FROM style_profiles WHERE id = ?", (profile_id,)
            )
            row = cursor.fetchone()
            if row:
                return StyleProfile.model_validate_json(row["data_json"])
        return None

    def delete_profile(self, profile_id: str = "default") -> bool:
        """Deletes a style profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM style_profiles WHERE id = ?", (profile_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves an app setting string."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return row["value"]
        return default

    def set_setting(self, key: str, value: str):
        """Sets an app setting string."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def get_ollama_config(self) -> OllamaConfig:
        """Loads configured Ollama settings or defaults."""
        url = self.get_setting("ollama_base_url", config.default_ollama_url) or config.default_ollama_url
        model = self.get_setting("ollama_model", config.default_ollama_model) or config.default_ollama_model
        timeout_str = self.get_setting("ollama_timeout", str(config.default_timeout_seconds))
        timeout = float(timeout_str) if timeout_str else config.default_timeout_seconds
        return OllamaConfig(base_url=url, model=model, timeout_seconds=timeout)

    def save_ollama_config(self, ollama_cfg: OllamaConfig):
        """Saves Ollama settings."""
        self.set_setting("ollama_base_url", ollama_cfg.base_url)
        self.set_setting("ollama_model", ollama_cfg.model)
        self.set_setting("ollama_timeout", str(ollama_cfg.timeout_seconds))

    def get_privacy_config(self) -> PrivacyFilterConfig:
        """Loads configured privacy settings."""
        raw = self.get_setting("privacy_config")
        if raw:
            try:
                return PrivacyFilterConfig.model_validate_json(raw)
            except Exception:
                pass
        return PrivacyFilterConfig()

    def save_privacy_config(self, priv_cfg: PrivacyFilterConfig):
        """Saves privacy settings."""
        self.set_setting("privacy_config", priv_cfg.model_dump_json())

    def save_messages_opt_in(self, messages: List[Dict[str, Any]]):
        """Stores messages only when user explicitly requested persistence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for msg in messages:
                cursor.execute(
                    "INSERT INTO imported_messages (sender, content, timestamp_str) VALUES (?, ?, ?)",
                    (msg.get("sender", ""), msg.get("content", ""), msg.get("timestamp_str")),
                )
            conn.commit()

    def clear_imported_messages(self):
        """Clears all stored imported messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM imported_messages")
            conn.commit()

    def purge_all_data(self):
        """
        Permanently purges all profiles, settings, and stored messages.
        Resets SQLite database completely.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM style_profiles")
            cursor.execute("DELETE FROM app_settings")
            cursor.execute("DELETE FROM imported_messages")
            conn.commit()


# Default singleton instance
storage = Storage()
