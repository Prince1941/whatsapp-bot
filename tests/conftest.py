import pytest
from pathlib import Path
import tempfile
from app.config import config
from app.storage import Storage


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Provides a fresh, temporary SQLite database for test isolation."""
    db_file = tmp_path / "test_assistant.db"
    test_storage = Storage(db_path=db_file)
    monkeypatch.setattr("app.storage.storage", test_storage)
    monkeypatch.setattr("app.web.main.storage", test_storage)
    return test_storage


@pytest.fixture
def sample_android_chat():
    return """15/01/24, 10:14 am - Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
15/01/24, 10:15 am - Alex: Hey are you around?
15/01/24, 10:16 am - Me: yeah just woke up lol
whats the plan for today?
15/01/24, 10:17 am - Alex: Wanna grab lunch?
15/01/24, 10:18 am - Me: <Media omitted>
15/01/24, 10:19 am - Me: yeah sounds good bro! see you at 1pm 👍
15/01/24, 10:20 am - Alex: Perfect!
"""


@pytest.fixture
def sample_ios_chat():
    return """[15/01/2024, 10:14:22] Messages and calls are end-to-end encrypted.
[15/01/2024, 10:15:01] Sarah: Are you coming to the party?
[15/01/2024, 10:16:30] Me: hey Sarah! yes absolutely 🎉
can i bring a friend?
[15/01/2024, 10:17:05] Sarah: Of course!
[15/01/2024, 10:18:00] Me: awesome thanks! cya soon
"""
