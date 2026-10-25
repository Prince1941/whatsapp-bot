import pytest
import io
import httpx
from unittest.mock import patch, AsyncMock
from app.web.main import app
from app.models import StyleProfile, LengthStats, CapitalizationStyle, EmojiStats


@pytest.mark.asyncio
async def test_get_root_page(temp_db):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        assert "WhatsApp Style AI Assistant" in response.text
        assert "Draft Replies" in response.text


@pytest.mark.asyncio
async def test_preview_chat_export_endpoint(temp_db, sample_android_chat):
    transport = httpx.ASGITransport(app=app)
    files = {"file": ("chat.txt", sample_android_chat.encode("utf-8"), "text/plain")}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/chat/preview", files=files)
        assert response.status_code == 200
        assert "Chat Export Analysis: chat.txt" in response.text
        assert "Alex" in response.text
        assert "Me" in response.text


@pytest.mark.asyncio
async def test_build_profile_endpoint(temp_db, sample_android_chat):
    transport = httpx.ASGITransport(app=app)

    # 1. Preview to initiate upload session
    files = {"file": ("chat.txt", sample_android_chat.encode("utf-8"), "text/plain")}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        preview_resp = await ac.post("/api/chat/preview", files=files)
        assert preview_resp.status_code == 200

        # Extract session_id from response
        import re
        match = re.search(r'name="session_id" value="([^"]+)"', preview_resp.text)
        assert match is not None
        session_id = match.group(1)

        # 2. Build profile from sender "Me"
        build_data = {
            "session_id": session_id,
            "sender_name": "Me",
            "profile_name": "Test Alex Chat",
            "redact_phones": "true",
            "redact_emails": "true",
            "redact_urls": "true",
            "redact_financial": "true",
            "filter_action": "redact",
            "custom_terms": "",
            "save_raw": "false",
        }
        build_resp = await ac.post("/api/profile/build", data=build_data)
        assert build_resp.status_code == 200
        assert "Successfully built style profile" in build_resp.text
        assert "Test Alex Chat" in build_resp.text


@pytest.mark.asyncio
async def test_draft_replies_endpoint_with_mocked_ollama(temp_db):
    transport = httpx.ASGITransport(app=app)

    # Save a profile first
    temp_db.save_profile(
        StyleProfile(
            profile_name="Casual Style",
            sender_name="Me",
            capitalization=CapitalizationStyle(primary_case="all_lowercase"),
            emojis=EmojiStats(top_emojis=["👍"]),
        )
    )

    mock_ollama_output = """
=== DRAFT 1 ===
sounds good, see you at 7pm 👍

=== DRAFT 2 ===
yes definitely! lets meet at 7pm
"""

    with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_ollama_output

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            data = {
                "context": "Alex: Are we meeting at 7pm?",
                "intent": "Confirm 7pm",
                "length_preference": "short",
                "style_intensity": "moderate",
                "num_drafts": 2,
            }
            resp = await ac.post("/api/draft", data=data)
            assert resp.status_code == 200
            assert "AI-GENERATED DRAFT — REVIEW BEFORE USE" in resp.text
            assert "sounds good, see you at 7pm 👍" in resp.text
            assert "Option 1" in resp.text
            assert "Option 2" in resp.text


@pytest.mark.asyncio
async def test_purge_all_data_endpoint(temp_db):
    temp_db.save_profile(StyleProfile(profile_name="To Purge", sender_name="Me"))
    assert temp_db.get_profile() is not None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/purge")
        assert resp.status_code == 200
        assert "All Data Purged" in resp.text

    assert temp_db.get_profile() is None
