import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.web.main import app
from app.models import StyleProfile, DraftRequest
from app.prompt import build_system_prompt, build_user_prompt, parse_draft_options


@pytest.mark.asyncio
async def test_kaise_ho_bhai_produces_answer_not_repetition(temp_db):
    """Verify that 'kaise ho bhai?' generates answers rather than repeating or echoing the question."""
    transport = httpx.ASGITransport(app=app)

    # Mock Ollama output with proper answers
    mock_ollama_output = """
=== DRAFT 1 ===
thik hu, tu bata

=== DRAFT 2 ===
mast hu bhai, tu suna
"""

    with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_ollama_output

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            data = {
                "context": "kaisa hai bhai?",
                "intent": "",  # Blank intent
                "length_preference": "short",
                "style_intensity": "moderate",
                "num_drafts": 2,
            }
            resp = await ac.post("/api/draft", data=data)
            assert resp.status_code == 200

            # Verify prompt sent to Ollama has Priority 1 response instructions and few-shot examples
            call_args = mock_gen.call_args
            sys_prompt = call_args.kwargs.get("system_prompt") or call_args.args[0]
            usr_prompt = call_args.kwargs.get("user_prompt") or call_args.args[1]

            assert "ANSWER DIRECTLY" in sys_prompt
            assert "NEVER ECHO OR REPEAT" in sys_prompt
            assert "LATEST INCOMING MESSAGE (from other person to me):" in usr_prompt
            assert '"kaisa hai bhai?"' in usr_prompt

            # Verify output displays answers, not echoes
            assert "thik hu, tu bata" in resp.text
            assert "mast hu bhai, tu suna" in resp.text


@pytest.mark.asyncio
async def test_blank_intent_still_produces_direct_reply(temp_db):
    """Verify that leaving the intent field blank still directs the model to answer directly."""
    transport = httpx.ASGITransport(app=app)

    mock_ollama_output = """
=== DRAFT 1 ===
all good here, what about you?
"""

    with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_ollama_output

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            data = {
                "context": "how's everything going?",
                "intent": "",  # Blank intent
                "length_preference": "short",
                "style_intensity": "moderate",
                "num_drafts": 1,
            }
            resp = await ac.post("/api/draft", data=data)
            assert resp.status_code == 200

            usr_prompt = mock_gen.call_args.kwargs.get("user_prompt") or mock_gen.call_args.args[1]
            assert "None provided — this field is optional. Infer a direct, natural answer" in usr_prompt
            assert "all good here, what about you?" in resp.text


@pytest.mark.asyncio
async def test_unknown_facts_not_invented(temp_db):
    """Verify that when context asks about unknown plans with no intent, the system enforces safe noncommittal answers."""
    req = DraftRequest(
        context="kal mil raha hai?",
        intent="",  # No availability information given
        length_preference="short",
        num_drafts=2,
    )
    profile = StyleProfile(profile_name="Test")
    sys_prompt = build_system_prompt(profile, "moderate")
    usr_prompt = build_user_prompt(req)

    # Verify anti-hallucination instruction is strictly enforced in prompts
    assert "Do NOT invent specific plans, commitments, or unknown facts" in usr_prompt
    assert "NEVER INVENT FACTS" in sys_prompt
    assert "kal ka pakka nahi hai abhi" in sys_prompt


def test_reply_options_distinct_and_do_not_echo():
    """Verify parse_draft_options eliminates echo options and produces distinct answers."""
    incoming = "kaisa hai bhai?"

    # Model returned an echo as draft 1, and an actual answer as draft 2
    raw = """
=== DRAFT 1 ===
kaisa hai bhai? aur kya hua?

=== DRAFT 2 ===
thik hu, tu bata
"""
    drafts = parse_draft_options(raw, expected_count=2, incoming_text=incoming)
    # Echo must be purged
    assert "kaisa hai bhai? aur kya hua?" not in drafts
    assert "thik hu, tu bata" in drafts


def test_reply_options_deduplicated():
    """Verify duplicate options are pruned."""
    raw = """
=== DRAFT 1 ===
thik hu, tu bata

=== DRAFT 2 ===
thik hu, tu bata

=== DRAFT 3 ===
mast hu bhai
"""
    drafts = parse_draft_options(raw, expected_count=3, incoming_text="kaisa hai bhai?")
    assert len(drafts) == 2
    assert drafts[0] == "thik hu, tu bata"
    assert drafts[1] == "mast hu bhai"
