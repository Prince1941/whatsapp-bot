import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.web.main import app
from app.models import (
    StyleProfile,
    DraftRequest,
    ChatMessage,
    CapitalizationStyle,
    EmojiStats,
    PunctuationStats,
    PrivacyFilterConfig,
)
from app.parser import parse_whatsapp_export
from app.privacy import PrivacyFilter
from app.analyzer import (
    build_style_profile,
    extract_spelling_preferences,
    extract_frequent_short_phrases,
    compute_confidence,
)
from app.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_draft_options,
    retrieve_relevant_phrases,
    apply_spelling_preferences,
)
from app.ollama import OllamaConnectionError


# ============================================================================
# CRITERION 1: Multi-sender import fixture — only selected user's messages contribute
# ============================================================================
def test_multisender_import_only_selected_user_contributes_to_style():
    """
    Given a multi-sender chat export where Participant A uses standard Romanized Hindi
    ('theek hoon bhai', 'main kal aaunga') and Participant B ('Prince Kr') uses short Hinglish
    ('thik hu bhai', 'chal thik h', 'nhi'), verify only Prince Kr's messages contribute
    to the generated style profile.
    """
    raw_chat = """15/01/24, 10:14 am - Messages and calls are end-to-end encrypted.
15/01/24, 10:15 am - Rohit AI: theek hoon bhai, aap kaise ho?
15/01/24, 10:16 am - Prince Kr: thik hu bhai
15/01/24, 10:17 am - Rohit AI: main kal subah aaunga office
15/01/24, 10:18 am - Prince Kr: chal thik h
15/01/24, 10:19 am - Rohit AI: kuch kaam hai kya abhi?
15/01/24, 10:20 am - Prince Kr: nhi abhi to nhi h
15/01/24, 10:21 am - Rohit AI: theek hai
15/01/24, 10:22 am - Prince Kr: ha
"""
    parse_res = parse_whatsapp_export(raw_chat)
    privacy = PrivacyFilter(PrivacyFilterConfig())

    # Filter messages specifically for 'Prince Kr'
    prince_msgs = privacy.filter_messages(parse_res.messages, target_sender="Prince Kr")

    # Build profile explicitly for 'Prince Kr'
    profile_prince = build_style_profile(
        messages=prince_msgs,
        sender_name="Prince Kr",
        profile_name="Prince Style",
    )

    # 1. Total analyzed messages must match ONLY Prince Kr's count (4 messages)
    assert profile_prince.total_training_messages == 4
    assert profile_prince.sender_name == "Prince Kr"

    # 2. Rohit's phrases/words must NOT appear in Prince Kr's snippets or phrases
    for snippet in profile_prince.sample_snippets:
        assert "Rohit" not in snippet
        assert "aaunga office" not in snippet
        assert "kuch kaam" not in snippet

    # 3. Spelling preferences must reflect Prince Kr's spellings (thik, hu, nhi, h)
    assert profile_prince.spelling_preferences.get("theek") == "thik"
    assert profile_prince.spelling_preferences.get("hoon") == "hu"
    assert profile_prince.spelling_preferences.get("nahi") == "nhi"
    assert profile_prince.spelling_preferences.get("hai") == "h"

    # 4. Rohit's spellings ('theek', 'hoon', 'main') must not be Prince Kr's preferences
    assert profile_prince.spelling_preferences.get("thik") is None
    assert profile_prince.spelling_preferences.get("hu") is None


# ============================================================================
# CRITERION 2: Preserve spelling variants from raw messages without pre-normalization
# ============================================================================
def test_preserve_spelling_variants_from_selected_user_raw_messages():
    """
    Ensure the analyzer preserves raw Romanized Hinglish variants (e.g. 'thik', 'h',
    'hu', 'nhi', 'kl', 'kr', 'bhai') and does not normalize them to dictionary Hindi or English.
    """
    raw_texts = [
        "thik h bhai",
        "kl batata hu",
        "nhi abhi nhi",
        "ha kr diya",
        "chal thik h",
        "bhej diye",
    ]
    msgs = [ChatMessage(content=t, sender="User") for t in raw_texts]

    # Check spelling preferences extraction
    prefs = extract_spelling_preferences(msgs)
    assert prefs.get("theek") == "thik"
    assert prefs.get("hai") == "h"
    assert prefs.get("hoon") == "hu"
    assert prefs.get("nahi") == "nhi"
    assert prefs.get("kal") == "kl"
    assert prefs.get("kar") == "kr"

    # Check that short phrases preserve raw text and casing
    phrases = extract_frequent_short_phrases(msgs)
    assert any("thik" in p for p in phrases)
    assert any("kl" in p for p in phrases)
    # Ensure they were not normalized to 'theek' or 'kal'
    assert not any("theek" in p for p in phrases)
    assert not any("kal" in p for p in phrases)


# ============================================================================
# CRITERION 3: History spelling preferences consistently preferred in profile & prompt
# ============================================================================
def test_repeated_spelling_preference_consistently_preferred_in_profile_and_prompt():
    """
    When history repeatedly uses one spelling (e.g. 'thik', 'h'), verify:
    1. Profile maps non-preferred variants to the preferred variant.
    2. Prompt includes explicit mandatory spelling rules.
    3. apply_spelling_preferences deterministically enforces the preferred spelling.
    """
    profile = StyleProfile(
        profile_name="Spelling Test",
        sender_name="Prince",
        total_training_messages=50,
        confidence_level="high",
        spelling_preferences={"theek": "thik", "hai": "h", "hoon": "hu", "nahi": "nhi"},
    )

    system_prompt = build_system_prompt(profile, intensity="strong")

    # Prompt must contain mandatory spelling section with exact rules
    assert "MANDATORY SPELLING CONVENTIONS FROM USER'S CHAT HISTORY" in system_prompt
    assert "'thik' (never 'theek')" in system_prompt
    assert "'h' (never 'hai')" in system_prompt
    assert "'hu' (never 'hoon')" in system_prompt

    # Deterministic post-processor converts any unlearned variant to user's variant
    assert apply_spelling_preferences("Theek hai bhai, main theek hoon", profile.spelling_preferences) == "Thik h bhai, main thik hu"
    assert apply_spelling_preferences("nahi hua", profile.spelling_preferences) == "nhi hua"


# ============================================================================
# CRITERION 4: For 'kaisa h bhai?', reply uses learned spelling/phrase instead of 'theek hoon'
# ============================================================================
@pytest.mark.asyncio
async def test_kaisa_h_bhai_uses_learned_spelling_and_phrase_not_theek_hoon(temp_db):
    """
    For 'kaisa h bhai?', reply must be short and use learned spelling/phrase
    (e.g., 'thik h', 'thik hu bhai') rather than textbook 'theek hoon'.
    """
    profile = StyleProfile(
        profile_name="Prince Style",
        sender_name="Prince Kr",
        total_training_messages=120,
        confidence_level="high",
        capitalization=CapitalizationStyle(primary_case="all_lowercase"),
        spelling_preferences={"theek": "thik", "hoon": "hu", "hai": "h", "nahi": "nhi"},
        frequent_short_phrases=["thik hu bhai", "thik h", "chal thik h"],
        intent_phrases={"greeting_status": ["thik hu bhai", "thik h"]},
    )
    temp_db.save_profile(profile)

    # 1. Verify phrase retrieval extracts learned greeting status phrase
    retrieved = retrieve_relevant_phrases(profile, "greeting_status")
    assert "thik hu bhai" in retrieved

    # 2. Verify system prompt incorporates these authentic phrases
    sys_prompt = build_system_prompt(profile, intensity="strong", incoming_text="kaisa h bhai?")
    assert "USER'S ACTUAL RECURRING PHRASES FOR THIS INTENT" in sys_prompt
    assert '"thik hu bhai"' in sys_prompt

    # 3. Simulate Ollama generating options with mixed or textbook spellings
    transport = httpx.ASGITransport(app=app)
    mock_ollama_output = """
=== DRAFT 1 ===
Mai theek hoon bhai

=== DRAFT 2 ===
Hai bhai, theek hai
"""
    with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_ollama_output

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            data = {
                "context": "kaisa h bhai?",
                "intent": "",
                "length_preference": "short",
                "style_intensity": "strong",
                "num_drafts": 2,
            }
            resp = await ac.post("/api/draft", data=data)
            assert resp.status_code == 200

            # The output must NOT contain textbook 'theek hoon' or 'theek hai'
            assert "theek hoon" not in resp.text
            assert "theek hai" not in resp.text
            # It must use the learned 'thik hu' and 'thik h'
            assert "thik hu bhai" in resp.text
            assert "thik h" in resp.text


# ============================================================================
# CRITERION 5: Blank intent still works cleanly
# ============================================================================
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


# ============================================================================
# CRITERION 6: No unsupported personal facts are invented
# ============================================================================
def test_unknown_facts_not_invented():
    """Verify that when context asks about unknown plans with no intent, the system enforces safe noncommittal answers."""
    req = DraftRequest(
        context="kal mil raha hai?",
        intent="",  # No availability information given
        length_preference="short",
        num_drafts=2,
    )
    profile = StyleProfile(profile_name="Test")
    sys_prompt = build_system_prompt(profile, "moderate", incoming_text=req.context)
    usr_prompt = build_user_prompt(req)

    # Verify anti-hallucination instruction is strictly enforced in prompts
    assert "Do NOT invent specific plans, commitments, or unknown facts" in usr_prompt
    assert "NEVER INVENT FACTS" in sys_prompt
    assert "kal ka pakka nahi hai abhi" in sys_prompt or "kl ka pakka nahi h" in sys_prompt


# ============================================================================
# CRITERION 7: Empty or insufficient profile reported honestly
# ============================================================================
def test_empty_or_insufficient_profile_reported_honestly():
    """
    Verify that an empty or low-message profile is honestly assessed as 'none' or 'low'
    confidence, with clear explanatory description warning the user.
    """
    # 0 messages -> 'none' confidence
    level_0, desc_0 = compute_confidence(0)
    assert level_0 == "none"
    assert "no messages" in desc_0.lower() or "insufficient" in desc_0.lower()

    # 4 messages -> 'low' confidence (insufficient for exact match, honestly warns user)
    level_4, desc_4 = compute_confidence(4)
    assert level_4 == "low"
    assert "limited" in desc_4.lower() or "differ" in desc_4.lower()

    # 10 messages -> 'low' confidence
    level_10, desc_10 = compute_confidence(10)
    assert level_10 == "low"
    assert "limited" in desc_10.lower() or "few" in desc_10.lower()

    # 60 messages -> 'medium'
    level_60, desc_60 = compute_confidence(60)
    assert level_60 == "medium"

    # 150 messages -> 'high'
    level_150, desc_150 = compute_confidence(150)
    assert level_150 == "high"


# ============================================================================
# CRITERION 8: Multiple options stay consistent with user's learned spellings
# ============================================================================
def test_multiple_options_stay_consistent_with_learned_spellings():
    """
    Verify that parse_draft_options applies the user's learned spellings
    consistently across all generated draft options.
    """
    raw_output = """
=== DRAFT 1 ===
Theek hai bhai, main theek hoon abhi

=== DRAFT 2 ===
Mai theek hoon bhai, aap batao

=== DRAFT 3 ===
Nahi bhai, theek hai
"""
    profile = StyleProfile(
        profile_name="Test",
        sender_name="Me",
        spelling_preferences={
            "theek": "thik",
            "hoon": "hu",
            "hai": "h",
            "nahi": "nhi",
            "main": "mai",
        },
    )

    drafts = parse_draft_options(
        raw_output,
        expected_count=3,
        incoming_text="kaisa h bhai?",
        profile=profile,
    )

    assert len(drafts) == 3
    for draft in drafts:
        # None of the options may use unlearned variants
        assert "theek" not in draft.lower()
        assert "hoon" not in draft.lower()
        assert "nahi" not in draft.lower()
        # All options must use learned variants
        assert any(w in draft.lower() for w in ["thik", "hu", "h", "nhi", "mai"])

    assert drafts[0] == "Thik h bhai, mai thik hu abhi"
    assert drafts[1] == "Mai thik hu bhai, aap batao"
    assert drafts[2] == "Nhi bhai, thik h"


# ============================================================================
# CRITERION 9: Ollama-unavailable and malformed-response cases handled cleanly
# ============================================================================
@pytest.mark.asyncio
async def test_ollama_unavailable_and_malformed_response_handled_cleanly(temp_db):
    """
    Verify clean, non-crashing handling when:
    1. Ollama is offline (OllamaConnectionError) -> Returns 503 with helpful setup instructions.
    2. Ollama returns malformed or empty output -> parse_draft_options returns fallback without crash.
    """
    transport = httpx.ASGITransport(app=app)

    # 1. Offline Ollama
    with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = OllamaConnectionError("Cannot connect to Ollama at http://localhost:11434")

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            data = {"context": "kaisa h bhai?", "intent": ""}
            resp = await ac.post("/api/draft", data=data)
            assert resp.status_code == 503
            assert "Ollama Connection Error" in resp.text
            assert "ollama serve" in resp.text

    # 2. Malformed / empty response from Ollama
    empty_drafts = parse_draft_options("", expected_count=2, incoming_text="kaisa h bhai?")
    assert len(empty_drafts) == 1
    assert "Unable to generate draft" in empty_drafts[0]

    gibberish = "Random conversational chatter without delimiters or answers."
    gibberish_drafts = parse_draft_options(gibberish, expected_count=2, incoming_text="kaisa h bhai?")
    assert len(gibberish_drafts) >= 1
    assert gibberish_drafts[0] == gibberish


# ============================================================================
# DEDUPLICATION & ECHO FILTERING TESTS
# ============================================================================
def test_reply_options_distinct_and_do_not_echo():
    """Verify parse_draft_options eliminates echo options and produces distinct answers."""
    incoming = "kaisa hai bhai?"

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


# ============================================================================
# FOCUSED INTENT & FACTUAL QUESTION TESTS (padh liya, khana khaya, etc.)
# ============================================================================
@pytest.mark.asyncio
async def test_padh_liya_three_intents_produce_concise_styled_answers(temp_db):
    """
    Verify 'padh liya?' with explicit intents produces concise versions of the intent
    using the user's learned spelling habits (ha, nhi, h), without adding unrelated content.
    """
    profile = StyleProfile(
        profile_name="Prince Kr Style",
        sender_name="Prince Kr",
        capitalization=CapitalizationStyle(primary_case="all_lowercase"),
        punctuation=PunctuationStats(period_ending_freq=0.05),
        spelling_preferences={"theek": "thik", "hai": "h", "hoon": "hu", "nahi": "nhi", "haan": "ha", "kal": "kl"},
    )
    temp_db.save_profile(profile)
    transport = httpx.ASGITransport(app=app)

    test_cases = [
        ("haan padh liya", "ha padh liya"),
        ("thoda bahut aur baaki hai", "thoda bahut aur baaki h"),
        ("nahi padha", "nhi padha"),
        ("bas thoda aur baaki h", "bas thoda aur baaki h"),
    ]

    for intent_input, expected_core in test_cases:
        mock_raw = f"""
=== DRAFT 1 ===
{intent_input}

=== DRAFT 2 ===
{intent_input}
"""
        with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_raw

            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                data = {
                    "context": "padh liya?",
                    "intent": intent_input,
                    "length_preference": "short",
                    "style_intensity": "strong",
                    "num_drafts": 2,
                }
                resp = await ac.post("/api/draft", data=data)
                assert resp.status_code == 200

                # Must not contain textbook unlearned spelling
                if "haan" in intent_input:
                    assert "haan" not in resp.text
                if "nahi" in intent_input:
                    assert "nahi" not in resp.text
                if " hai" in intent_input:
                    assert " hai" not in resp.text

                # Must contain the user's concise styled version
                assert expected_core in resp.text


@pytest.mark.asyncio
async def test_blank_intent_unknown_fact_returns_needs_user_input(temp_db):
    """
    When the incoming message asks a factual personal question ('padh liya?')
    and intent is blank with no answer in context, the assistant must return
    a needs-user-input result rather than fabricating an answer.
    """
    profile = StyleProfile(profile_name="Prince Kr Style", sender_name="Prince Kr")
    temp_db.save_profile(profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        data = {
            "context": "padh liya?",
            "intent": "",  # Blank intent
            "length_preference": "short",
            "style_intensity": "moderate",
            "num_drafts": 2,
        }
        resp = await ac.post("/api/draft", data=data)
        assert resp.status_code == 200

        # UI must clearly notify user that a factual answer is needed
        assert "Answer Required in Intent Field" in resp.text
        assert "whether you studied" in resp.text
        assert "Please specify your answer in the Intent field" in resp.text


def test_style_variations_preserve_intent_meaning():
    """
    Verify style variations (casing, spelling normalization, punctuation)
    preserve the intent's meaning and eliminate hallucinated tasks.
    """
    profile = StyleProfile(
        profile_name="Prince Kr Style",
        sender_name="Prince Kr",
        capitalization=CapitalizationStyle(primary_case="all_lowercase"),
        punctuation=PunctuationStats(period_ending_freq=0.05),
        spelling_preferences={"theek": "thik", "hai": "h", "hoon": "hu", "nahi": "nhi", "haan": "ha"},
    )

    # Simulated Ollama output that hallucinated an unrelated instruction
    raw_output = """
=== DRAFT 1 ===
thoda aur bhi padh raha hu.

=== DRAFT 2 ===
baki kaam karo tu padha.
"""
    drafts = parse_draft_options(
        raw_output=raw_output,
        expected_count=2,
        incoming_text="padh liye?",
        profile=profile,
        user_intent="bas thoda aur baaki h",
    )

    # Hallucinated draft 'baki kaam karo tu padha' must be eliminated
    for d in drafts:
        assert "baki kaam karo" not in d
        assert "tu padha" not in d

    # Both drafts must faithfully convey that some studying remains
    assert len(drafts) == 2
    assert "bas thoda aur baaki h" in drafts[0]
    assert any("thoda" in d for d in drafts)


def test_multiple_options_do_not_contradict_one_another():
    """
    Verify that multiple draft options for an intent do not contradict each other
    (e.g., negative intent 'nahi padha' must not produce affirmative drafts).
    """
    profile = StyleProfile(
        profile_name="Prince Kr Style",
        sender_name="Prince Kr",
        spelling_preferences={"theek": "thik", "hai": "h", "hoon": "hu", "nahi": "nhi", "haan": "ha"},
    )

    # Model returned 1 negative and 1 affirmative draft
    raw_output = """
=== DRAFT 1 ===
nhi padha abhi

=== DRAFT 2 ===
ha padh liya maine
"""
    drafts = parse_draft_options(
        raw_output=raw_output,
        expected_count=2,
        incoming_text="padh liya?",
        profile=profile,
        user_intent="nahi padha",
    )

    assert len(drafts) == 2
    for d in drafts:
        # None of the options may say yes when intent is no
        assert "ha padh liya" not in d
        assert "nhi" in d or "nahi" in d


@pytest.mark.asyncio
async def test_intent_reaches_ollama_prompt_correctly(temp_db):
    """
    Verify that the intent field is correctly forwarded to the Ollama prompt,
    not truncated, and explicitly labeled as the factual content to convey.
    """
    profile = StyleProfile(profile_name="Prince Kr Style", sender_name="Prince Kr")
    temp_db.save_profile(profile)
    transport = httpx.ASGITransport(app=app)

    with patch("app.web.main.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "=== DRAFT 1 ===\nbas thoda aur baaki h\n=== DRAFT 2 ===\nthoda baaki h abhi"

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            data = {
                "context": "padh liye?",
                "intent": "bas thoda aur baaki h",
                "length_preference": "short",
                "style_intensity": "strong",
                "num_drafts": 2,
            }
            resp = await ac.post("/api/draft", data=data)
            assert resp.status_code == 200

            call_args = mock_gen.call_args
            usr_prompt = call_args.kwargs.get("user_prompt") or call_args.args[1]
            sys_prompt = call_args.kwargs.get("system_prompt") or call_args.args[0]

            # Intent must be explicitly labeled and intact
            assert "bas thoda aur baaki h" in usr_prompt
            assert "USER INTENT / FACTS TO CONVEY:" in usr_prompt
            assert "Intent is the factual content to convey; style profile only controls phrasing" in usr_prompt
            assert "Intent is the factual content to convey; style profile only controls phrasing" in sys_prompt
            assert '"padh liye?"' in usr_prompt

