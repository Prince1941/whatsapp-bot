from app.models import StyleProfile, DraftRequest, CapitalizationStyle, EmojiStats, PunctuationStats
from app.prompt import build_system_prompt, build_user_prompt, parse_draft_options


def test_build_system_prompt_rules():
    profile = StyleProfile(
        profile_name="My Test Style",
        sender_name="Me",
        capitalization=CapitalizationStyle(primary_case="all_lowercase"),
        emojis=EmojiStats(emoji_frequency=1.2, top_emojis=["😂", "🔥"]),
        punctuation=PunctuationStats(period_ending_freq=0.05),
        slang_and_phrases=["bro", "haha", "ngl"],
        custom_notes="Always be upbeat and clear.",
    )

    system_prompt = build_system_prompt(profile, intensity="strong")

    # Verify anti-hallucination and safety rules
    assert "FACTUAL ACCURACY & INTENT COME FIRST" in system_prompt
    assert "DO NOT INVENT FACTS" in system_prompt
    assert "NO VERBATIM COPYING" in system_prompt

    # Verify style directions
    assert "lowercase" in system_prompt
    assert "😂" in system_prompt
    assert "bro" in system_prompt
    assert "Always be upbeat and clear." in system_prompt


def test_build_user_prompt():
    req = DraftRequest(
        context="Alex: Want to join us for dinner tonight?",
        intent="Decline politely because I have an early morning flight.",
        length_preference="short",
        style_intensity="moderate",
        num_drafts=2,
    )
    user_prompt = build_user_prompt(req)
    assert "CONVERSATION CONTEXT:" in user_prompt
    assert "Want to join us for dinner tonight?" in user_prompt
    assert "early morning flight" in user_prompt
    assert "Short" in user_prompt
    assert "exactly 2 distinct draft" in user_prompt


def test_parse_draft_options_delimited():
    raw_output = """
=== DRAFT 1 ===
wish i could but got an early morning flight tomorrow! have fun guys

=== DRAFT 2 ===
cant make it tonight, need to pack for an early flight tomorrow ✈️
"""
    drafts = parse_draft_options(raw_output, expected_count=2)
    assert len(drafts) == 2
    assert "early morning flight tomorrow!" in drafts[0]
    assert "cant make it tonight" in drafts[1]


def test_parse_draft_options_numbered_fallback():
    raw_output = """
1. Hey man, sounds good see you there!
2. Perfect, let's catch up at 5pm.
"""
    drafts = parse_draft_options(raw_output, expected_count=2)
    assert len(drafts) == 2
    assert drafts[0] == "Hey man, sounds good see you there!"
    assert drafts[1] == "Perfect, let's catch up at 5pm."
