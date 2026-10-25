import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import (
    StyleProfile,
    DraftRequest,
    CapitalizationStyle,
    EmojiStats,
    PunctuationStats,
)
from app.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_draft_options,
    extract_incoming_and_history,
    is_echo_of_incoming,
)


def test_build_system_prompt_rules():
    profile = StyleProfile(
        profile_name="My Test Style",
        sender_name="Me",
        capitalization=CapitalizationStyle(primary_case="all_lowercase"),
        emojis=EmojiStats(emoji_frequency=1.2, top_emojis=["😂", "🔥"]),
        punctuation=PunctuationStats(period_ending_freq=0.05, question_freq=0.5),
        slang_and_phrases=["bro", "haha", "ngl"],
        custom_notes="Always be upbeat and clear.",
    )

    system_prompt = build_system_prompt(profile, intensity="strong")

    # Verify response task is Priority 1
    assert "PRIORITY 1: CORE RESPONSE TASK & COMMUNICATIVE INTENT" in system_prompt
    assert "FIRST-PERSON PERSPECTIVE" in system_prompt
    assert "ANSWER DIRECTLY" in system_prompt
    assert "NEVER ECHO OR REPEAT" in system_prompt
    assert "NEVER INVENT FACTS" in system_prompt

    # Verify few-shot examples are present
    assert "kaise ho bhai?" in system_prompt
    assert "thik hu, tu bata" in system_prompt
    assert "kal mil raha hai?" in system_prompt
    assert "kal ka pakka nahi hai abhi" in system_prompt

    # Verify style directions
    assert "lowercase" in system_prompt
    assert "😂" in system_prompt
    assert "bro" in system_prompt
    assert "Always be upbeat and clear." in system_prompt

    # Verify question_freq instruction does NOT force asking questions over answering
    assert "Never ask a question instead of answering" in system_prompt


def test_extract_incoming_and_history():
    # 1. Single message
    h1, inc1 = extract_incoming_and_history("kaisa hai bhai?")
    assert h1 == ""
    assert inc1 == "kaisa hai bhai?"

    # 2. Single message with sender prefix
    h2, inc2 = extract_incoming_and_history("Rahul: kaisa hai bhai?")
    assert h2 == ""
    assert inc2 == "kaisa hai bhai?"

    # 3. Multi-turn conversation
    multi = """Alex: Hey are you free?
Me: Yeah just finished work
Alex: kaisa hai bhai?"""
    h3, inc3 = extract_incoming_and_history(multi)
    assert "Alex: Hey are you free?" in h3
    assert "Me: Yeah just finished work" in h3
    assert inc3 == "kaisa hai bhai?"

    # 4. WhatsApp timestamped format
    wa = "15/01/24, 10:14 am - Amit: kaisa hai bhai?"
    _, inc4 = extract_incoming_and_history(wa)
    assert inc4 == "kaisa hai bhai?"


def test_build_user_prompt_with_blank_intent():
    req = DraftRequest(
        context="kaisa hai bhai?",
        intent="",
        length_preference="short",
        style_intensity="moderate",
        num_drafts=2,
    )
    user_prompt = build_user_prompt(req)

    assert "LATEST INCOMING MESSAGE (from other person to me):" in user_prompt
    assert '"kaisa hai bhai?"' in user_prompt
    assert "USER INTENT:" in user_prompt
    assert "None provided — this field is optional" in user_prompt
    assert "Short" in user_prompt
    assert "exactly 2 distinct draft" in user_prompt
    assert "do not repeat, echo, or rephrase their question" in user_prompt


def test_build_user_prompt_with_explicit_intent():
    req = DraftRequest(
        context="kal mil raha hai?",
        intent="Decline politely, busy with exams",
        length_preference="short",
        style_intensity="moderate",
        num_drafts=2,
    )
    user_prompt = build_user_prompt(req)
    assert "kal mil raha hai?" in user_prompt
    assert "Decline politely, busy with exams" in user_prompt
    assert "Do NOT restate or quote this instruction" in user_prompt


def test_is_echo_of_incoming():
    incoming = "kaisa hai bhai?"
    # Exact echo
    assert is_echo_of_incoming("kaisa hai bhai?", incoming) is True
    # Minor punctuation / case variant
    assert is_echo_of_incoming("Kaisa hai bhai???", incoming) is True
    # Rephrasing question echo
    assert is_echo_of_incoming("kaisa hai bhai? aur kya hua?", incoming) is True

    # Genuine answers must NOT be flagged as echoes
    assert is_echo_of_incoming("thik hu, tu bata", incoming) is False
    assert is_echo_of_incoming("mast hu bhai, tu suna", incoming) is False
    assert is_echo_of_incoming("all good bro!", incoming) is False


def test_parse_draft_options_delimited():
    raw_output = """
=== DRAFT 1 ===
thik hu, tu bata

=== DRAFT 2 ===
mast hu bhai, tu suna
"""
    drafts = parse_draft_options(raw_output, expected_count=2, incoming_text="kaisa hai bhai?")
    assert len(drafts) == 2
    assert drafts[0] == "thik hu, tu bata"
    assert drafts[1] == "mast hu bhai, tu suna"


def test_parse_draft_options_filters_incoming_echoes():
    # If the model produced one echo option and one real answer option
    raw_output = """
=== DRAFT 1 ===
kaisa hai bhai?

=== DRAFT 2 ===
thik hu, tu bata
"""
    drafts = parse_draft_options(raw_output, expected_count=2, incoming_text="kaisa hai bhai?")
    assert len(drafts) >= 1
    # The echo option must be eliminated!
    assert "kaisa hai bhai?" not in drafts
    assert "thik hu, tu bata" in drafts


def test_parse_draft_options_numbered_fallback():
    raw_output = """
1. thik hu, tu bata
2. mast hu bhai, tu suna
"""
    drafts = parse_draft_options(raw_output, expected_count=2, incoming_text="kaisa hai bhai?")
    assert len(drafts) == 2
    assert drafts[0] == "thik hu, tu bata"
    assert drafts[1] == "mast hu bhai, tu suna"


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    tests = [
        test_build_system_prompt_rules,
        test_extract_incoming_and_history,
        test_build_user_prompt_with_blank_intent,
        test_build_user_prompt_with_explicit_intent,
        test_is_echo_of_incoming,
        test_parse_draft_options_delimited,
        test_parse_draft_options_filters_incoming_echoes,
        test_parse_draft_options_numbered_fallback,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests in test_prompt.py passed successfully!")
