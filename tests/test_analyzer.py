from app.models import ChatMessage
from app.analyzer import (
    analyze_length,
    analyze_capitalization,
    analyze_punctuation,
    analyze_emojis,
    extract_vocabulary_and_slang,
    extract_spelling_preferences,
    extract_frequent_short_phrases,
    compute_confidence,
    build_style_profile,
    select_safe_sample_snippets,
)


def test_analyze_length():
    messages = [
        ChatMessage(sender="Me", content="hey"),
        ChatMessage(sender="Me", content="sounds good to me see you later"),
        ChatMessage(sender="Me", content="this is a very long message with lots of details to explain why I will be running quite late today for our meeting so please start without me"),
    ]
    stats = analyze_length(messages)
    assert stats.avg_words_per_message > 0
    assert stats.short_ratio > 0
    assert stats.long_ratio > 0


def test_analyze_capitalization_lowercase():
    messages = [
        ChatMessage(sender="Me", content="hey bro"),
        ChatMessage(sender="Me", content="whats up"),
        ChatMessage(sender="Me", content="see you soon"),
        ChatMessage(sender="Me", content="Okay fine"),
    ]
    cap = analyze_capitalization(messages)
    assert cap.primary_case == "all_lowercase"
    assert cap.lowercase_rate == 0.75


def test_analyze_punctuation():
    messages = [
        ChatMessage(sender="Me", content="great!!"),
        ChatMessage(sender="Me", content="are you coming??"),
        ChatMessage(sender="Me", content="maybe..."),
        ChatMessage(sender="Me", content="no."),
    ]
    punct = analyze_punctuation(messages)
    assert punct.exclamation_freq > 0
    assert punct.question_freq > 0
    assert punct.ellipsis_freq > 0
    assert punct.period_ending_freq == 0.25


def test_analyze_emojis():
    messages = [
        ChatMessage(sender="Me", content="awesome 🎉🚀"),
        ChatMessage(sender="Me", content="haha 😂"),
        ChatMessage(sender="Me", content="sounds good"),
    ]
    em_stats = analyze_emojis(messages)
    assert em_stats.emoji_frequency == 1.0
    assert "😂" in em_stats.top_emojis
    assert "🎉" in em_stats.top_emojis


def test_slang_extraction():
    messages = [
        ChatMessage(sender="Me", content="haha that was funny lol"),
        ChatMessage(sender="Me", content="bro let's go rn"),
        ChatMessage(sender="Me", content="ngl it was pretty cool tbh"),
    ]
    slang = extract_vocabulary_and_slang(messages)
    assert "haha" in slang
    assert "lol" in slang
    assert "bro" in slang
    assert "ngl" in slang
    assert "tbh" in slang


def test_extract_spelling_preferences_from_raw_messages():
    """Verify that empirical spelling variants are learned directly from user data."""
    messages = [
        ChatMessage(sender="Me", content="thik h bhai"),
        ChatMessage(sender="Me", content="chal thik h"),
        ChatMessage(sender="Me", content="mai karta hu"),
        ChatMessage(sender="Me", content="nhi hua abhi"),
        ChatMessage(sender="Me", content="kl milte h"),
        ChatMessage(sender="Me", content="ha bhai"),
    ]
    prefs = extract_spelling_preferences(messages)
    # The user used 'thik' (not theek), 'h' (not hai), 'hu' (not hoon), 'nhi' (not nahi), 'kl' (not kal)
    assert prefs.get("theek") == "thik"
    assert prefs.get("hai") == "h"
    assert prefs.get("hoon") == "hu"
    assert prefs.get("nahi") == "nhi"
    assert prefs.get("kal") == "kl"


def test_extract_frequent_short_phrases():
    """Verify that frequent short authentic responses are preserved with exact casing and abbreviations."""
    messages = [
        ChatMessage(sender="Me", content="Chal thik h"),
        ChatMessage(sender="Me", content="Chal thik h"),
        ChatMessage(sender="Me", content="Nhi"),
        ChatMessage(sender="Me", content="Nhi"),
        ChatMessage(sender="Me", content="bhej diye"),
        ChatMessage(sender="Me", content="call me at 9876543210"),  # PII, must be excluded
    ]
    phrases = extract_frequent_short_phrases(messages)
    assert "Chal thik h" in phrases
    assert "Nhi" in phrases
    assert "bhej diye" in phrases
    assert not any("9876543210" in p for p in phrases)


def test_confidence_reporting():
    """Verify honest reporting of match confidence."""
    c0, _ = compute_confidence(0)
    assert c0 == "none"

    c_low, desc_low = compute_confidence(8)
    assert c_low == "low"
    assert "Limited confidence" in desc_low

    c_med, _ = compute_confidence(45)
    assert c_med == "medium"

    c_high, desc_high = compute_confidence(250)
    assert c_high == "high"
    assert "High confidence" in desc_high


def test_build_full_style_profile():
    messages = [
        ChatMessage(sender="Prince", content="thik h bhai"),
        ChatMessage(sender="Prince", content="chal thik h"),
        ChatMessage(sender="Prince", content="ha ha"),
        ChatMessage(sender="Prince", content="cya later"),
    ]
    profile = build_style_profile(messages, sender_name="Prince", profile_name="Prince Style")
    assert profile.profile_name == "Prince Style"
    assert profile.sender_name == "Prince"
    assert profile.total_training_messages == 4
    assert profile.confidence_level == "low"
    assert profile.spelling_preferences.get("hai") == "h"
    assert profile.spelling_preferences.get("theek") == "thik"


def test_select_safe_sample_snippets_filters_pii():
    messages = [
        ChatMessage(sender="Me", content="sounds good see you there"),
        ChatMessage(sender="Me", content="call me at [REDACTED_PHONE] now"),
        ChatMessage(sender="Me", content="my number is 9876543"),
        ChatMessage(sender="Me", content="haha yeah sure thing bro"),
    ]
    samples = select_safe_sample_snippets(messages)
    assert "sounds good see you there" in samples
    assert "haha yeah sure thing bro" in samples
    for snip in samples:
        assert "[REDACTED" not in snip
        assert not any(c.isdigit() for c in snip)
