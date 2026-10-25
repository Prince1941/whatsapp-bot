from app.models import ChatMessage
from app.analyzer import (
    analyze_length,
    analyze_capitalization,
    analyze_punctuation,
    analyze_emojis,
    extract_vocabulary_and_slang,
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
    assert em_stats.emoji_frequency == 1.0  # 3 emojis / 3 messages
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


def test_build_full_style_profile():
    messages = [
        ChatMessage(sender="Me", content="hey bro whats up"),
        ChatMessage(sender="Me", content="sounds good see you at 5 👍"),
        ChatMessage(sender="Me", content="haha yeah no worries"),
        ChatMessage(sender="Me", content="cya later"),
    ]
    profile = build_style_profile(messages, sender_name="Me", profile_name="Test Profile")
    assert profile.profile_name == "Test Profile"
    assert profile.sender_name == "Me"
    assert profile.total_training_messages == 4
    assert profile.capitalization.primary_case == "all_lowercase"
    assert "👍" in profile.emojis.top_emojis
    assert len(profile.sample_snippets) > 0


def test_select_safe_sample_snippets_filters_pii():
    messages = [
        ChatMessage(sender="Me", content="sounds good see you there"),
        ChatMessage(sender="Me", content="call me at [REDACTED_PHONE] now"),
        ChatMessage(sender="Me", content="my number is 9876543"),
        ChatMessage(sender="Me", content="haha yeah sure thing bro"),
    ]
    samples = select_safe_sample_snippets(messages)
    # The message with [REDACTED_PHONE] and digits should NOT be in samples
    assert "sounds good see you there" in samples
    assert "haha yeah sure thing bro" in samples
    for snip in samples:
        assert "[REDACTED" not in snip
        assert not any(c.isdigit() for c in snip)
