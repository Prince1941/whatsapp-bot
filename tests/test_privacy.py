from app.models import ChatMessage, PrivacyFilterConfig
from app.privacy import PrivacyFilter


def test_redact_phone_numbers():
    config = PrivacyFilterConfig(redact_phone_numbers=True)
    p_filter = PrivacyFilter(config)

    text = "Call me at +1 555-123-4567 or on 9876543210 later."
    sanitized = p_filter.sanitize_text(text)
    assert "[REDACTED_PHONE]" in sanitized
    assert "9876543210" not in sanitized


def test_redact_emails_and_urls():
    config = PrivacyFilterConfig(redact_emails=True, redact_urls=True)
    p_filter = PrivacyFilter(config)

    text = "Check out https://github.com/Prince1941 and email me at test@example.com"
    sanitized = p_filter.sanitize_text(text)
    assert "[REDACTED_URL]" in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "test@example.com" not in sanitized
    assert "https://github.com" not in sanitized


def test_custom_banned_terms():
    config = PrivacyFilterConfig(custom_banned_terms=["ProjectTitan", "SecretPassword123"])
    p_filter = PrivacyFilter(config)

    text = "Do not mention ProjectTitan or the key SecretPassword123."
    sanitized = p_filter.sanitize_text(text)
    assert "[REDACTED_TERM]" in sanitized
    assert "ProjectTitan" not in sanitized
    assert "SecretPassword123" not in sanitized


def test_sender_filtering_and_media_exclusion():
    messages = [
        ChatMessage(sender="Alice", content="Hello there!"),
        ChatMessage(sender="Me", content="Hey Alice!"),
        ChatMessage(sender="SYSTEM", content="End to end encrypted", is_system=True),
        ChatMessage(sender="Me", content="<Media omitted>", is_media=True),
        ChatMessage(sender="Me", content="My phone is 555-987-6543"),
    ]

    config = PrivacyFilterConfig(redact_phone_numbers=True, action="redact")
    p_filter = PrivacyFilter(config)

    filtered = p_filter.filter_messages(messages, target_sender="Me")

    # Only 'Me' messages kept, media excluded, system excluded
    assert len(filtered) == 2
    assert all(m.sender == "Me" for m in filtered)
    assert filtered[0].content == "Hey Alice!"
    assert "[REDACTED_PHONE]" in filtered[1].content


def test_drop_action_on_sensitive_content():
    messages = [
        ChatMessage(sender="Me", content="Clean message here"),
        ChatMessage(sender="Me", content="My password is supersecretpass"),
    ]

    config = PrivacyFilterConfig(action="drop")
    p_filter = PrivacyFilter(config)

    filtered = p_filter.filter_messages(messages, target_sender="Me")
    assert len(filtered) == 1
    assert filtered[0].content == "Clean message here"
