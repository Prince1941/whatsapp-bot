import re
from typing import List, Tuple
from app.models import ChatMessage, PrivacyFilterConfig

# Common Regex patterns for sensitive information
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Matches international and local phone numbers (e.g. +1-555-123-4567, 9876543210, +91 98765 43210)
PHONE_REGEX = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}\b"
)

# URLs and Links
URL_REGEX = re.compile(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_+.~#?&/=]*")

# Credit cards / 13-19 digit account numbers
FINANCIAL_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Password or API key tokens (common secret signatures)
SECRET_REGEX = re.compile(
    r"(?i)\b(?:password|passwd|pin|otp|secret|api_key|token)[\s:=]+([^\s,;]+)"
)


class PrivacyFilter:
    """Sanitizes chat text to prevent PII and sensitive info from entering style profiles."""

    def __init__(self, config: PrivacyFilterConfig):
        self.config = config
        self._compiled_custom_terms = [
            re.compile(re.escape(term.strip()), re.IGNORECASE)
            for term in config.custom_banned_terms
            if term.strip()
        ]

    def contains_sensitive_info(self, text: str) -> Tuple[bool, str]:
        """Check if text contains any configured sensitive patterns."""
        if self.config.redact_emails and EMAIL_REGEX.search(text):
            return True, "email"
        if self.config.redact_urls and URL_REGEX.search(text):
            return True, "url"
        if self.config.redact_financial and FINANCIAL_REGEX.search(text):
            return True, "financial"
        if SECRET_REGEX.search(text):
            return True, "secret"

        # Check phone numbers (ensure minimal digits to avoid false positives on small numbers)
        if self.config.redact_phone_numbers:
            for match in PHONE_REGEX.finditer(text):
                digits_only = re.sub(r"\D", "", match.group(0))
                if 7 <= len(digits_only) <= 15:
                    return True, "phone"

        for term_re in self._compiled_custom_terms:
            if term_re.search(text):
                return True, "custom_banned_term"

        return False, ""

    def sanitize_text(self, text: str) -> str:
        """Redacts sensitive occurrences in text with privacy placeholders."""
        sanitized = text

        if self.config.redact_emails:
            sanitized = EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized)

        if self.config.redact_urls:
            sanitized = URL_REGEX.sub("[REDACTED_URL]", sanitized)

        if self.config.redact_financial:
            sanitized = FINANCIAL_REGEX.sub("[REDACTED_FINANCIAL]", sanitized)

        sanitized = SECRET_REGEX.sub(r"\1: [REDACTED_SECRET]", sanitized)

        if self.config.redact_phone_numbers:
            def _replace_phone(m):
                digits = re.sub(r"\D", "", m.group(0))
                if 7 <= len(digits) <= 15:
                    return "[REDACTED_PHONE]"
                return m.group(0)

            sanitized = PHONE_REGEX.sub(_replace_phone, sanitized)

        for term_re in self._compiled_custom_terms:
            sanitized = term_re.sub("[REDACTED_TERM]", sanitized)

        return sanitized

    def filter_messages(
        self, messages: List[ChatMessage], target_sender: str
    ) -> List[ChatMessage]:
        """
        Filters messages:
        1. Keeps ONLY messages from target_sender (excluding other participants and system).
        2. Drops media placeholders and empty content.
        3. Applies privacy action (redact or drop).
        """
        target_sender_norm = target_sender.strip().lower()
        accepted_messages: List[ChatMessage] = []

        for msg in messages:
            # Exclude other participants, system messages, and media
            if msg.is_system or msg.is_media:
                continue
            if msg.sender.strip().lower() != target_sender_norm:
                continue

            content = msg.content.strip()
            if not content:
                continue

            has_sensitive, reason = self.contains_sensitive_info(content)

            if has_sensitive and self.config.action == "drop":
                # Drop this message entirely from training
                continue

            # If action is "redact" or no sensitive info found:
            sanitized_content = self.sanitize_text(content) if has_sensitive else content
            accepted_messages.append(
                ChatMessage(
                    sender=msg.sender,
                    timestamp_str=msg.timestamp_str,
                    content=sanitized_content,
                    is_system=False,
                    is_media=False,
                    raw_line_num=msg.raw_line_num,
                )
            )

        return accepted_messages
