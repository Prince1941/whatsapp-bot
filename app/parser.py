import re
from typing import List, Dict, Optional, Tuple
from app.models import ChatMessage, ParseResult

# Common patterns for WhatsApp exports:
# Stripping invisible unicode markers like \u200e (Left-to-Right Mark) and \u202f (narrow non-breaking space)
UNICODE_CLEANUP = re.compile(r"[\u200e\u200f\u202a-\u202e]")

# Regex for Android/Standard format:
# "15/01/24, 10:14 am - Sender Name: Hello world"
# "1/15/2024, 10:14 - +1 234 567 8900: Hello"
ANDROID_MSG_RE = re.compile(
    r"^(\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP]\.?[mM]\.?)?)\s*-\s*([^:]+?):\s*(.*)$"
)

# Regex for Android system messages:
# "15/01/24, 10:14 am - Messages and calls are end-to-end encrypted..."
ANDROID_SYS_RE = re.compile(
    r"^(\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP]\.?[mM]\.?)?)\s*-\s*(.*)$"
)

# Regex for iOS bracketed format:
# "[15/01/2024, 10:14:22] Sender Name: Hello world"
# "[1/15/24, 10:14:22 AM] Sender Name: Hello"
IOS_MSG_RE = re.compile(
    r"^\[(\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP]\.?[mM]\.?)?)\]\s+([^:]+?):\s*(.*)$"
)

# Regex for iOS system messages:
# "[15/01/2024, 10:14:22] Messages and calls are end-to-end encrypted..."
IOS_SYS_RE = re.compile(
    r"^\[(\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP]\.?[mM]\.?)?)\]\s+(.*)$"
)

# Media omission keywords across WhatsApp versions
MEDIA_PATTERNS = [
    re.compile(r"^<media omitted>$", re.IGNORECASE),
    re.compile(r"^<(?:image|video|audio|document|sticker|contact card|gif) omitted>$", re.IGNORECASE),
    re.compile(r"^(?:image|video|audio|document|sticker|contact card|gif)\s+omitted$", re.IGNORECASE),
    re.compile(r"^this message was deleted$", re.IGNORECASE),
    re.compile(r"^you deleted this message$", re.IGNORECASE),
]

# System message regex patterns
SYSTEM_PATTERNS = [
    re.compile(r"messages and calls are end-to-end encrypted", re.IGNORECASE),
    re.compile(r"\bcreated group\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were)\s+added\b", re.IGNORECASE),
    re.compile(r"\badded\s+.+?\bto\s+(?:the|this)\s+group\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were)\s+removed\b", re.IGNORECASE),
    re.compile(r"\bremoved\s+.+?\bfrom\s+(?:the|this)\s+group\b", re.IGNORECASE),
    re.compile(r"^(?:you\s+left|.+?\s+left)$", re.IGNORECASE),
    re.compile(r"\bjoined using this group's invite link\b", re.IGNORECASE),
    re.compile(r"\bchanged the subject\b", re.IGNORECASE),
    re.compile(r"\bchanged the group description\b", re.IGNORECASE),
    re.compile(r"\bchanged this group's icon\b", re.IGNORECASE),
    re.compile(r"\bsecurity code changed\b", re.IGNORECASE),
    re.compile(r"\bstarted a call\b", re.IGNORECASE),
    re.compile(r"\bmissed (?:group )?(?:voice|video) call\b", re.IGNORECASE),
    re.compile(r"\bdisappearing messages\b", re.IGNORECASE),
]


def clean_line(line: str) -> str:
    """Normalize line by stripping control characters and non-breaking spaces."""
    cleaned = UNICODE_CLEANUP.sub("", line)
    # Replace non-breaking spaces with standard space
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ")
    return cleaned.strip("\r")


def is_media_omitted(content: str) -> bool:
    """Check if the message content matches a media omitted placeholder."""
    trimmed = content.strip()
    return any(pattern.search(trimmed) for pattern in MEDIA_PATTERNS)


def is_system_notice(content: str) -> bool:
    """Check if content matches common WhatsApp system event notices."""
    trimmed = content.strip()
    return any(p.search(trimmed) for p in SYSTEM_PATTERNS)


def parse_line(line: str) -> Optional[Tuple[str, str, str, bool]]:
    """
    Attempt to match a line against known WhatsApp message formats.
    Returns (timestamp_str, sender, content, is_system) or None if continuation line.
    """
    cleaned = clean_line(line)

    # 1. Try iOS Message
    ios_m = IOS_MSG_RE.match(cleaned)
    if ios_m:
        date_str, time_str, sender, content = ios_m.groups()
        ts = f"{date_str} {time_str}"
        return (ts, sender.strip(), content, False)

    # 2. Try Android Message
    android_m = ANDROID_MSG_RE.match(cleaned)
    if android_m:
        date_str, time_str, sender, content = android_m.groups()
        ts = f"{date_str} {time_str}"
        return (ts, sender.strip(), content, False)

    # 3. Try iOS System Message
    ios_sys = IOS_SYS_RE.match(cleaned)
    if ios_sys:
        date_str, time_str, sys_content = ios_sys.groups()
        ts = f"{date_str} {time_str}"
        return (ts, "SYSTEM", sys_content.strip(), True)

    # 4. Try Android System Message
    android_sys = ANDROID_SYS_RE.match(cleaned)
    if android_sys:
        date_str, time_str, sys_content = android_sys.groups()
        ts = f"{date_str} {time_str}"
        return (ts, "SYSTEM", sys_content.strip(), True)

    return None


def parse_whatsapp_export(text: str) -> ParseResult:
    """
    Parses full WhatsApp .txt export text into structured ChatMessage objects,
    handling multiline messages, attachments, and system messages.
    """
    lines = text.splitlines()
    messages: List[ChatMessage] = []
    senders: Dict[str, int] = {}
    current_message: Optional[ChatMessage] = None

    system_count = 0
    media_count = 0
    user_count = 0

    for idx, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            if current_message:
                current_message.content += "\n"
            continue

        parsed = parse_line(raw_line)

        if parsed is not None:
            ts, sender, content, is_sys = parsed

            # Check if this sender looks like a system announcement that had a colon
            if not is_sys and is_system_notice(content):
                is_sys = True
                sender = "SYSTEM"

            is_media = is_media_omitted(content)

            new_msg = ChatMessage(
                sender=sender,
                timestamp_str=ts,
                content=content,
                is_system=is_sys,
                is_media=is_media,
                raw_line_num=idx,
            )

            messages.append(new_msg)
            current_message = new_msg

            if is_sys:
                system_count += 1
            else:
                if not is_media:
                    user_count += 1
                if sender not in senders:
                    senders[sender] = 0
                senders[sender] += 1

            if is_media:
                media_count += 1
        else:
            # Continuation of previous message (multiline)
            if current_message:
                cleaned = clean_line(raw_line)
                current_message.content += "\n" + cleaned
            else:
                # Leading lines before any timestamp header (rare system preamble)
                cleaned = clean_line(raw_line)
                if is_system_notice(cleaned):
                    msg = ChatMessage(
                        sender="SYSTEM",
                        timestamp_str=None,
                        content=cleaned,
                        is_system=True,
                        is_media=False,
                        raw_line_num=idx,
                    )
                    messages.append(msg)
                    current_message = msg
                    system_count += 1

    # Final pass: trim contents
    for msg in messages:
        msg.content = msg.content.strip()

    return ParseResult(
        messages=messages,
        senders=senders,
        total_lines=len(lines),
        system_message_count=system_count,
        media_count=media_count,
        user_message_count=user_count,
    )
