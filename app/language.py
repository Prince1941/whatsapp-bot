import re
from typing import Tuple, Optional, Literal

LanguageScriptType = Literal["auto", "hinglish", "devanagari", "english"]

# Common Romanized Hinglish vocabulary markers
HINGLISH_WORDS = {
    "kya", "kar", "rha", "raha", "rahe", "rhi", "rahi", "hu", "hoon", "hai", "h",
    "bhai", "kaisa", "kaise", "kaisi", "thik", "theek", "nhi", "nahi", "nah",
    "ha", "haa", "haan", "kl", "kal", "aaj", "kaha", "kahan", "kidhar", "bol",
    "bata", "suna", "padh", "padha", "padhi", "kha", "khaya", "chal", "chalo",
    "aa", "aaya", "aao", "ruk", "de", "bhej", "accha", "acha", "badhiya", "mast",
    "sab", "kuch", "baaki", "baki", "thoda", "yr", "yaar", "mera", "meri", "mere",
    "tere", "teri", "tera", "apna", "apni", "pe", "par", "me", "mein", "ko", "se",
    "bhi", "to", "toh", "kr", "karoge", "karenge", "karte", "karta", "karti",
    "gaya", "gaye", "gayi", "gya", "gye", "gyia", "hua", "huye", "hue", "hui",
    "ab", "abhi", "waha", "yaha", "idhar", "udhar", "kyu", "kyun", "kyunki"
}

# Common English vocabulary markers
ENGLISH_WORDS = {
    "the", "is", "are", "am", "you", "i", "what", "where", "how", "when", "why",
    "we", "they", "he", "she", "it", "yes", "no", "not", "have", "has", "had",
    "will", "would", "can", "could", "do", "did", "done", "about", "there", "here",
    "good", "great", "fine", "okay", "ok", "thanks", "thank", "please", "call",
    "dinner", "lunch", "food", "party", "meeting", "tomorrow", "today", "tonight",
    "busy", "free", "working", "going", "come", "coming", "sure", "sounds", "with",
    "for", "at", "on", "in", "from", "just", "woke", "up", "see", "around"
}


def count_devanagari_chars(text: str) -> int:
    """Counts characters in the Devanagari Unicode block (U+0900 to U+097F)."""
    return sum(1 for ch in text if "\u0900" <= ch <= "\u097f")


def count_latin_chars(text: str) -> int:
    """Counts Latin alphabetic characters."""
    return sum(1 for ch in text if ("a" <= ch.lower() <= "z"))


def has_devanagari(text: str) -> bool:
    """Returns True if the text contains any Devanagari characters."""
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def detect_language_and_script(text: str) -> str:
    """
    Detects whether the text is:
    - 'devanagari': Devanagari Hindi
    - 'hinglish': Romanized Hindi/Hinglish
    - 'english': Standard English
    """
    if not text or not text.strip():
        return "hinglish"

    dev_count = count_devanagari_chars(text)
    latin_count = count_latin_chars(text)

    # 1. If Devanagari is present and significant or dominant
    if dev_count > 0:
        # If dominant over Latin or is at least a word's worth of Devanagari
        if dev_count >= latin_count or dev_count >= 3:
            return "devanagari"

    # 2. Latin script: Distinguish between Romanized Hinglish vs English
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
    if not words:
        # Fallback if no letters (e.g. emojis/punctuation only)
        return "hinglish"

    hinglish_matches = sum(1 for w in words if w in HINGLISH_WORDS)
    english_matches = sum(1 for w in words if w in ENGLISH_WORDS)

    if hinglish_matches > 0:
        return "hinglish"
    if english_matches > 0 and hinglish_matches == 0:
        return "english"

    # Default for Latin script in Indian WhatsApp context
    return "hinglish"


def resolve_target_language_script(
    setting: str,
    intent: Optional[str] = "",
    latest_incoming: Optional[str] = "",
) -> Tuple[str, str]:
    """
    Resolves the target language and script based on setting, user intent, and incoming message.
    Returns (target_code, reason_description).
    target_code is one of: 'devanagari', 'hinglish', 'english'.
    """
    mode = (setting or "auto").strip().lower()

    if mode == "devanagari":
        return "devanagari", "Explicit setting: Devanagari Hindi"
    elif mode == "english":
        return "english", "Explicit setting: English"
    elif mode == "hinglish":
        return "hinglish", "Explicit setting: Romanized Hinglish"

    # Mode is "auto" (default)
    # Rule 1: Follow user's explicit Intent when non-empty
    clean_intent = (intent or "").strip()
    if clean_intent:
        detected = detect_language_and_script(clean_intent)
        return detected, f"Auto-detected from Intent: {detected}"

    # Rule 2: When Intent is blank, follow latest incoming message
    clean_inc = (latest_incoming or "").strip()
    if clean_inc:
        detected = detect_language_and_script(clean_inc)
        return detected, f"Auto-detected from incoming message: {detected}"

    # Default fallback
    return "hinglish", "Auto default: Romanized Hinglish"


def validate_script(text: str, target_script: str) -> Tuple[bool, Optional[str]]:
    """
    Validates that a generated text complies strictly with the target script.
    - 'devanagari': Must contain Devanagari and must NOT be primarily Latin.
    - 'hinglish': Must NOT contain Devanagari characters.
    - 'english': Must NOT contain Devanagari characters and should be in Latin script.
    Returns (is_valid, error_reason).
    """
    if not text or not text.strip():
        return False, "Empty text"

    dev_count = count_devanagari_chars(text)
    latin_count = count_latin_chars(text)

    if target_script == "devanagari":
        if dev_count == 0:
            return False, "Expected Devanagari script, but output contained no Devanagari characters."
        if latin_count > dev_count and latin_count >= 4:
            return False, "Expected Devanagari script, but output was predominantly Latin/English script."
        return True, None

    if target_script in ("hinglish", "english"):
        if dev_count > 0:
            return False, f"Expected {target_script} in Roman script, but output contained Devanagari characters."
        if latin_count == 0 and not re.search(r"[\d\W]", text):
            return False, f"Expected {target_script} in Roman script, but output contained no Latin letters."
        return True, None

    return True, None
