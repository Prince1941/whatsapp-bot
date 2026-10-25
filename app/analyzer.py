import re
from collections import Counter
from typing import List, Set
from app.models import (
    ChatMessage,
    StyleProfile,
    LengthStats,
    CapitalizationStyle,
    PunctuationStats,
    EmojiStats,
)

# Regex matching Unicode emojis across standard emoji planes
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001F900-\U0001F9FF"  # supplemental symbols
    r"\U0001FA00-\U0001FA6F"  # chess, symbols
    r"\U0001FA70-\U0001FAFF"  # extended pictographs
    r"\U00002600-\U000026FF"  # misc symbols
    r"]+",
    flags=re.UNICODE,
)

# Common colloquial / informal words and slang markers
COMMON_SLANG_CANDIDATES = {
    "haha", "hahaha", "lol", "lmao", "rofl", "bro", "yaar", "dude", "man",
    "tbh", "ngl", "idk", "rn", "fr", "imho", "gonna", "wanna", "gotta",
    "bet", "cool", "yeah", "nah", "yep", "nope", "okies", "okayy", "okie",
    "omg", "btw", "thx", "pls", "plz", "np", "cya", "ty", "k", "kk", "aye",
    "lit", "fam", "vibes", "afaik", "bcz", "coz", "ur", "u", "r"
}

OPENER_CANDIDATES = {
    "hey", "yo", "hi", "hello", "morning", "gm", "evening", "sup", "whats up",
    "what's up", "hola", "dear", "namaste"
}

CLOSER_CANDIDATES = {
    "cya", "bye", "see ya", "cheers", "good night", "gn", "ttyl", "take care",
    "thanks", "thx", "talk soon", "later", "peace"
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "it", "this", "that", "i", "you", "he", "she", "we", "they", "my",
    "your", "his", "her", "their", "our", "me", "him", "them", "us", "be",
    "have", "has", "had", "do", "does", "did", "so", "as", "if", "but", "by",
    "from", "into", "up", "down", "out", "about", "are", "was", "were", "been"
}


def extract_emojis(text: str) -> List[str]:
    """Extract individual emojis found in a text."""
    matches = EMOJI_PATTERN.findall(text)
    # Flatten strings if multi-char cluster
    emojis: List[str] = []
    for match in matches:
        for char in match:
            emojis.append(char)
    return emojis


def analyze_length(messages: List[ChatMessage]) -> LengthStats:
    """Analyze typical word and character length distribution."""
    if not messages:
        return LengthStats()

    total_words = 0
    total_chars = 0
    short_count = 0  # < 8 words
    med_count = 0    # 8 - 25 words
    long_count = 0   # > 25 words

    for msg in messages:
        words = msg.content.split()
        num_words = len(words)
        total_words += num_words
        total_chars += len(msg.content)

        if num_words < 8:
            short_count += 1
        elif num_words <= 25:
            med_count += 1
        else:
            long_count += 1

    n = len(messages)
    return LengthStats(
        avg_words_per_message=round(total_words / n, 1),
        avg_chars_per_message=round(total_chars / n, 1),
        short_ratio=round(short_count / n, 2),
        medium_ratio=round(med_count / n, 2),
        long_ratio=round(long_count / n, 2),
    )


def analyze_capitalization(messages: List[ChatMessage]) -> CapitalizationStyle:
    """Analyze whether the user tends to type in lowercase, sentence case, or mixed."""
    if not messages:
        return CapitalizationStyle()

    lowercase_first_count = 0
    all_caps_count = 0
    analyzed_count = 0

    for msg in messages:
        text = msg.content.strip()
        if not text:
            continue

        # Look at the first alphabetical character
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            continue

        analyzed_count += 1
        first_alpha = alpha_chars[0]

        if first_alpha.islower():
            lowercase_first_count += 1

        if len(alpha_chars) >= 4 and all(c.isupper() for c in alpha_chars):
            all_caps_count += 1

    if analyzed_count == 0:
        return CapitalizationStyle()

    lowercase_rate = round(lowercase_first_count / analyzed_count, 2)
    all_caps_rate = round(all_caps_count / analyzed_count, 2)

    if all_caps_rate > 0.4:
        primary = "all_caps"
    elif lowercase_rate > 0.55:
        primary = "all_lowercase"
    elif lowercase_rate < 0.2:
        primary = "sentence_case"
    else:
        primary = "mixed"

    return CapitalizationStyle(primary_case=primary, lowercase_rate=lowercase_rate)


def analyze_punctuation(messages: List[ChatMessage]) -> PunctuationStats:
    """Analyze punctuation patterns, exclamation mark usage, and trailing periods."""
    if not messages:
        return PunctuationStats()

    n = len(messages)
    exclamations = 0
    questions = 0
    ellipses = 0
    period_endings = 0
    commas = 0

    for msg in messages:
        content = msg.content.strip()
        exclamations += content.count("!")
        questions += content.count("?")
        ellipses += content.count("...")
        commas += content.count(",")

        # Does the message end in a period?
        if content.endswith(".") and not content.endswith("..."):
            period_endings += 1

    return PunctuationStats(
        exclamation_freq=round(exclamations / n, 2),
        question_freq=round(questions / n, 2),
        ellipsis_freq=round(ellipses / n, 2),
        period_ending_freq=round(period_endings / n, 2),
        comma_freq=round(commas / n, 2),
    )


def analyze_emojis(messages: List[ChatMessage]) -> EmojiStats:
    """Analyze frequency and most popular emojis."""
    if not messages:
        return EmojiStats()

    emoji_counter: Counter = Counter()
    total_emojis = 0

    for msg in messages:
        emojis = extract_emojis(msg.content)
        total_emojis += len(emojis)
        emoji_counter.update(emojis)

    freq = round(total_emojis / len(messages), 2)
    top = [emoji for emoji, _ in emoji_counter.most_common(8)]

    return EmojiStats(emoji_frequency=freq, top_emojis=top)


def extract_vocabulary_and_slang(messages: List[ChatMessage]) -> List[str]:
    """Find recurrent slang, contractions, and distinctive informal terms."""
    word_counter: Counter = Counter()

    for msg in messages:
        # Tokenize words, removing punctuation
        clean_text = re.sub(r"[^\w\s']", " ", msg.content.lower())
        tokens = clean_text.split()
        word_counter.update(tokens)

    detected: Set[str] = set()

    # 1. Match against known slang candidates that appear at least once
    for candidate in COMMON_SLANG_CANDIDATES:
        if word_counter[candidate] > 0:
            detected.add(candidate)

    # 2. Add high-frequency distinctive words (appear >= 3 times, not stopwords)
    for word, count in word_counter.most_common(30):
        if len(word) >= 3 and word not in STOPWORDS and count >= 3:
            if not word.isdigit():
                detected.add(word)

    # Return top 15 distinctive terms
    return sorted(list(detected))[:15]


def extract_openers_and_closers(messages: List[ChatMessage]) -> tuple[List[str], List[str]]:
    """Detect typical greeting words and sign-offs."""
    openers: Counter = Counter()
    closers: Counter = Counter()

    for msg in messages:
        content = msg.content.strip().lower()
        words = re.sub(r"[^\w\s]", "", content).split()
        if not words:
            continue

        first_word = words[0]
        last_word = words[-1]

        if first_word in OPENER_CANDIDATES:
            openers[first_word] += 1

        if last_word in CLOSER_CANDIDATES:
            closers[last_word] += 1

    top_openers = [word for word, _ in openers.most_common(4)]
    top_closers = [word for word, _ in closers.most_common(4)]
    return top_openers, top_closers


def select_safe_sample_snippets(messages: List[ChatMessage], limit: int = 4) -> List[str]:
    """
    Selects a few short, generic, non-PII representative messages.
    Does not include messages with redaction tags, numbers, URLs, or long text.
    """
    candidates = []

    for msg in messages:
        content = msg.content.strip()
        words = content.split()
        # Ideal short example is 3 to 10 words
        if 3 <= len(words) <= 10:
            # Check for absence of redactions, digits, or suspicious characters
            if "[REDACTED" in content or any(c.isdigit() for c in content):
                continue
            if "http" in content.lower() or "@" in content:
                continue
            # Avoid questions in sample snippets so the model is not biased to echo questions
            if "?" in content:
                continue
            candidates.append(content)

    # Return unique snippets up to limit
    seen = set()
    unique_snippets = []
    for cand in candidates:
        if cand.lower() not in seen:
            seen.add(cand.lower())
            unique_snippets.append(cand)
            if len(unique_snippets) >= limit:
                break

    return unique_snippets


def build_style_profile(
    messages: List[ChatMessage],
    sender_name: str,
    profile_name: str = "My Style",
    custom_notes: str = "",
) -> StyleProfile:
    """Extracts a comprehensive StyleProfile from sanitized chat messages."""
    length_stats = analyze_length(messages)
    capitalization = analyze_capitalization(messages)
    punctuation = analyze_punctuation(messages)
    emojis = analyze_emojis(messages)
    slang = extract_vocabulary_and_slang(messages)
    openers, closers = extract_openers_and_closers(messages)
    samples = select_safe_sample_snippets(messages)

    return StyleProfile(
        id="default",
        profile_name=profile_name,
        sender_name=sender_name,
        total_training_messages=len(messages),
        length_stats=length_stats,
        capitalization=capitalization,
        punctuation=punctuation,
        emojis=emojis,
        slang_and_phrases=slang,
        common_openers=openers,
        common_closers=closers,
        sample_snippets=samples,
        custom_notes=custom_notes,
    )
