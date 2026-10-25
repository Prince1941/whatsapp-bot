import re
from typing import List, Tuple, Optional, Dict
from app.models import StyleProfile
from app.language import validate_script, has_devanagari


FORMAL_ENGLISH_PATTERNS = [
    r"^(?:dear|hello\s+there|good\s+day|greetings|kind\s+regards|sincerely)\b",
    r"\b(?:as\s+per\s+your|please\s+be\s+advised|i\s+would\s+like\s+to|furthermore|moreover)\b",
    r"\b(?:i\s+hope\s+this\s+message\s+finds\s+you)\b",
]

HALLUCINATED_TASK_TRIGGERS = [
    r"\bruk\s+bhijwata\s+hu\b",
    r"\bint\s+kar\s+diye\b",
    r"\bdekha\s+int\s+ka\b",
    r"\bcout\b",
    r"\bendl\b",
]


def is_echo(draft: str, incoming: str) -> bool:
    """Checks if draft merely repeats or echoes incoming message."""
    d = re.sub(r"[^\w\s]", "", draft.lower()).strip()
    inc = re.sub(r"[^\w\s]", "", incoming.lower()).strip()
    if not d or not inc:
        return False
    if d == inc:
        return True
    if len(inc.split()) <= 6 and d.startswith(inc) and "?" in draft:
        return True
    return False


def validate_single_draft(
    draft: str,
    incoming: str,
    intent_category: str,
    user_intent: str = "",
    length_preference: str = "short",
    target_script: str = "hinglish",
) -> Tuple[bool, Optional[str]]:
    """
    Validates a single generated draft option.
    Returns (is_valid, rejection_reason).
    """
    cleaned = draft.strip()
    if not cleaned:
        return False, "Empty draft"

    # 1. Target script validation
    valid_script, script_err = validate_script(cleaned, target_script)
    if not valid_script:
        return False, script_err

    # 2. Echo of incoming
    if is_echo(cleaned, incoming):
        return False, "Draft repeats incoming message"

    # 3. Formal English rejection (only when not drafting English)
    c_low = cleaned.lower()
    if target_script != "english":
        for pat in FORMAL_ENGLISH_PATTERNS:
            if re.search(pat, c_low):
                return False, "Draft contains formal English phrases"

    # 4. Known task hallucinations when not in user intent
    if not user_intent:
        for pat in HALLUCINATED_TASK_TRIGGERS:
            if re.search(pat, c_low):
                return False, f"Draft contains hallucinated task phrase matching {pat}"

    # 5. Length limits
    words = cleaned.split()
    max_allowed = {"short": 15, "medium": 35, "long": 70}.get(length_preference, 20)
    if len(words) > max_allowed:
        return False, f"Draft length ({len(words)} words) exceeds {max_allowed} word guideline"

    # 6. Semantic intent compatibility
    if user_intent:
        if target_script == "devanagari":
            dev_neg = {"नहीं", "ना", "नही", "मत"}
            dev_aff = {"हाँ", "हा", "हाँजी", "हो", "कर"}
            dev_partial = {"थोड़ा", "बाकी", "और", "अभी"}

            has_user_neg = any(w in user_intent for w in dev_neg)
            has_user_aff = any(w in user_intent for w in dev_aff) and not has_user_neg
            has_user_part = any(w in user_intent for w in dev_partial)

            has_draft_neg = any(w in cleaned for w in dev_neg)
            has_draft_aff = any(w in cleaned for w in dev_aff)

            if has_user_neg and (not has_draft_neg or has_draft_aff):
                return False, "Draft contradicts negative Devanagari intent"
            if has_user_aff and not has_user_part and has_draft_neg:
                return False, "Draft contradicts affirmative Devanagari intent"
        else:
            intent_words = set(re.findall(r"\b[A-Za-z]+\b", user_intent.lower()))
            is_neg = bool(intent_words & {"nhi", "nahi", "nah", "no", "not", "mat"})
            is_aff = bool(intent_words & {"ha", "haa", "haan", "yes", "kar", "ho", "done"}) and not is_neg
            is_partial = bool(intent_words & {"thoda", "baaki", "baki", "aur", "abhi", "kuch"})
            d_words = set(re.findall(r"\b[A-Za-z]+\b", c_low))

            if is_neg and (not (d_words & {"nhi", "nahi", "nah", "no", "not", "mat"}) or (d_words & {"ha", "haan", "yes"})):
                return False, "Draft contradicts negative intent"
            if is_aff and not is_partial and (d_words & {"nhi", "nahi", "nah", "no", "not", "baaki", "baki", "thoda"}):
                return False, "Draft contradicts affirmative intent"
            if is_partial and (not (d_words & {"thoda", "baaki", "baki", "aur", "abhi"}) or (d_words & {"pura ho gaya", "done"})):
                return False, "Draft contradicts partial intent"
            if any(w in c_low for w in ["baki kaam karo", "mehnat karo", "so raha hu", "tu padha"]):
                return False, "Draft contains hallucinated commands"

    elif intent_category == "activity_check":
        # If user asks what you are doing, replying "thik hu" without user intent is semantically wrong
        if (c_low.startswith("thik hu") or c_low.startswith("theek hu") or c_low.startswith("badhiya hu")):
            if not any(w in c_low for w in ["ghar", "clg", "kuch", "free", "bas", "padh", "kaam", "chill"]):
                return False, "Draft answers 'how are you' instead of 'what are you doing'"

    elif intent_category == "wellbeing":
        if any(w in c_low for w in ["bhej diya", "ho gaya", "pdf"]):
            return False, "Draft does not answer wellbeing inquiry"

    return True, None


def apply_spelling_preferences(
    text: str, spelling_prefs: Dict[str, str], preserve_words: Optional[set] = None
) -> str:
    """Deterministically transforms words to match user's preferred Romanized spellings."""
    if not text or not spelling_prefs or has_devanagari(text):
        return text
    modified = text
    preserve = {w.lower() for w in preserve_words} if preserve_words else set()
    for non_pref, pref in spelling_prefs.items():
        if non_pref.lower() in preserve or pref.lower() in preserve:
            continue
        pattern = re.compile(r"\b" + re.escape(non_pref) + r"\b", re.IGNORECASE)

        def _repl(m):
            orig = m.group(0)
            if orig.isupper():
                return pref.upper()
            if orig[0].isupper() and len(orig) > 1:
                return pref.capitalize()
            return pref

        modified = pattern.sub(_repl, modified)
    return modified


def repair_and_dedupe_options(
    raw_drafts: List[str],
    expected_count: int,
    incoming: str,
    intent_category: str,
    user_intent: str = "",
    profile: Optional[StyleProfile] = None,
    length_preference: str = "short",
    target_script: str = "hinglish",
) -> List[str]:
    """
    Cleans, validates, dedupes, and normalizes draft options according to target_script.
    """
    if not raw_drafts and not user_intent:
        return ["Unable to generate draft. Please try again."]

    spelling_prefs = profile.spelling_preferences if profile else {}
    is_lowercase = profile.capitalization.primary_case == "all_lowercase" if profile else False
    strip_periods = profile.punctuation.period_ending_freq < 0.2 if profile else False
    user_intent_words = set(re.findall(r"\b\w+\b", user_intent.lower())) if user_intent else set()

    valid_drafts: List[str] = []
    seen = set()

    for d in raw_drafts:
        clean = re.sub(r"^(?:(?:Option|Draft)\s*\d+:?|Me:|Reply:)\s*", "", d, flags=re.IGNORECASE).strip()
        clean = clean.strip("\"'")

        is_valid, _ = validate_single_draft(
            clean, incoming, intent_category, user_intent, length_preference, target_script=target_script
        )
        if not is_valid:
            continue

        # Only apply Romanized spelling preferences to Hinglish, preserving user's intent words
        if target_script == "hinglish":
            styled = apply_spelling_preferences(clean, spelling_prefs, preserve_words=user_intent_words)
            if is_lowercase:
                styled = styled.lower()
        else:
            styled = clean

        if strip_periods:
            styled = styled.rstrip(".!")

        norm = styled.lower()
        if norm not in seen:
            seen.add(norm)
            valid_drafts.append(styled)
            if len(valid_drafts) >= expected_count:
                break

    # When intent is specified, preserve user's exact wording and spelling
    if user_intent:
        base = user_intent.strip()
        if strip_periods:
            base = base.rstrip(".!")
        if base.lower() not in seen:
            seen.add(base.lower())
            valid_drafts.insert(0, base)

    # Pad with safe options in the target script if necessary
    if user_intent and len(valid_drafts) < expected_count:
        if target_script == "devanagari":
            safe_fallbacks = [base]
        elif target_script == "english":
            safe_fallbacks = [base, f"{base} thanks", "Yes, all good"]
        else:
            safe_fallbacks = [base, f"{base} bhai"]

        for fb in safe_fallbacks:
            if target_script == "hinglish":
                fb_styled = apply_spelling_preferences(fb, spelling_prefs)
                if is_lowercase:
                    fb_styled = fb_styled.lower()
            else:
                fb_styled = fb

            if strip_periods:
                fb_styled = fb_styled.rstrip(".!")
            norm = fb_styled.lower()
            if norm not in seen:
                seen.add(norm)
                valid_drafts.append(fb_styled)
                if len(valid_drafts) >= expected_count:
                    break

    # Fallback if no valid options survived
    if not valid_drafts:
        if target_script == "devanagari":
            valid_drafts = ["ठीक हूँ, तुम बताओ"]
        elif target_script == "english":
            valid_drafts = ["Doing well, how about you?"]
        else:
            if profile and profile.spelling_preferences:
                fav_h = profile.spelling_preferences.get("hai", "h")
                fav_thik = profile.spelling_preferences.get("theek", "thik")
                valid_drafts = [f"{fav_thik} {fav_h}, tu bata"]
            else:
                valid_drafts = ["thik hu, tu bata"]

    return valid_drafts[:expected_count]

