import re
from typing import List, Tuple, Optional, Dict
from app.models import StyleProfile, DraftRequest


def clean_sender_prefix(line: str) -> str:
    """Strips leading timestamps and sender prefixes like 'Alex: ' or '[10:14] John: '."""
    cleaned = line.strip()
    # Strip bracketed timestamp: "[15/01/24, 10:14:00] Name: Message"
    cleaned = re.sub(r"^\[.*?\]\s*", "", cleaned)
    # Strip dash timestamp: "15/01/24, 10:14 am - Name: Message"
    cleaned = re.sub(r"^\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP]\.?[mM]\.?)?\s*-\s*", "", cleaned)
    # Strip sender name prefix like "Alex: "
    cleaned = re.sub(r"^[^:\n]{1,30}:\s*", "", cleaned)
    return cleaned.strip()


def extract_incoming_and_history(context: str) -> Tuple[str, str]:
    """
    Splits conversation context into (history_str, latest_incoming_message).
    The last non-empty line/turn is treated as the incoming message from the other person.
    """
    lines = [line.strip() for line in context.strip().splitlines() if line.strip()]
    if not lines:
        return "", ""

    if len(lines) == 1:
        return "", clean_sender_prefix(lines[0])

    # Multiple lines: all prior lines are history, last line is latest incoming
    history = "\n".join(lines[:-1])
    latest = clean_sender_prefix(lines[-1])
    return history, latest


def is_echo_of_incoming(draft: str, incoming_cleaned: str) -> bool:
    """
    Checks if a draft simply repeats or echoes the incoming question/message
    without actually answering it.
    """
    if not incoming_cleaned:
        return False

    d_norm = re.sub(r"[^\w\s]", "", draft.lower()).strip()
    in_norm = re.sub(r"[^\w\s]", "", incoming_cleaned.lower()).strip()

    if not in_norm:
        return False

    # Exact or near-exact repetition
    if d_norm == in_norm:
        return True

    # If incoming was a short question and draft just restates it
    if len(in_norm.split()) <= 6 and d_norm.startswith(in_norm) and "?" in draft:
        if len(d_norm.split()) <= len(in_norm.split()) + 3:
            return True

    return False


def is_factual_personal_question(incoming_text: str, context_history: str = "") -> Tuple[bool, Optional[str]]:
    """
    Determines whether an incoming message asks a factual personal question
    (e.g., whether the user studied, ate, slept, completed a task, or their location)
    that cannot be answered without personal knowledge.
    """
    inc = incoming_text.strip().lower()
    if not inc:
        return False, None

    # Exclude social greetings / phatic communication
    greeting_triggers = [
        "kaisa", "kaise", "kaisi", "how are you", "how r u", "wassup",
        "what's up", "whats up", "kya haal", "haal chal"
    ]
    if any(g in inc for g in greeting_triggers):
        return False, None

    # Exclude general availability / plans inquiry handled by noncommittal response
    plans_triggers = ["mil raha", "milna", "chalega"]
    if any(p in inc for p in plans_triggers):
        return False, None

    # Patterns for factual personal inquiries
    study_re = re.compile(r"\b(?:padh[a-z]*|study[a-z]*|homework|assignment|exam)\b", re.IGNORECASE)
    food_re = re.compile(r"\b(?:kha(?:na|ya|ye|liya|liye|raha|rahe|ye)|lunch|dinner|breakfast|nashta)\b", re.IGNORECASE)
    sleep_re = re.compile(r"\b(?:so(?:ya|ye|gaya|gaye|raha|rahe)?|sleep[a-z]*|uth[a-z]*)\b", re.IGNORECASE)
    reach_re = re.compile(r"\b(?:pahuch[a-z]*|reach[a-z]*|aaya|aa gaya)\b", re.IGNORECASE)
    action_re = re.compile(r"\b(?:bhej[a-z]*|kar liya|kar diya|ho gaya|hua kya|bheja|banaya)\b", re.IGNORECASE)
    location_re = re.compile(r"\b(?:kaha|kahan|kidhar|where are you|where r u|ghar (?:par|pe)|office me)\b", re.IGNORECASE)

    # Check question form (ends with ?, or contains kya/kab/kidhar/kaha/did/have/are)
    is_question = "?" in inc or any(w in inc.split() for w in ["kya", "kab", "kahan", "kaha", "kidhar", "did", "have", "are"])
    if not is_question:
        return False, None

    topic = None
    if study_re.search(inc):
        topic = "whether you studied"
    elif food_re.search(inc):
        topic = "whether you had food"
    elif sleep_re.search(inc):
        topic = "whether you are awake or asleep"
    elif reach_re.search(inc):
        topic = "whether you reached"
    elif location_re.search(inc):
        topic = "your location"
    elif action_re.search(inc):
        topic = "whether this task or action was completed"

    if not topic:
        return False, None

    # Check if context history already contains user's explicit answer
    if context_history:
        hist_lower = context_history.lower()
        user_lines = [line for line in hist_lower.splitlines() if line.strip().startswith("me:")]
        if user_lines:
            last_user_msg = " ".join(user_lines)
            if topic == "whether you studied" and ("padh" in last_user_msg or "study" in last_user_msg):
                return False, None
            if topic == "whether you had food" and ("kha" in last_user_msg or "dinner" in last_user_msg or "lunch" in last_user_msg):
                return False, None
            if topic == "your location" and ("ghar" in last_user_msg or "office" in last_user_msg or "yaha" in last_user_msg):
                return False, None

    return True, topic


def detect_intent_category(incoming_text: str, user_intent: Optional[str] = "") -> str:
    """Classifies the communicative goal of the conversation turn."""
    inc_lower = incoming_text.lower()
    intent_lower = (user_intent or "").lower()

    # 1. Greetings & "How are you?" status inquiry
    status_triggers = [
        "kaisa", "kaise", "kaisi", "how are you", "how r u", "wassup",
        "what's up", "whats up", "kya haal", "haal chal", "sab thik", "sab theek"
    ]
    if any(trigger in inc_lower for trigger in status_triggers) and not user_intent:
        return "greeting_status"

    # 2. Availability / plans without confirmation
    plans_triggers = ["mil raha", "milna", "free hai", "free h", "available", "kab mil", "chalega"]
    if any(trigger in inc_lower for trigger in plans_triggers) and not user_intent:
        return "plans_noncommittal"

    # 3. Confirmation inquiry
    confirm_triggers = ["ho gya", "hua kya", "kar diya", "bheja", "done?", "verify"]
    if any(trigger in inc_lower for trigger in confirm_triggers):
        return "affirmation"

    # 4. Gratitude / thanks
    thanks_triggers = ["thanks", "thank you", "thnx", "thx", "shukriya"]
    if any(trigger in inc_lower for trigger in thanks_triggers):
        return "thanks"

    return "general"


def retrieve_relevant_phrases(profile: StyleProfile, intent_category: str) -> List[str]:
    """Retrieves authentic short phrases written by the user matching the communicative intent."""
    if not profile:
        return []

    phrases = []
    # Check intent_phrases first
    if profile.intent_phrases and intent_category in profile.intent_phrases:
        phrases.extend(profile.intent_phrases[intent_category][:4])

    # Fallback to frequent short phrases if none found for category
    if not phrases and profile.frequent_short_phrases:
        # If greeting status, check for status words
        if intent_category == "greeting_status":
            for p in profile.frequent_short_phrases:
                p_low = p.lower()
                if any(w in p_low for w in ["thik", "theek", "mast", "badhiya", "ha", "chal"]):
                    phrases.append(p)
                    if len(phrases) >= 3:
                        break

    return phrases[:4]


def apply_spelling_preferences(text: str, spelling_prefs: Dict[str, str]) -> str:
    """
    Deterministically transforms words to match the user's learned spelling conventions.
    E.g. replaces 'theek' -> 'thik', 'hai' -> 'h', 'hoon' -> 'hu', 'nahi' -> 'nhi'.
    Preserves capitalization of the original word.
    """
    if not spelling_prefs or not text:
        return text

    modified = text
    for non_pref, pref in spelling_prefs.items():
        # Match word boundaries case-insensitively
        pattern = re.compile(r"\b" + re.escape(non_pref) + r"\b", re.IGNORECASE)

        def _repl(match):
            orig = match.group(0)
            if orig.isupper():
                return pref.upper()
            elif orig[0].isupper() and len(orig) > 1:
                return pref.capitalize()
            return pref

        modified = pattern.sub(_repl, modified)

    return modified


def build_system_prompt(profile: StyleProfile, intensity: str, incoming_text: str = "", user_intent: str = "") -> str:
    """
    Builds the system prompt instructing the model on response priority,
    communicative intent, safety, anti-echo rules, and learned spelling preferences.
    """
    intent_category = detect_intent_category(incoming_text, user_intent)
    relevant_phrases = retrieve_relevant_phrases(profile, intent_category)

    # Capitalization instructions
    if profile.capitalization.primary_case == "all_lowercase":
        case_instruction = "Use mostly lowercase letters (e.g. 'thik h tu bata' or 'chal thik h')."
    elif profile.capitalization.primary_case == "sentence_case":
        case_instruction = "Use standard sentence case capitalization."
    else:
        case_instruction = "Use a casual mix of sentence case and lowercase."

    # Punctuation instructions
    punct_notes = []
    if profile.punctuation.period_ending_freq < 0.2:
        punct_notes.append("Rarely or never end short messages with a period.")
    else:
        punct_notes.append("Use standard periods at the end of thoughts.")

    if profile.punctuation.exclamation_freq > 0.5:
        punct_notes.append("Use exclamation marks occasionally for enthusiasm.")

    if profile.punctuation.question_freq > 0.4:
        punct_notes.append(
            "You may include a natural follow-up question (e.g. 'tu bata?' or 'you?') AFTER answering first. Never ask a question instead of answering."
        )

    punctuation_instruction = " ".join(punct_notes) if punct_notes else "Natural casual punctuation."

    # Emoji instructions
    if profile.emojis.top_emojis:
        top_emojis_str = " ".join(profile.emojis.top_emojis[:6])
        if profile.emojis.emoji_frequency > 0.8:
            emoji_instruction = f"Use emojis regularly (around 1 per message). Favorite emojis: {top_emojis_str}."
        elif profile.emojis.emoji_frequency > 0.2:
            emoji_instruction = f"Use emojis moderately (e.g. 1 emoji every 1-2 messages). Preferred emojis: {top_emojis_str}."
        else:
            emoji_instruction = f"Use emojis sparingly. Preferred emojis if relevant: {top_emojis_str}."
    else:
        emoji_instruction = "Do not use emojis unless naturally fitting."

    # Data-grounded spelling conventions
    spelling_block = ""
    if profile.spelling_preferences:
        rules = []
        for non_pref, pref in sorted(profile.spelling_preferences.items())[:12]:
            rules.append(f"'{pref}' (never '{non_pref}')")
        spelling_block = (
            f"\n=== MANDATORY SPELLING CONVENTIONS FROM USER'S CHAT HISTORY ===\n"
            f"The user specifically uses these exact Romanized spellings. You MUST adhere to them:\n"
            f"- Use: {', '.join(rules)}.\n"
            f"- Do NOT 'correct' or normalize the user's spellings to textbook Hindi (e.g. do not replace '{profile.spelling_preferences.get('hai', 'h')}' with 'hai' or '{profile.spelling_preferences.get('theek', 'thik')}' with 'theek').\n"
        )

    # Retrieved authentic user phrases (only when intent is blank so we do not override intent)
    phrase_block = ""
    if not user_intent:
        if relevant_phrases:
            formatted_phrases = "\n".join(f"- \"{p}\"" for p in relevant_phrases)
            phrase_block = (
                f"\n=== USER'S ACTUAL RECURRING PHRASES FOR THIS INTENT ===\n"
                f"These are exact short phrases the user actually wrote in past chats for similar situations. Prefer using these exact phrases and rhythm:\n"
                f"{formatted_phrases}\n"
            )
        elif profile.frequent_short_phrases:
            formatted_phrases = "\n".join(f"- \"{p}\"" for p in profile.frequent_short_phrases[:5])
            phrase_block = (
                f"\n=== USER'S FREQUENT SHORT MESSAGES ===\n"
                f"Short phrases the user regularly uses:\n"
                f"{formatted_phrases}\n"
            )

    # Style samples (filter out questions)
    safe_samples = [s for s in profile.sample_snippets if not s.strip().endswith("?")][:3]
    sample_block = ""
    if safe_samples:
        formatted_samples = "\n".join(f"- {s}" for s in safe_samples)
        sample_block = (
            f"\nCADENCE EXAMPLES (For rhythm and tone ONLY — never copy verbatim, never override intent answering):\n{formatted_samples}\n"
        )

    intensity_guidance = {
        "subtle": "Apply stylistic traits subtly. Prioritize clear, direct, and polite phrasing.",
        "moderate": "Balance stylistic habits naturally with direct answers.",
        "strong": "Strictly emphasize the user's personal chat spellings, lowercase style, and slang.",
    }.get(intensity, "Balance stylistic habits naturally.")

    custom_notes_block = f"\nUser Custom Preferences: {profile.custom_notes}" if profile.custom_notes.strip() else ""

    prompt = f"""You are an AI writing assistant drafting a WhatsApp reply on behalf of the account owner ("Me").
The drafts you produce will be reviewed, edited, and approved by the user before anything is sent.

=== PRIORITY 1: CORE RESPONSE TASK & COMMUNICATIVE INTENT ===
1. FIRST-PERSON PERSPECTIVE: The incoming message was sent by the OTHER PERSON to the user. You are writing MY direct reply back to them.
2. INTENT IS FACTUAL CONTENT; STYLE ONLY CONTROLS PHRASING:
   - Intent is the factual content to convey; style profile only controls phrasing.
   - When an explicit INTENT is provided, your reply MUST convey that exact factual answer.
   - Do NOT alter, reverse, ignore, or contradict the intent.
   - Do NOT add unrelated questions, explanations, excuses, claims, or instructions (e.g. do NOT invent "baki kaam karo", "so raha hu", "mehnat karo").
   - For simple questions and explicit intent, produce a concise direct answer (prefer 1 short sentence, 2 to 6 words).
   - Every draft option must be semantically faithful to the same intent without contradicting one another.
3. ANSWER DIRECTLY: Respond to the communicative intent of the incoming message:
   - If they ask how you are (e.g. "kaisa hai?", "kaise ho bhai?", "kaisa h bhai?", "how are you?"):
     Directly answer your state first (e.g. using the user's preferred spelling like "thik hu, tu bata", "thik h", "thik hu bhai", or "all good").
   - When no specific intent or facts are supplied, provide a short, natural, neutral answer rather than repeating their question.
   - For simple greetings, keep the reply very short (prefer 1 concise sentence, 2-6 words).
   - A brief reciprocal question (e.g. "tu bata?", "tu suna?", "you?") is allowed ONLY AFTER directly answering first.
4. NEVER ECHO OR REPEAT:
   - Do NOT copy, translate, paraphrase, or ask the incoming question back as your reply.
   - For example: if incoming is "kaisa h bhai?", NEVER reply with "kaisa h bhai?" or "kaisi hai?" or "kya hua".
5. NEVER INVENT FACTS:
   - Do NOT invent specific plans, commitments, or unknown facts.
   - Do NOT hallucinate personal plans, availability, tasks, actions, or status not specified in the intent or context.
   - If asked whether you did something, studied, ate, or where you are (e.g. "padh liya?", "khana khaya?", "kaha ho?") and intent is missing, DO NOT GUESS OR FABRICATE AN ANSWER.
   - If asked about availability/plans (e.g. "kal mil raha hai?") and no facts are provided, answer with a concise clarification or safe noncommittal response (e.g., "kal ka pakka nahi hai abhi" or "kl ka pakka nahi h, shaam ko batata hu" or "dekhna padega, kitne baje?").
6. LANGUAGE & DIALECT MATCHING:
   - Always reply in the same language and dialect as the conversation (e.g., casual Hinglish for Hinglish, English for English).
{spelling_block}{phrase_block}
=== PRIORITY 2: STYLE PROFILE GUIDANCE ===
- Sender Profile: {profile.sender_name} ({profile.total_training_messages} messages analyzed, {profile.confidence_level} confidence)
- Intensity: {intensity_guidance}
- Capitalization: {case_instruction}
- Punctuation: {punctuation_instruction}
- Emojis: {emoji_instruction}
- Characteristic Vocabulary: {', '.join(profile.slang_and_phrases[:15]) if profile.slang_and_phrases else 'Casual WhatsApp phrasing'}{sample_block}{custom_notes_block}

=== OUTPUT FORMAT ===
Output ONLY the requested draft reply options separated by delimiter headers.
Do not include any conversational preamble, quotes, explanations, or analysis.
"""
    return prompt.strip()


def build_user_prompt(request: DraftRequest, profile: Optional[StyleProfile] = None) -> str:
    """
    Builds the user turn prompt detailing incoming message, conversation history,
    optional intent, desired length, and number of draft options requested.
    """
    history, incoming = extract_incoming_and_history(request.context)
    target_incoming = incoming if incoming else request.context.strip()
    history_block = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""

    length_guidelines = {
        "short": "Short (1 sentence or a quick 2-6 word response, typical for quick WhatsApp replies)",
        "medium": "Medium (1-2 sentences, natural chat conversational length)",
        "long": "Long (2-4 sentences, explanatory or detailed reply)",
    }.get(request.length_preference, "Short (1 sentence, 2-6 words)")

    has_intent = bool(request.intent and request.intent.strip())

    if has_intent:
        clean_intent = request.intent.strip()
        intent_block = f"""USER INTENT / FACTS TO CONVEY:
{clean_intent}
(CRITICAL: Intent is the factual content to convey; style profile only controls phrasing. Express this exact meaning directly in my voice. Do NOT restate or quote this instruction. Do NOT add unrelated facts, tasks, or claims)."""
        demonstrations = """EXAMPLES OF DIRECT ANSWERS PRESERVING INTENT:
Incoming: "khana kha liya?"
Intent: "nahi abhi nahi khaya"
=== DRAFT 1 ===
nhi abhi nhi khaya
=== DRAFT 2 ===
nhi khaya abhi

Incoming: "file bhej diya?"
Intent: "haan bhej diya"
=== DRAFT 1 ===
ha bhej diya
=== DRAFT 2 ===
ha bhej diya bhai

"""
    else:
        demonstrations = ""
        intent_category = detect_intent_category(target_incoming, "")
        if intent_category == "greeting_status":
            intent_block = """USER INTENT:
(None provided — this field is optional. Answer how I am directly in 1 short sentence using my learned style/phrasing like 'thik h' or 'thik hu bhai', and optionally ask 'tu bata?'. Do NOT invent specific plans, commitments, or unknown facts)."""
        else:
            intent_block = """USER INTENT:
(None provided — this field is optional. Infer a direct, natural answer from context. Do NOT invent specific plans, commitments, or unknown facts)."""

    # Retrieved candidates hint only when intent is blank
    retrieved_hint = ""
    if profile and not has_intent:
        intent_category = detect_intent_category(target_incoming, "")
        rel_phrases = retrieve_relevant_phrases(profile, intent_category)
        if rel_phrases:
            retrieved_hint = f"\nRELEVANT AUTHENTIC USER PHRASES TO FAVOR:\n" + "\n".join(f"- {p}" for p in rel_phrases) + "\n"

    num_drafts = request.num_drafts

    prompt = f"""{history_block}LATEST INCOMING MESSAGE (from other person to me):
"{target_incoming}"

{intent_block}{retrieved_hint}

{demonstrations}TARGET LENGTH: {length_guidelines}
STYLE INTENSITY: {request.style_intensity}

INSTRUCTIONS:
Generate exactly {num_drafts} distinct draft reply options from my perspective answering the latest incoming message.
- Every option must directly ANSWER the other person; do not repeat, echo, or rephrase their question.
- Intent is the factual content to convey; style profile only controls phrasing.
- Preserve the exact meaning of the intent across all options; do NOT add unrelated claims, actions, or details.
- For simple questions and explicit intent, keep the reply to 1 short sentence (2 to 6 words).
- Apply my exact learned spellings consistently across all options.
Format your output strictly with delimiter headers like this:

=== DRAFT 1 ===
[draft 1 text]

=== DRAFT 2 ===
[draft 2 text]

Do not output any markdown formatting, bullet points, numbering, or introductory commentary outside the draft delimiters.
"""
    return prompt.strip()


def parse_draft_options(
    raw_output: str,
    expected_count: int,
    incoming_text: str = "",
    profile: Optional[StyleProfile] = None,
    user_intent: str = "",
) -> List[str]:
    """
    Parses the delimited draft options from the model's raw completion output.
    Ensures options strictly convey user_intent when provided, applies deterministic
    spelling normalization from profile, filters echoes, and ensures distinct replies.
    """
    spelling_prefs = profile.spelling_preferences if profile else {}
    is_lowercase = profile.capitalization.primary_case == "all_lowercase" if profile else False
    strip_periods = profile.punctuation.period_ending_freq < 0.2 if profile else False

    # When intent is present, prepare styled intent
    styled_intent = ""
    if user_intent and user_intent.strip():
        styled_intent = apply_spelling_preferences(user_intent.strip(), spelling_prefs)
        if is_lowercase:
            styled_intent = styled_intent.lower()
        if strip_periods:
            styled_intent = styled_intent.rstrip(".!")

    # If output is completely empty and no intent, return fallback error
    if (not raw_output or not raw_output.strip()) and not styled_intent:
        return ["Unable to generate draft. Please try again."]

    _, incoming_clean = extract_incoming_and_history(incoming_text) if incoming_text else ("", "")

    pattern = re.compile(r"(?:===|---|###)?\s*DRAFT\s*\d+\s*(?:===|---|:)?", re.IGNORECASE)
    raw_drafts: List[str] = []

    if raw_output and pattern.search(raw_output):
        parts = pattern.split(raw_output)
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            if cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1].strip()

            if cleaned and not cleaned.lower().startswith("here are") and not cleaned.lower().startswith("draft"):
                raw_drafts.append(cleaned)
    elif raw_output:
        numbered_matches = re.split(r"(?:^|\n)\s*(?:\d+[\.\)]|Draft\s*\d+:?)\s*", raw_output.strip(), flags=re.IGNORECASE)
        numbered_candidates = [m.strip() for m in numbered_matches if m.strip()]
        if len(numbered_candidates) >= 2:
            raw_drafts = numbered_candidates
        else:
            lines = [line.strip() for line in raw_output.strip().split("\n\n") if line.strip()]
            for line in lines:
                stripped = re.sub(r"^(?:\d+[\.\)]|Draft\s*\d+:?)\s*", "", line, flags=re.IGNORECASE).strip()
                if stripped:
                    raw_drafts.append(stripped)

    seen = set()
    filtered_drafts: List[str] = []

    if styled_intent:
        intent_low = user_intent.lower()
        intent_words = set(re.findall(r"\b[A-Za-z]+\b", intent_low))
        is_neg = bool(intent_words & {"nhi", "nahi", "nah", "no", "not", "mat"})
        is_aff = bool(intent_words & {"ha", "haa", "haan", "yes", "kar", "ho", "done"}) and not is_neg
        is_partial = bool(intent_words & {"thoda", "baaki", "baki", "aur", "abhi", "kuch"})

        for d in raw_drafts:
            d_clean = re.sub(r"^(?:Me|Reply|Option\s*\d+):\s*", "", d, flags=re.IGNORECASE).strip()
            if not d_clean or (incoming_clean and is_echo_of_incoming(d_clean, incoming_clean)):
                continue

            # Semantic contradiction checks
            d_words = set(re.findall(r"\b[A-Za-z]+\b", d_clean.lower()))
            if is_neg and (not (d_words & {"nhi", "nahi", "nah", "no", "not", "mat"}) or (d_words & {"ha", "haan", "yes"})):
                continue
            if is_aff and not is_partial and (d_words & {"nhi", "nahi", "nah", "no", "not", "baaki", "baki", "thoda"}):
                continue
            if is_partial and (not (d_words & {"thoda", "baaki", "baki", "aur", "abhi"}) or (d_words & {"pura ho gaya", "done"})):
                continue
            if any(w in d_clean.lower() for w in ["baki kaam karo", "mehnat karo", "so raha hu", "tu padha"]):
                continue

            d_clean = apply_spelling_preferences(d_clean, spelling_prefs)
            if is_lowercase:
                d_clean = d_clean.lower()
            if strip_periods:
                d_clean = d_clean.rstrip(".!")

            key = d_clean.lower()
            if key not in seen:
                seen.add(key)
                filtered_drafts.append(d_clean)

        # Ensure base styled intent is included
        if styled_intent.lower() not in seen:
            seen.add(styled_intent.lower())
            filtered_drafts.insert(0, styled_intent)

        # If more options needed to satisfy expected_count, generate faithful natural variations
        if len(filtered_drafts) < expected_count:
            if is_partial:
                vars_list = [f"thoda aur baaki h abhi", f"bas thoda baaki h", f"{styled_intent} abhi"]
            elif is_aff:
                vars_list = [f"{styled_intent} bhai", styled_intent.replace("liya", "liye"), f"{styled_intent} ho gya"]
            elif is_neg:
                vars_list = [f"{styled_intent} abhi", f"nhi abhi {styled_intent}".replace("nhi nhi", "nhi"), f"{styled_intent} bhai"]
            else:
                vars_list = [f"{styled_intent} abhi", f"{styled_intent} bhai"]

            for v in vars_list:
                v_s = apply_spelling_preferences(v, spelling_prefs)
                if is_lowercase:
                    v_s = v_s.lower()
                if strip_periods:
                    v_s = v_s.rstrip(".!")
                if v_s.lower() not in seen:
                    seen.add(v_s.lower())
                    filtered_drafts.append(v_s)
                    if len(filtered_drafts) >= expected_count:
                        break

        return filtered_drafts[:expected_count]

    # No intent provided: standard flow
    for d in raw_drafts:
        d_clean = d.strip()
        if not d_clean:
            continue
        d_clean = re.sub(r"^(?:Me|Reply|Option\s*\d+):\s*", "", d_clean, flags=re.IGNORECASE).strip()

        # Apply deterministic spelling normalization
        if spelling_prefs:
            d_clean = apply_spelling_preferences(d_clean, spelling_prefs)
        if is_lowercase:
            d_clean = d_clean.lower()
        if strip_periods:
            d_clean = d_clean.rstrip(".!")

        key = d_clean.lower()
        if key in seen:
            continue

        if incoming_clean and is_echo_of_incoming(d_clean, incoming_clean):
            continue

        seen.add(key)
        filtered_drafts.append(d_clean)

    # If greeting status and drafts count is under expected_count, supplement with user's learned authentic phrase
    if profile and incoming_clean and len(filtered_drafts) < expected_count:
        intent_cat = detect_intent_category(incoming_clean)
        if intent_cat == "greeting_status":
            rel_phrases = retrieve_relevant_phrases(profile, intent_cat)
            if rel_phrases:
                top_phrase = rel_phrases[0]
                if not any(top_phrase.lower() in d.lower() for d in filtered_drafts):
                    filtered_drafts.append(top_phrase)

    # Fallback if all options were filtered
    if not filtered_drafts:
        if profile and profile.spelling_preferences:
            fav_h = profile.spelling_preferences.get("hai", "h")
            fav_thik = profile.spelling_preferences.get("theek", "thik")
            filtered_drafts = [f"{fav_thik} {fav_h}, tu bata"]
        else:
            filtered_drafts = ["thik hu, tu bata"]

    return filtered_drafts[:expected_count]
