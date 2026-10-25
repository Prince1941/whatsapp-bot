import re
import json
from typing import List, Tuple, Optional, Dict
from app.models import StyleProfile, DraftRequest, ChatPair
from app.validator import repair_and_dedupe_options, is_echo


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


def build_system_prompt(
    profile: StyleProfile,
    intensity: str,
    incoming_text: str = "",
    user_intent: str = "",
    target_script: str = "hinglish",
) -> str:
    """
    Builds the system prompt instructing the model on persona, response priority,
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

    # High-priority target script instructions
    if target_script == "devanagari":
        script_guidance = (
            "=== STRICT TARGET SCRIPT: DEVANAGARI HINDI (हिन्दी) ===\n"
            "1. You MUST write ALL draft replies strictly in Devanagari Hindi script (e.g. हाँ, ठीक हूँ, नहीं, अभी नहीं).\n"
            "2. NEVER use the Latin alphabet / Roman letters (no A-Z).\n"
            "3. NEVER transliterate into Romanized Hinglish or English.\n"
            "4. Match casual chat tone, but write every word in Devanagari script."
        )
        spelling_block = ""
        phrase_block = ""
        sample_block = ""
    elif target_script == "english":
        script_guidance = (
            "=== STRICT TARGET SCRIPT: ENGLISH ===\n"
            "1. You MUST write ALL draft replies strictly in natural English using the Latin alphabet.\n"
            "2. NEVER use Hindi or Devanagari words.\n"
            "3. Keep tone casual and direct."
        )
        spelling_block = ""
        phrase_block = ""
        sample_block = ""
    else:  # hinglish
        script_guidance = (
            "=== STRICT TARGET SCRIPT: ROMANIZED HINGLISH ===\n"
            "1. You MUST write ALL draft replies in Romanized Hinglish using the Latin alphabet (e.g. 'ha', 'thik hu', 'nhi').\n"
            "2. NEVER use Devanagari script characters.\n"
            "3. Use the user's Romanized spelling shortcuts."
        )
        spelling_block = ""
        if profile.spelling_preferences:
            rules = []
            for non_pref, pref in sorted(profile.spelling_preferences.items())[:12]:
                rules.append(f"'{pref}' (never '{non_pref}')")
            spelling_block = (
                f"\n=== MANDATORY SPELLING CONVENTIONS FROM USER'S CHAT HISTORY ===\n"
                f"The user specifically uses these exact Romanized spellings. You MUST adhere to them:\n"
                f"- Use: {', '.join(rules)}.\n"
                f"- Do NOT 'correct' or normalize the user's spellings to textbook Hindi.\n"
            )

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

    prompt = f"""You write WhatsApp replies AS ME. You are not an assistant. Output only the reply text.
Rules:
1. First, correctly answer the latest incoming message. Never answer a different question.
2. Target script is {target_script}. Mimic my style within this script: casing, punctuation, emoji use, typical length.
3. Don't copy example replies verbatim unless they fit perfectly.
4. Don't invent facts. Don't be formal. No explanations, quotes, or labels.
5. Length: short/conversational. Style intensity: {intensity}.

{script_guidance}

=== PRIORITY 1: CORE RESPONSE TASK & COMMUNICATIVE INTENT ===
1. FIRST-PERSON PERSPECTIVE: The incoming message was sent by the OTHER PERSON to the user. You are writing MY direct reply back to them.
2. INTENT IS FACTUAL CONTENT; STYLE ONLY CONTROLS PHRASING:
   - Intent is the factual content to convey; style profile only controls phrasing.
   - When an explicit INTENT is provided, your reply MUST convey that exact factual answer.
   - Do NOT alter, reverse, ignore, or contradict the intent.
   - Do NOT transliterate, translate, or normalize the user's spelling.
   - Do NOT add unrelated questions, explanations, excuses, claims, or instructions.
   - For simple questions and explicit intent, produce a concise direct answer (prefer 1 short sentence, 2 to 6 words).
   - Every draft option must be semantically faithful to the same intent without contradicting one another.
3. ANSWER DIRECTLY: Respond to the communicative intent of the incoming message:
   - If they ask how you are (e.g. "kaisa hai?", "kaise ho bhai?", "kaisa h bhai?", "how are you?"):
     Directly answer your state first (e.g. using the user's preferred spelling like "thik hu, tu bata", "thik h", "thik hu bhai", or "all good").
   - When no specific intent or facts are supplied, provide a short, natural, neutral answer rather than repeating their question.
   - For simple greetings, keep the reply very short (prefer 1 concise sentence, 2-6 words).
   - A brief reciprocal question is allowed ONLY AFTER directly answering first.
4. NEVER ECHO OR REPEAT:
   - Do NOT copy, translate, paraphrase, or ask the incoming question back as your reply.
   - For example: if incoming is "kaisa h bhai?", NEVER reply with "kaisa h bhai?" or "kaisi hai?" or "kya hua".
5. NEVER INVENT FACTS:
   - Do NOT invent specific plans, commitments, or unknown facts.
   - Do NOT hallucinate personal plans, availability, tasks, actions, or status not specified in the intent or context.
   - If asked whether you did something, studied, ate, or where you are (e.g. "padh liya?", "khana khaya?", "kaha ho?") and intent is missing, DO NOT GUESS OR FABRICATE AN ANSWER.
   - If asked about availability/plans (e.g. "kal mil raha hai?") and no facts are provided, answer with a concise clarification or safe noncommittal response (e.g., "kal ka pakka nahi hai abhi" or "kl ka pakka nahi h, shaam ko batata hu" or "dekhna padega, kitne baje?").
{spelling_block}{phrase_block}
=== PRIORITY 2: STYLE PROFILE GUIDANCE ===
- Sender Profile: {profile.sender_name} ({profile.total_training_messages} messages analyzed, {profile.confidence_level} confidence)
- Intensity: {intensity_guidance}
- Capitalization: {case_instruction}
- Punctuation: {punctuation_instruction}
- Emojis: {emoji_instruction}
- Characteristic Vocabulary: {', '.join(profile.slang_and_phrases[:15]) if profile.slang_and_phrases else 'Casual WhatsApp phrasing'}{sample_block}{custom_notes_block}

=== OUTPUT FORMAT ===
Output ONLY the requested draft reply options separated by delimiter headers or as a JSON list of strings.
Do not include any conversational preamble, quotes, explanations, or analysis.
"""
    return prompt.strip()


def build_user_prompt(
    request: DraftRequest,
    profile: Optional[StyleProfile] = None,
    retrieved_pairs: Optional[List[ChatPair]] = None,
    detected_intent: Optional[str] = None,
    intent_instruction: Optional[str] = None,
    target_script: str = "hinglish",
) -> str:
    """
    Builds the user turn prompt detailing incoming message, conversation history,
    retrieved few-shot pairs, detected intent, optional intent, and instructions.
    """
    history, incoming = extract_incoming_and_history(request.context)
    target_incoming = incoming if incoming else request.context.strip()
    history_block = f"Recent conversation:\n{history}\n\n" if history else ""

    length_guidelines = {
        "short": "Short (1 sentence or a quick 2-6 word response, typical for quick WhatsApp replies)",
        "medium": "Medium (1-2 sentences, natural chat conversational length)",
        "long": "Long (2-4 sentences, explanatory or detailed reply)",
    }.get(request.length_preference, "Short (1 sentence, 2-6 words)")

    has_intent = bool(request.intent and request.intent.strip())

    script_labels = {
        "devanagari": "Devanagari Hindi script (हिन्दी / देवनागरी) — write entirely in Devanagari, no English letters",
        "english": "English (Latin alphabet) — write in English, no Hindi or Devanagari",
        "hinglish": "Romanized Hinglish (Latin alphabet) — write in Roman script, no Devanagari",
    }
    script_label = script_labels.get(target_script, "Romanized Hinglish")

    if has_intent:
        clean_intent = request.intent.strip()
        intent_block = f"""USER INTENT / FACTS TO CONVEY:
{clean_intent}
(CRITICAL: Intent is the factual content to convey; style profile only controls phrasing. Express this exact meaning directly in my voice. Do NOT restate or quote this instruction. Do NOT add unrelated facts, tasks, or claims. Do NOT translate or transliterate if already matching the target script)."""

        if target_script == "devanagari":
            demonstrations = """EXAMPLES OF DIRECT ANSWERS PRESERVING INTENT (DEVANAGARI SCRIPT):
Incoming: "खाना खा लिया?"
Intent: "नहीं अभी नहीं खाया"
=== DRAFT 1 ===
नहीं अभी नहीं खाया
=== DRAFT 2 ===
अभी नहीं खाया

Incoming: "पढ़ लिया?"
Intent: "हाँ पढ़ लिया"
=== DRAFT 1 ===
हाँ पढ़ लिया
=== DRAFT 2 ===
हाँ पढ़ लिया भाई

"""
        elif target_script == "english":
            demonstrations = """EXAMPLES OF DIRECT ANSWERS PRESERVING INTENT (ENGLISH):
Incoming: "Did you study?"
Intent: "Yes finished studying"
=== DRAFT 1 ===
Yes, finished studying
=== DRAFT 2 ===
Yeah done with studying

Incoming: "Have you eaten?"
Intent: "Not yet"
=== DRAFT 1 ===
Not yet
=== DRAFT 2 ===
No, not yet

"""
        else:
            demonstrations = """EXAMPLES OF DIRECT ANSWERS PRESERVING INTENT (ROMANIZED HINGLISH):
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
        intent_category = detected_intent or detect_intent_category(target_incoming, "")
        if intent_category in ("greeting_status", "wellbeing"):
            if target_script == "devanagari":
                intent_block = """USER INTENT:
(None provided — this field is optional. Answer how I am directly in 1 short sentence using Devanagari Hindi like 'ठीक हूँ, तुम बताओ' or 'सब बढ़िया'. Do NOT invent specific plans or facts)."""
            elif target_script == "english":
                intent_block = """USER INTENT:
(None provided — this field is optional. Answer how I am directly in 1 short sentence in English like 'I am good, how about you?'. Do NOT invent specific plans or facts)."""
            else:
                intent_block = """USER INTENT:
(None provided — this field is optional. Answer how I am directly in 1 short sentence using my learned style/phrasing like 'thik h' or 'thik hu bhai', and optionally ask 'tu bata?'. Do NOT invent specific plans, commitments, or unknown facts)."""
        else:
            intent_block = """USER INTENT:
(None provided — this field is optional. Infer a direct, natural answer from context. Do NOT invent specific plans, commitments, or unknown facts)."""

    # Style summary
    style_summary = f"Casual WhatsApp style in {target_script}"
    if profile:
        fav_words = ", ".join(profile.slang_and_phrases[:8]) if profile.slang_and_phrases else "casual chat"
        style_summary = f"{profile.sender_name} style (casing: {profile.capitalization.primary_case}, slang: {fav_words})"

    # Retrieved few-shot pairs block
    retrieved_block = ""
    if retrieved_pairs:
        lines = []
        for p in retrieved_pairs[:6]:
            lines.append(f'- Incoming: "{p.incoming}" -> My reply: "{p.reply}"')
        script_note = " (Note: adapt tone and brevity, but keep target script)" if target_script != "hinglish" else ""
        retrieved_block = f"Similar past exchanges{script_note}:\n" + "\n".join(lines) + "\n\n"

    # Intent instruction
    intent_line = ""
    if detected_intent or intent_instruction:
        intent_cat_str = detected_intent or "general"
        instr_str = f": {intent_instruction}" if intent_instruction else ""
        intent_line = f"Detected intent: {intent_cat_str}{instr_str}\n\n"

    num_drafts = request.num_drafts

    prompt = f"""Style profile: {style_summary}
{retrieved_block}{history_block}{intent_line}LATEST INCOMING MESSAGE (from other person to me):
"{target_incoming}"

{intent_block}

TARGET LANGUAGE & SCRIPT (STRICT REQUIREMENT):
{script_label}

{demonstrations}TARGET LENGTH: {length_guidelines}
STYLE INTENSITY: {request.style_intensity}

INSTRUCTIONS:
Generate exactly {num_drafts} distinct draft reply options from my perspective answering the latest incoming message.
- SCRIPT REQUIREMENT: Output MUST be strictly in {script_label}. Do NOT use any other script.
- Every option must directly ANSWER the other person; do not repeat, echo, or rephrase their question.
- Intent is the factual content to convey; style profile only controls phrasing.
- Preserve the exact meaning and script of the intent across all options; do NOT add unrelated claims, actions, or details.
- For simple questions and explicit intent, keep the reply to 1 short sentence (2 to 6 words).
Write {num_drafts} different reply options, as a JSON list of strings (e.g. ["option 1", "option 2"]).
Alternatively format with delimiter headers:

=== DRAFT 1 ===
[draft 1 text]

=== DRAFT 2 ===
[draft 2 text]

Do not output any markdown commentary or explanations outside the JSON list or delimiters.
"""
    return prompt.strip()


def parse_draft_options(
    raw_output: str,
    expected_count: int,
    incoming_text: str = "",
    profile: Optional[StyleProfile] = None,
    user_intent: str = "",
    intent_category: str = "casual_chat",
    length_preference: str = "short",
    target_script: str = "hinglish",
) -> List[str]:
    """
    Parses draft options from model completion output (supporting JSON arrays,
    JSON objects, or delimited headers). Validates, normalizes, dedupes, and repairs options.
    """
    if not raw_output or not raw_output.strip():
        # Fall back to repair_and_dedupe_options directly
        return repair_and_dedupe_options(
            [],
            expected_count=expected_count,
            incoming=incoming_text,
            intent_category=intent_category,
            user_intent=user_intent,
            profile=profile,
            length_preference=length_preference,
            target_script=target_script,
        )

    raw_candidates: List[str] = []

    # 1. Try parsing JSON list or object
    cleaned_text = raw_output.strip()
    try:
        data = json.loads(cleaned_text)
        if isinstance(data, list):
            raw_candidates = [str(x).strip() for x in data if x]
        elif isinstance(data, dict):
            found_key = False
            for key in ("drafts", "replies", "options", "results", "reply_options"):
                if key in data and isinstance(data[key], list):
                    raw_candidates = [str(x).strip() for x in data[key] if x]
                    found_key = True
                    break
            if not found_key:
                for k, v in data.items():
                    if k and isinstance(k, str) and not k.startswith("{") and len(k.split()) <= 20:
                        raw_candidates.append(k.strip())
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.strip():
                                raw_candidates.append(item.strip())
                    elif isinstance(v, str) and v.strip():
                        raw_candidates.append(v.strip())
    except Exception:

        # Check if text contains a JSON array substring
        json_match = re.search(r"\[\s*\"(?:\\\"|[^\"])*\"(?:\s*,\s*\"(?:\\\"|[^\"])*\")*\s*\]", cleaned_text)
        if json_match:
            try:
                arr = json.loads(json_match.group(0))
                if isinstance(arr, list):
                    raw_candidates = [str(x).strip() for x in arr if x]
            except Exception:
                pass

    # 2. Delimiter parsing fallback (=== DRAFT N ===)
    if not raw_candidates:
        pattern = re.compile(r"(?:===|---|###)?\s*DRAFT\s*\d+\s*(?:===|---|:)?", re.IGNORECASE)
        if pattern.search(cleaned_text):
            parts = pattern.split(cleaned_text)
            for part in parts:
                p = part.strip()
                if p.startswith('"') and p.endswith('"'):
                    p = p[1:-1].strip()
                if p.startswith("'") and p.endswith("'"):
                    p = p[1:-1].strip()
                if p and not p.lower().startswith("here are") and not p.lower().startswith("draft"):
                    raw_candidates.append(p)

    # 3. Numbered list fallback (1. ..., 2. ...)
    if not raw_candidates:
        numbered_matches = re.split(r"(?:^|\n)\s*(?:\d+[\.\)]|Draft\s*\d+:?)\s*", cleaned_text, flags=re.IGNORECASE)
        numbered_candidates = [m.strip() for m in numbered_matches if m.strip()]
        if len(numbered_candidates) >= 2:
            raw_candidates = numbered_candidates
        else:
            lines = [line.strip() for line in cleaned_text.split("\n\n") if line.strip()]
            for line in lines:
                stripped = re.sub(r"^(?:\d+[\.\)]|Draft\s*\d+:?)\s*", "", line, flags=re.IGNORECASE).strip()
                if stripped:
                    raw_candidates.append(stripped)

    # Validate, dedupe, and repair using validator
    return repair_and_dedupe_options(
        raw_candidates,
        expected_count=expected_count,
        incoming=incoming_text,
        intent_category=intent_category,
        user_intent=user_intent,
        profile=profile,
        length_preference=length_preference,
        target_script=target_script,
    )

