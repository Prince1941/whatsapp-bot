import re
from typing import List, Tuple, Optional
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
        # Check if there is an answer before or after
        # If the whole draft is just the incoming words plus minimal fluff
        if len(d_norm.split()) <= len(in_norm.split()) + 3:
            return True

    return False


def build_system_prompt(profile: StyleProfile, intensity: str) -> str:
    """
    Builds the system prompt instructing the model on response priority,
    communicative intent, safety, anti-echo rules, and style guidance.
    """
    # Capitalization instructions
    if profile.capitalization.primary_case == "all_lowercase":
        case_instruction = "Use mostly lowercase letters (e.g., 'thik hu tu bata' or 'sounds good see you')."
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
            "You may include a natural follow-up question (e.g., 'tu bata?' or 'you?') AFTER answering first. Never ask a question instead of answering."
        )

    if profile.punctuation.ellipsis_freq > 0.2:
        punct_notes.append("Occasionally use ellipses ('...') for casual transitions.")

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

    # Slang & Vocabulary
    if profile.slang_and_phrases:
        slang_list = ", ".join(profile.slang_and_phrases[:15])
        vocab_instruction = f"Incorporate natural colloquialisms and vocabulary from style when appropriate: {slang_list}."
    else:
        vocab_instruction = "Keep phrasing casual, natural, and conversational."

    # Openers & closers
    openers_str = ", ".join(profile.common_openers) if profile.common_openers else "none"
    closers_str = ", ".join(profile.common_closers) if profile.common_closers else "none"

    # Style samples (filter out any questions to avoid prompting the model to echo)
    safe_samples = [s for s in profile.sample_snippets if not s.strip().endswith("?")][:4]
    sample_block = ""
    if safe_samples:
        formatted_samples = "\n".join(f"- {s}" for s in safe_samples)
        sample_block = (
            f"\nCADENCE EXAMPLES (For rhythm and tone ONLY — never copy verbatim, never override answering):\n{formatted_samples}"
        )

    intensity_guidance = {
        "subtle": "Apply stylistic traits subtly. Prioritize clear, direct, and polite phrasing.",
        "moderate": "Balance stylistic habits naturally with direct answers.",
        "strong": "Emphasize personal casual chat patterns, lowercase style, slang, and emojis closely.",
    }.get(intensity, "Balance stylistic habits naturally.")

    custom_notes_block = f"\nUser Custom Preferences: {profile.custom_notes}" if profile.custom_notes.strip() else ""

    prompt = f"""You are an AI writing assistant drafting a WhatsApp reply on behalf of the account owner ("Me").
The drafts you produce will be reviewed, edited, and approved by the user before anything is sent.

=== PRIORITY 1: CORE RESPONSE TASK & COMMUNICATIVE INTENT ===
1. FIRST-PERSON PERSPECTIVE: The incoming message was sent by the OTHER PERSON to the user. You are writing MY direct reply back to them.
2. ANSWER DIRECTLY: Respond to the communicative intent of the incoming message:
   - If they ask how you are (e.g. "kaisa hai?", "kaise ho?", "how are you?", "what's up?"):
     Directly answer your state first (e.g. "thik hu", "mast hu", "all good", "doing well").
   - When no specific intent or facts are supplied, provide a short, natural, neutral answer like "thik hu, tu bata" or "all good, what about you?".
   - A brief reciprocal question (e.g. "tu bata?", "tu suna?", "you?") is allowed ONLY AFTER directly answering first.
3. NEVER ECHO OR REPEAT:
   - Do NOT copy, translate, paraphrase, or ask the incoming question back as your reply.
   - For example: if incoming is "kaisa hai bhai?", NEVER reply with "kaisa hai bhai?" or "kaisi hai?".
4. NEVER INVENT FACTS:
   - Do NOT hallucinate personal plans, availability, promises, or commitments not specified in the intent.
   - If asked about availability/plans and no facts are provided, answer with a concise clarification or safe noncommittal response (e.g., "kal ka pakka nahi hai, shaam ko batata hu" or "dekhna padega, kitne baje?").
5. LANGUAGE & DIALECT MATCHING:
   - Always reply in the same language and dialect as the conversation (e.g., casual Hinglish for Hinglish, English for English, Hindi for Hindi).
6. PROPORTIONAL LENGTH:
   - For a simple greeting or "how are you?", keep the reply short (prefer 1 concise sentence, 3-8 words).
7. STYLE MUST NOT OVERRIDE ANSWERING:
   - Style profile informs tone, spelling, and emojis, but the primary job is to answer the message.

=== FEW-SHOT EXAMPLES ===
- Incoming: "kaise ho bhai?" | Intent: none
  Good Reply: "thik hu, tu bata"
  Bad Reply: "kaise ho bhai? kya chal raha hai?" (WRONG: echoes question instead of answering)

- Incoming: "kaisa hai bhai?" | Intent: none
  Good Reply: "mast hu bhai, tu suna"
  Bad Reply: "kaisi hai? aur kya hua?" (WRONG: rephrases the incoming question)

- Incoming: "kal mil raha hai?" | Intent: none (no availability info provided)
  Good Reply: "kal ka pakka nahi hai abhi, shaam ko batata hu"
  Bad Reply: "ha kal 5 baje milte hai cafe me" (WRONG: invented availability/plans)

- Incoming: "Are you coming to the party tonight?" | Intent: "Yes, around 9pm"
  Good Reply: "yeah will reach around 9pm!"
  Bad Reply: "I want to convey that I will reach around 9pm" (WRONG: restating intent instruction)

- Incoming: "How are you doing?" | Intent: none
  Good Reply: "all good here, how about you?"
  Bad Reply: "how are you doing today?" (WRONG: repeats incoming question)

=== PRIORITY 2: STYLE PROFILE GUIDANCE ===
- Intensity: {intensity_guidance}
- Capitalization: {case_instruction}
- Punctuation: {punctuation_instruction}
- Emojis: {emoji_instruction}
- Vocabulary: {vocab_instruction}
- Greetings / Openers (if initiating): {openers_str}
- Sign-offs: {closers_str}{sample_block}{custom_notes_block}

=== OUTPUT FORMAT ===
Output ONLY the requested draft reply options separated by delimiter headers.
Do not include any conversational preamble, quotes, explanations, or analysis.
"""
    return prompt.strip()


def build_user_prompt(request: DraftRequest) -> str:
    """
    Builds the user turn prompt detailing incoming message, conversation history,
    optional intent, desired length, and number of draft options requested.
    """
    length_guidelines = {
        "short": "Short (1 sentence or a quick 3-8 word response, typical for quick WhatsApp replies)",
        "medium": "Medium (1-2 sentences, natural chat conversational length)",
        "long": "Long (2-4 sentences, explanatory or detailed reply)",
    }.get(request.length_preference, "Medium (1-2 sentences)")

    history, incoming = extract_incoming_and_history(request.context)

    # If incoming couldn't be extracted, use raw context
    target_incoming = incoming if incoming else request.context.strip()

    history_block = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""

    if request.intent and request.intent.strip():
        intent_block = f"""USER INTENT / FACTS TO CONVEY:
{request.intent.strip()}
(Express this meaning directly in my voice. Do NOT restate or quote this instruction)."""
    else:
        intent_block = """USER INTENT:
(None provided — this field is optional. Infer a direct, natural answer from the conversation context. For a greeting or 'how are you?', give a simple positive/neutral answer like 'thik hu' or 'all good', and optionally a brief reciprocal ask like 'tu bata?'. Do NOT invent specific plans, commitments, or unknown facts)."""

    num_drafts = request.num_drafts

    prompt = f"""{history_block}LATEST INCOMING MESSAGE (from other person to me):
"{target_incoming}"

{intent_block}

TARGET LENGTH: {length_guidelines}
STYLE INTENSITY: {request.style_intensity}

INSTRUCTIONS:
Generate exactly {num_drafts} distinct draft reply options from my perspective answering the latest incoming message.
Every option must directly ANSWER the other person; do not repeat, echo, or rephrase their question.
Format your output strictly with delimiter headers like this:

=== DRAFT 1 ===
[draft 1 text]

=== DRAFT 2 ===
[draft 2 text]

Do not output any markdown formatting, bullet points, numbering, or introductory commentary outside the draft delimiters.
"""
    return prompt.strip()


def parse_draft_options(raw_output: str, expected_count: int, incoming_text: str = "") -> List[str]:
    """
    Parses the delimited draft options from the model's raw completion output.
    Gracefully handles varying delimiters or non-standard formatting, and ensures
    drafts do not merely echo the incoming message.
    """
    # Clean incoming text for comparison
    _, incoming_clean = extract_incoming_and_history(incoming_text) if incoming_text else ("", "")

    # Match === DRAFT N === or --- DRAFT N --- or DRAFT N:
    pattern = re.compile(r"(?:===|---|###)?\s*DRAFT\s*\d+\s*(?:===|---|:)?", re.IGNORECASE)
    raw_drafts: List[str] = []

    if pattern.search(raw_output):
        parts = pattern.split(raw_output)
        for part in parts:
            cleaned = part.strip()
            # Remove any lingering quotes around the draft
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            if cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1].strip()

            if cleaned and not cleaned.lower().startswith("here are") and not cleaned.lower().startswith("draft"):
                raw_drafts.append(cleaned)
    else:
        # Fallback: check for numbered lines (e.g. "1. Draft one\n2. Draft two")
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

    # Filter out echoes, duplicates, and clean up
    seen = set()
    filtered_drafts: List[str] = []

    for d in raw_drafts:
        d_clean = d.strip()
        if not d_clean:
            continue
        # Strip any meta headers like "Reply:" or "Me:"
        d_clean = re.sub(r"^(?:Me|Reply|Option\s*\d+):\s*", "", d_clean, flags=re.IGNORECASE).strip()

        # Check for duplicate
        key = d_clean.lower()
        if key in seen:
            continue

        # Check if draft merely echoes the incoming question without answering
        if incoming_clean and is_echo_of_incoming(d_clean, incoming_clean):
            continue

        seen.add(key)
        filtered_drafts.append(d_clean)

    # If all drafts were filtered out as echoes (rare edge case), provide a clean direct fallback
    if not filtered_drafts and incoming_clean:
        # If incoming is a greeting or 'kaisa hai'
        lowered = incoming_clean.lower()
        if any(w in lowered for w in ["kaisa", "kaise", "kaisi", "how are you", "wassup", "what's up", "whats up"]):
            filtered_drafts = ["thik hu, tu bata"]
        else:
            filtered_drafts = [raw_output.strip()]

    # If still empty, return the raw stripped output
    if not filtered_drafts and raw_output.strip():
        filtered_drafts = [raw_output.strip()]

    return filtered_drafts[:expected_count]
