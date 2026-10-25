from typing import List
from app.models import StyleProfile, DraftRequest


def build_system_prompt(profile: StyleProfile, intensity: str) -> str:
    """
    Builds the system prompt instructing the model on safety, anti-hallucination,
    intent priority, and stylistic constraints.
    """
    # Capitalization instructions
    if profile.capitalization.primary_case == "all_lowercase":
        case_instruction = "Use mostly lowercase letters (e.g., 'hey let me know when you arrive' or 'sounds good')."
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
        punct_notes.append("Use exclamation marks frequently for enthusiasm.")

    if profile.punctuation.question_freq > 0.4:
        punct_notes.append("Frequently ask engaged questions.")

    if profile.punctuation.ellipsis_freq > 0.2:
        punct_notes.append("Occasionally use ellipses ('...') for thought transitions.")

    punctuation_instruction = " ".join(punct_notes)

    # Emoji instructions
    if profile.emojis.top_emojis:
        top_emojis_str = " ".join(profile.emojis.top_emojis[:6])
        if profile.emojis.emoji_frequency > 0.8:
            emoji_instruction = f"Use emojis regularly (around 1-2 per message). Favorite emojis to favor: {top_emojis_str}."
        elif profile.emojis.emoji_frequency > 0.2:
            emoji_instruction = f"Use emojis moderately (e.g. 1 emoji every 1-2 messages). Preferred emojis: {top_emojis_str}."
        else:
            emoji_instruction = f"Use emojis sparingly. If used, prefer: {top_emojis_str}."
    else:
        emoji_instruction = "Do not use emojis unless specifically requested."

    # Slang & Vocabulary
    if profile.slang_and_phrases:
        slang_list = ", ".join(profile.slang_and_phrases[:12])
        vocab_instruction = f"Common colloquialisms and phrases that fit this style: {slang_list}."
    else:
        vocab_instruction = "Keep phrasing casual and friendly."

    # Openers and closers
    openers_str = ", ".join(profile.common_openers) if profile.common_openers else "hey, hi"
    closers_str = ", ".join(profile.common_closers) if profile.common_closers else "cya, thanks"

    # Style samples (if any)
    sample_block = ""
    if profile.sample_snippets:
        formatted_samples = "\n".join(f"- {s}" for s in profile.sample_snippets[:4])
        sample_block = (
            f"\nRepresentative short snippets of the rhythm (DO NOT copy these verbatim):\n{formatted_samples}"
        )

    intensity_guidance = {
        "subtle": "Apply stylistic traits subtly. Prioritize clear, polite, and neutral phrasing.",
        "moderate": "Balance stylistic habits naturally with clear communication.",
        "strong": "Emphasize the user's casual chat patterns, lowercase style, slang, and emojis closely.",
    }.get(intensity, "Balance stylistic habits naturally.")

    custom_notes_block = f"\nUser Custom Preferences: {profile.custom_notes}" if profile.custom_notes.strip() else ""

    prompt = f"""You are an AI writing assistant drafting a WhatsApp reply on behalf of a human user.
The drafts you produce will be reviewed, edited, and approved by the user before anything is sent.

CRITICAL PRIVACY & FACTUALITY RULES:
1. FACTUAL ACCURACY & INTENT COME FIRST: Express the exact intent and instructions provided by the user.
2. DO NOT INVENT FACTS: Never hallucinate personal commitments, appointments, memories, feelings, promises, or private details not mentioned in the context or intent. If details are missing, draft a natural clarifying reply or a safe neutral response.
3. NO VERBATIM COPYING: Never copy past training messages word-for-word. Adopt the cadence and style, not the specific content.
4. NO META-TALK: Output ONLY the reply drafts. Do not include introductory text, conversational fluff, explanations, or quotes.

STYLE PROFILE GUIDANCE:
- Intensity: {intensity_guidance}
- Capitalization: {case_instruction}
- Punctuation: {punctuation_instruction}
- Emojis: {emoji_instruction}
- Vocabulary: {vocab_instruction}
- Common greetings: {openers_str}
- Common sign-offs: {closers_str}{sample_block}{custom_notes_block}
"""
    return prompt.strip()


def build_user_prompt(request: DraftRequest) -> str:
    """
    Builds the user turn prompt detailing context, intent, desired length,
    and number of draft options requested.
    """
    length_guidelines = {
        "short": "Short (1 sentence or a quick 3-8 word response, typical for quick WhatsApp replies)",
        "medium": "Medium (1-2 sentences, natural chat conversational length)",
        "long": "Long (2-4 sentences, explanatory or detailed reply)",
    }.get(request.length_preference, "Medium (1-2 sentences)")

    intent_text = (
        request.intent.strip()
        if request.intent and request.intent.strip()
        else "Formulate a natural, context-appropriate reply that fits the conversation."
    )

    num_drafts = request.num_drafts

    prompt = f"""CONVERSATION CONTEXT:
---
{request.context.strip()}
---

USER INTENDED MESSAGE / GOAL:
{intent_text}

TARGET LENGTH: {length_guidelines}
STYLE INTENSITY: {request.style_intensity}

INSTRUCTIONS:
Generate exactly {num_drafts} distinct draft reply options matching the specified intent and style.
Format your output strictly with delimiter headers like this:

=== DRAFT 1 ===
[draft 1 text]

=== DRAFT 2 ===
[draft 2 text]

Do not output any markdown formatting, bullet points, numbering, or introductory commentary outside the draft delimiters.
"""
    return prompt.strip()


def parse_draft_options(raw_output: str, expected_count: int) -> List[str]:
    """
    Parses the delimited draft options from the model's raw completion output.
    Gracefully handles varying delimiters or non-standard formatting.
    """
    import re

    # Match === DRAFT N === or --- DRAFT N --- or DRAFT N:
    pattern = re.compile(r"(?:===|---|###)?\s*DRAFT\s*\d+\s*(?:===|---|:)?", re.IGNORECASE)
    drafts = []

    if pattern.search(raw_output):
        parts = pattern.split(raw_output)
        for part in parts:
            cleaned = part.strip()
            # Remove any lingering quotes around the draft
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            if cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1].strip()

            if cleaned and not cleaned.lower().startswith("here are"):
                drafts.append(cleaned)
    else:
        # Fallback: check for numbered lines (e.g. "1. Draft one\n2. Draft two")
        numbered_matches = re.split(r"(?:^|\n)\s*(?:\d+[\.\)]|Draft\s*\d+:?)\s*", raw_output.strip(), flags=re.IGNORECASE)
        numbered_candidates = [m.strip() for m in numbered_matches if m.strip()]
        if len(numbered_candidates) >= 2:
            drafts = numbered_candidates
        else:
            lines = [line.strip() for line in raw_output.strip().split("\n\n") if line.strip()]
            for line in lines:
                stripped = re.sub(r"^(?:\d+[\.\)]|Draft\s*\d+:?)\s*", "", line, flags=re.IGNORECASE).strip()
                if stripped:
                    drafts.append(stripped)

    # If still empty, return the raw stripped output
    if not drafts and raw_output.strip():
        drafts = [raw_output.strip()]

    return drafts[:expected_count]
