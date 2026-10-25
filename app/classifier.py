import re
from typing import Tuple, Optional
from pydantic import BaseModel


class IntentClassification(BaseModel):
    category: str  # activity_check, wellbeing, greeting, request, plan_making, yes_no_question, info_question, emotional, casual_chat, other
    confidence: float
    needs_user_facts: bool
    instruction: str


# Regex / keyword triggers for intent classification
ACTIVITY_TRIGGERS = [
    r"\bkya\s+(?:kar|kr|krta|karte|karega|karegi)\s+(?:rha|raha|rhe|rahe|rhi|rahi)\s*(?:h|hai|ho)?\b",
    r"\bkya\s+(?:kar|kr)\s+(?:rahe|raha|rha|rhe|rhi)\s*h?o?\b",
    r"\b(?:kaha|kahan|kidhar)\s*(?:h|hai|ho|pe\s*h?o?|par\s*h?o?)\b",
    r"\bwhere\s+(?:are\s+you|r\s+u)\b",
    r"\bwhat\s+(?:are\s+you|r\s+u)\s+doing\b",
    r"\bkya\s+(?:chal|chl)\s+(?:rha|raha)\s*(?:h|hai)?\b",
    r"\b(?:abhi\s+)?free\s+(?:hai|h|ho)\b",
]

WELLBEING_TRIGGERS = [
    r"\b(?:kaisa|kaise|kaisi)\s+(?:hai|h|ho)\b",
    r"\bhow\s+(?:are\s+you|r\s+u|is\s+it\s+going)\b",
    r"\bsab\s+(?:thik|theek|shi|sahi|badiya|badhiya)\s*(?:h|hai)?\b",
    r"\b(?:kya|kya\s+h)\s*haal\b",
    r"\bhaal\s*chal\b",
    r"\bwassup\b",
    r"\bwhats?\s*up\b",
    r"\btabiyat\s*(?:kaisi|kaisa|thik)\b",
]

GREETING_TRIGGERS = [
    r"^(?:hey|hello|hi|hii|hiii|yo|salam|assalam|namaste|oye|oyee)\b",
    r"\bgood\s+(?:morning|afternoon|evening|night)\b",
]

REQUEST_TRIGGERS = [
    r"\b(?:call\s+(?:kar|kr|utha|karo|uthana)|call\s+me)\b",
    r"\b(?:bhej\s+(?:de|do|na|dena)|send\s+(?:kar|kr|karo|me))\b",
    r"\b(?:share\s+(?:kar|kr|karo))\b",
    r"\b(?:help\s+(?:chahiye|kr|kar|karo|please))\b",
    r"\b(?:de\s+(?:de|do|na|dena))\b",
    r"\b(?:bata\s+(?:de|do|na|dena))\b",
    r"\b(?:sun\s+(?:na|kuch))\b",
    r"\b(?:ek\s+kaam\s+(?:kar|kr|karo))\b",
]

PLAN_TRIGGERS = [
    r"\b(?:milte\s+h|milna\s+h|milenge|kab\s+milna|kab\s+mile)\b",
    r"\b(?:chalega|chaloge|chale|chal\s+rahe|chal\s+rha)\s*(?:kya)?\b",
    r"\b(?:kal|shaam|aaj|weekend)\s+(?:free\s+h|ka\s+kya\s+plan|milna)\b",
    r"\b(?:bahar|movie|clg|college)\s+(?:chale|chalna)\b",
]

YES_NO_TRIGGERS = [
    r"\b(?:padh|study)\s*(?:liya|liye|raha|rha|li|diya)\s*(?:kya|\?)?",
    r"\b(?:kha(?:na|ya|ye|liya|liye))\s*(?:kya|\?)?",
    r"\b(?:so\s*(?:gaya|gye|raha|rha|ja))\s*(?:kya|\?)?",
    r"\b(?:pahuch|reach|aa)\s*(?:gaya|gaye|gya|gyee)\s*(?:kya|\?)?",
    r"\b(?:ho\s*(?:gaya|gya|gyee|gye)|kar\s*(?:diya|liya))\s*(?:kya|\?)?",
    r"\b(?:aa\s+rha\s+h\s+kya|aa\s+rahe\s+ho\s+kya)\b",
    r"\b(?:dekh\s+liya|dekh\s+liye)\s*(?:kya|\?)?",
]

INFO_TRIGGERS = [
    r"\b(?:kitne|kitna)\s*(?:baje|time|ka|rupaye|marks)\b",
    r"\b(?:kab|when)\s+(?:h|hai|aayega|hoga|chalna)\b",
    r"\b(?:konsa|kaun\s*sa|which)\b",
    r"\b(?:kya\s+hua|kya\s+bola|kya\s+hoga|why|kyu|kyun)\b",
]

EMOTIONAL_TRIGGERS = [
    r"\b(?:gussa\s+hu|bore\s+ho\s+rha|mood\s+off|tension\s+h|dar\s+lag\s+rha|sad\s+hu|khush\s+hu|maza\s+aa\s+gaya|dimaag\s+kharab)\b",
]


def classify_incoming_intent(incoming_text: str, user_intent: Optional[str] = "") -> IntentClassification:
    """
    Classifies the communicative intent of the incoming message turn.
    Returns IntentClassification with category, confidence, needs_user_facts, and generation instruction.
    """
    text = incoming_text.strip().lower()
    has_user_intent = bool(user_intent and user_intent.strip())

    if not text:
        return IntentClassification(
            category="other",
            confidence=0.5,
            needs_user_facts=False,
            instruction="Reply casually and naturally in casual Hinglish.",
        )

    # 1. Plan Making ("kal milte hai?", "kal free h?", "chalega kya?")
    if any(re.search(pat, text, re.IGNORECASE) for pat in PLAN_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message asks about plans or meeting. Reply faithfully using the specified intent."
        else:
            instr = (
                "Incoming message proposes plans or asks availability for later. "
                "Answer with a safe noncommittal response (e.g. 'kl ka pakka nahi h abhi', 'shaam ko batata hu', 'dekhna padega kitne baje'). "
                "Do not make specific commitments without intent."
            )
        return IntentClassification(
            category="plan_making",
            confidence=0.90,
            needs_user_facts=False,
            instruction=instr,
        )

    # 2. Activity Check ("kya kar rha h", "kaha h", "where are you", "what are you doing")
    if any(re.search(pat, text, re.IGNORECASE) for pat in ACTIVITY_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message is an activity check. Directly answer what you are doing using the specified intent."
        else:
            instr = (
                "Incoming message is an activity check ('what are you doing?' or 'where are you?'). "
                "Reply naturally stating what you are doing/busy with. Since no intent was provided, use a safe, "
                "casual generic Hinglish answer (e.g. 'kuch khas nhi, tu bata', 'kuch nhi bas ghar pe', 'chill kar rha hu'). "
                "NEVER invent fake tasks, places, times, or people."
            )
        return IntentClassification(
            category="activity_check",
            confidence=0.92,
            needs_user_facts=False,
            instruction=instr,
        )


    # 2. Wellbeing ("kaisa hai", "kaise ho", "sab thik?")
    if any(re.search(pat, text, re.IGNORECASE) for pat in WELLBEING_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message is a wellbeing inquiry. Directly answer with the state given in intent."
        else:
            instr = (
                "Incoming message is a wellbeing inquiry ('how are you?'). "
                "Directly answer your state first (e.g. 'thik hu bhai, tu bata', 'badhiya hu, tu suna', 'all good'). "
                "Prefer 1 short sentence (2-5 words)."
            )
        return IntentClassification(
            category="wellbeing",
            confidence=0.95,
            needs_user_facts=False,
            instruction=instr,
        )

    # 3. Request ("bhej de", "call kar", "send me")
    if any(re.search(pat, text, re.IGNORECASE) for pat in REQUEST_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message is a request/call. Respond directly acknowledging or fulfilling using the intent."
        else:
            instr = (
                "Incoming message is a request or call prompt. "
                "Acknowledge directly and concisely (e.g. 'ha karta hu', 'ha bhejta hu', 'ruk 5 min', 'ha kar call'). "
                "Do not invent fake excuses or promises."
            )
        return IntentClassification(
            category="request",
            confidence=0.90,
            needs_user_facts=False,
            instruction=instr,
        )

    # 4. Yes/No & Personal Completion Questions ("padh liya?", "khana khaya?", "so gaya?")
    if any(re.search(pat, text, re.IGNORECASE) for pat in YES_NO_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message is a yes/no factual question. Convey the exact factual answer from the intent in casual style."
            needs_facts = False
        else:
            # Missing intent for personal factual question requires fact
            instr = (
                "Incoming message asks whether you did/completed something. "
                "If no facts are available, indicate you need to check or give a noncommittal response, but DO NOT fabricate an answer."
            )
            needs_facts = True
        return IntentClassification(
            category="yes_no_question",
            confidence=0.90,
            needs_user_facts=needs_facts,
            instruction=instr,
        )

    # 5. Plan Making ("kal milte hai?", "chalega kya?")
    if any(re.search(pat, text, re.IGNORECASE) for pat in PLAN_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message asks about plans or meeting. Reply faithfully using the specified intent."
        else:
            instr = (
                "Incoming message proposes plans or asks availability. "
                "Answer with a safe noncommittal response (e.g. 'kl ka pakka nahi h abhi', 'shaam ko batata hu', 'dekhna padega kitne baje'). "
                "Do not make specific commitments without intent."
            )
        return IntentClassification(
            category="plan_making",
            confidence=0.88,
            needs_user_facts=False,
            instruction=instr,
        )

    # 6. Greeting ("hey", "hello", "hi")
    if any(re.search(pat, text, re.IGNORECASE) for pat in GREETING_TRIGGERS) and len(text.split()) <= 4:
        return IntentClassification(
            category="greeting",
            confidence=0.95,
            needs_user_facts=False,
            instruction="Incoming message is a simple greeting. Reply with a casual, short greeting (e.g. 'ha bol', 'hey', 'hello bhai').",
        )

    # 7. Information Question ("kitne baje", "konsa subject")
    if any(re.search(pat, text, re.IGNORECASE) for pat in INFO_TRIGGERS):
        if has_user_intent:
            instr = "Incoming message is an informational question. Answer with the fact provided in intent."
        else:
            instr = "Incoming message asks for specific information. If unknown, ask clarification or say you will check."
        return IntentClassification(
            category="info_question",
            confidence=0.85,
            needs_user_facts=not has_user_intent,
            instruction=instr,
        )

    # 8. Emotional sharing
    if any(re.search(pat, text, re.IGNORECASE) for pat in EMOTIONAL_TRIGGERS):
        return IntentClassification(
            category="emotional",
            confidence=0.85,
            needs_user_facts=False,
            instruction="Incoming message expresses emotion or venting. Reply supportively and casually in Hinglish.",
        )

    # 9. General / Casual Chat fallback
    return IntentClassification(
        category="casual_chat",
        confidence=0.70,
        needs_user_facts=False,
        instruction="Reply naturally in casual Hinglish adhering to the user's learned style and spelling conventions.",
    )
