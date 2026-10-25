import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.language import (
    count_devanagari_chars,
    count_latin_chars,
    has_devanagari,
    detect_language_and_script,
    resolve_target_language_script,
    validate_script,
)
from app.validator import validate_single_draft, repair_and_dedupe_options, apply_spelling_preferences
from app.prompt import build_system_prompt, build_user_prompt, parse_draft_options
from app.models import StyleProfile, DraftRequest


class TestLanguageDetectionAndResolution:
    def test_romanized_hinglish_intent_detection(self):
        detected = detect_language_and_script("thik hu bhai, tu bata")
        assert detected == "hinglish"

        target, reason = resolve_target_language_script("auto", intent="thik hu bhai")
        assert target == "hinglish"
        assert "Intent" in reason

    def test_devanagari_hindi_intent_detection(self):
        detected = detect_language_and_script("हाँ पढ़ लिया भाई")
        assert detected == "devanagari"

        target, reason = resolve_target_language_script("auto", intent="हाँ पढ़ लिया")
        assert target == "devanagari"
        assert "Intent" in reason

    def test_english_intent_detection(self):
        detected = detect_language_and_script("Yes I finished the assignment yesterday")
        assert detected == "english"

        target, reason = resolve_target_language_script("auto", intent="Yes I am free today")
        assert target == "english"
        assert "Intent" in reason

    def test_blank_intent_follows_incoming_message(self):
        # 1. Incoming in Devanagari with blank intent
        target_dev, reason_dev = resolve_target_language_script("auto", intent="", latest_incoming="खाना खा लिया क्या?")
        assert target_dev == "devanagari"
        assert "incoming message" in reason_dev

        # 2. Incoming in English with blank intent
        target_en, reason_en = resolve_target_language_script("auto", intent="", latest_incoming="Did you eat lunch yet?")
        assert target_en == "english"
        assert "incoming message" in reason_en

        # 3. Incoming in Hinglish with blank intent
        target_h, reason_h = resolve_target_language_script("auto", intent="", latest_incoming="khana kha liya kya?")
        assert target_h == "hinglish"
        assert "incoming message" in reason_h

    def test_explicit_setting_overrides_auto(self):
        target, reason = resolve_target_language_script("devanagari", intent="thik hu bhai", latest_incoming="kaisa hai")
        assert target == "devanagari"
        assert "Explicit setting" in reason

        target_en, _ = resolve_target_language_script("english", intent="thik hu bhai")
        assert target_en == "english"

        target_h, _ = resolve_target_language_script("hinglish", intent="ठीक हूँ")
        assert target_h == "hinglish"

    def test_mixed_language_input_dominant_script(self):
        # Dominant Devanagari with minor Latin token (e.g. "PDF पढ़ लिया")
        detected_dev = detect_language_and_script("हाँ PDF पढ़ लिया भाई")
        assert detected_dev == "devanagari"

        # Dominant Latin with single Hindi word
        detected_lat = detect_language_and_script("bro kal meeting me chalte hai")
        assert detected_lat == "hinglish"


class TestScriptValidation:
    def test_devanagari_validation(self):
        # Valid Devanagari
        valid, err = validate_script("हाँ पढ़ लिया", "devanagari")
        assert valid is True
        assert err is None

        # Invalid: Roman script when Devanagari expected
        valid, err = validate_script("haan padh liya bhai", "devanagari")
        assert valid is False
        assert "Expected Devanagari" in err

    def test_hinglish_validation(self):
        # Valid Hinglish (Roman)
        valid, err = validate_script("ha padh liya bhai", "hinglish")
        assert valid is True

        # Invalid: Devanagari when Hinglish expected
        valid, err = validate_script("हाँ भाई", "hinglish")
        assert valid is False
        assert "Devanagari characters" in err

    def test_english_validation(self):
        # Valid English
        valid, err = validate_script("Yes I already ate lunch", "english")
        assert valid is True

        # Invalid: Devanagari in English
        valid, err = validate_script("Yes I ate ठीक है", "english")
        assert valid is False


class TestValidatorAndPreservation:
    def test_intent_spelling_preservation_not_altered(self):
        profile = StyleProfile(
            profile_name="Test",
            sender_name="Me",
            spelling_preferences={"thik": "theek", "hu": "hoon", "hai": "h"},
        )
        # When user types "thik hu bhai", it must NOT be altered to "theek hoon bhai"
        drafts = repair_and_dedupe_options(
            raw_drafts=["theek hoon bhai"],
            expected_count=2,
            incoming="kaisa hai",
            intent_category="greeting_status",
            user_intent="thik hu bhai",
            profile=profile,
            target_script="hinglish",
        )
        # Position 0 must preserve exact user intent
        assert drafts[0] == "thik hu bhai"
        assert "ठीक" not in drafts[0]

    def test_no_transliteration_to_devanagari_when_hinglish_provided(self):
        profile = StyleProfile(profile_name="Test", sender_name="Me")
        drafts = repair_and_dedupe_options(
            raw_drafts=["thik hu bhai"],
            expected_count=1,
            incoming="kaisa hai?",
            intent_category="greeting_status",
            user_intent="thik hu bhai",
            profile=profile,
            target_script="hinglish",
        )
        assert drafts[0] == "thik hu bhai"
        assert not has_devanagari(drafts[0])

    def test_devanagari_intent_yields_devanagari_output(self):
        profile = StyleProfile(profile_name="Test", sender_name="Me")
        drafts = repair_and_dedupe_options(
            raw_drafts=["हाँ पढ़ लिया", "बिल्कुल पढ़ लिया"],
            expected_count=2,
            incoming="पढ़ लिया?",
            intent_category="activity_check",
            user_intent="हाँ पढ़ लिया",
            profile=profile,
            target_script="devanagari",
        )
        for d in drafts:
            assert has_devanagari(d)
            assert not any("a" <= ch.lower() <= "z" for ch in d)

    def test_devanagari_intent_semantic_rejection(self):
        # A negative draft contradicts affirmative Devanagari intent
        is_valid, reason = validate_single_draft(
            draft="नहीं पढ़ा अभी",
            incoming="पढ़ लिया?",
            intent_category="activity_check",
            user_intent="हाँ पढ़ लिया",
            target_script="devanagari",
        )
        assert is_valid is False
        assert "contradicts affirmative" in reason

    def test_english_intent_yields_english_output(self):
        profile = StyleProfile(profile_name="Test", sender_name="Me")
        drafts = repair_and_dedupe_options(
            raw_drafts=["Yes I completed it", "Done with it"],
            expected_count=2,
            incoming="Did you finish the work?",
            intent_category="activity_check",
            user_intent="Yes I finished it",
            profile=profile,
            target_script="english",
        )
        for d in drafts:
            assert not has_devanagari(d)
            assert "finished" in d or "completed" in d or "done" in d.lower()


class TestPromptScriptInstruction:
    def test_devanagari_prompt_contains_strict_instruction(self):
        profile = StyleProfile(
            profile_name="Test",
            sender_name="Me",
            spelling_preferences={"hai": "h", "theek": "thik"},
        )
        sys_prompt = build_system_prompt(
            profile=profile,
            intensity="moderate",
            incoming_text="पढ़ लिया?",
            user_intent="हाँ पढ़ लिया",
            target_script="devanagari",
        )
        # Must instruct Devanagari Hindi
        assert "DEVANAGARI HINDI" in sys_prompt
        # Must NOT include Romanized spelling rules block in Devanagari prompt
        assert "MANDATORY SPELLING CONVENTIONS FROM USER'S CHAT HISTORY" not in sys_prompt

    def test_hinglish_prompt_contains_hinglish_guidance(self):
        profile = StyleProfile(
            profile_name="Test",
            sender_name="Me",
            spelling_preferences={"hai": "h", "theek": "thik"},
        )
        sys_prompt = build_system_prompt(
            profile=profile,
            intensity="moderate",
            incoming_text="padh liya?",
            user_intent="ha padh liya",
            target_script="hinglish",
        )
        assert "ROMANIZED HINGLISH" in sys_prompt
        assert "MANDATORY SPELLING CONVENTIONS" in sys_prompt

    def test_user_prompt_demonstrations_match_script(self):
        req_dev = DraftRequest(
            context="पढ़ लिया?",
            intent="हाँ पढ़ लिया",
            language_script="devanagari",
        )
        user_prompt_dev = build_user_prompt(req_dev, target_script="devanagari")
        assert "DEVANAGARI SCRIPT" in user_prompt_dev
        assert "हाँ पढ़ लिया" in user_prompt_dev

        req_en = DraftRequest(
            context="Did you finish?",
            intent="Yes finished",
            language_script="english",
        )
        user_prompt_en = build_user_prompt(req_en, target_script="english")
        assert "ENGLISH" in user_prompt_en
        assert "Yes finished" in user_prompt_en


if __name__ == "__main__":
    classes = [
        TestLanguageDetectionAndResolution,
        TestScriptValidation,
        TestValidatorAndPreservation,
        TestPromptScriptInstruction,
    ]
    total = 0
    passed = 0
    for cls in classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                method = getattr(instance, name)
                if callable(method):
                    total += 1
                    try:
                        method()
                        passed += 1
                        print(f"PASS: {cls.__name__}.{name}")
                    except Exception as e:
                        print(f"FAIL: {cls.__name__}.{name}: {e}")
                        raise
    print(f"\n==========================================")
    print(f"RESULT: ALL {passed}/{total} TESTS PASSED SUCCESSFULLY!")
    print(f"==========================================")
