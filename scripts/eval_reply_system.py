import asyncio
import os
import re
import random
import time
from typing import List, Dict, Any, Tuple
from app.parser import parse_whatsapp_export
from app.retrieval import extract_chat_pairs, search_similar_pairs, index_chat_export_pairs
from app.classifier import classify_incoming_intent
from app.prompt import build_system_prompt, build_user_prompt, parse_draft_options
from app.providers import get_llm_provider
from app.storage import Storage
from app.models import DraftRequest, StyleProfile


def count_emojis(text: str) -> int:
    return len(re.findall(r"[\U00010000-\U0010ffff]", text))


def extract_slang_tokens(text: str, target_words: set) -> int:
    words = set(re.findall(r"\b[A-Za-z]+\b", text.lower()))
    return len(words & target_words)


async def evaluate_system(
    chat_file_path: str = "/Users/princekumar/Downloads/_chat.txt",
    user_name: str = "Prince Kr",
    model_name: str = "qwen2.5:7b",
    num_eval_pairs: int = 25,
):
    print("=" * 70)
    print("WHATSAPP ASSISTANT EVALUATION BENCHMARK")
    print(f"Model: {model_name} | Target User: {user_name}")
    print("=" * 70)

    s = Storage()
    profile = s.get_profile("default")
    if not profile:
        print("Error: Default profile not found in database.")
        return

    # 1. Parse and extract pairs
    if not os.path.exists(chat_file_path):
        print(f"File {chat_file_path} not found. Using stored database pairs.")
        stored = s.get_chat_pairs("default")
        all_pairs = [p for p, _ in stored]
    else:
        with open(chat_file_path, "r", encoding="utf-8") as f:
            parse_res = parse_whatsapp_export(f.read())
        all_pairs = extract_chat_pairs(parse_res.messages, user_name)

    print(f"Total authentic pairs extracted: {len(all_pairs)}")
    if len(all_pairs) < 100:
        print("Not enough pairs for 100 holdout, using available pairs.")
        holdout_pairs = all_pairs[-len(all_pairs)//2:]
        train_pairs = all_pairs[:-len(all_pairs)//2]
    else:
        # Hold out 100 pairs from the middle/end
        random.seed(42)
        shuffled = list(all_pairs)
        random.shuffle(shuffled)
        holdout_pairs = shuffled[:100]
        train_pairs = shuffled[100:]

    print(f"Train/Retrieval index pool: {len(train_pairs)} pairs")
    print(f"Holdout evaluation set: {len(holdout_pairs)} pairs")

    # Target slang words from user's spelling preferences
    slang_targets = set(profile.spelling_preferences.values()) | {"ha", "nhi", "h", "hu", "bhai", "kl", "acha"}

    # 2. Evaluate sample of holdout pairs
    eval_subset = holdout_pairs[:num_eval_pairs]
    results = []

    intent_matches = 0
    real_lengths = []
    gen_lengths = []
    real_slang_count = 0
    gen_slang_count = 0
    real_emoji_count = 0
    gen_emoji_count = 0

    provider = get_llm_provider("ollama", model_name=model_name)

    print(f"\nRunning evaluation on {len(eval_subset)} holdout conversations...")
    start_time = time.time()

    for idx, pair in enumerate(eval_subset, 1):
        inc = pair.incoming
        real_rep = pair.reply

        # Classify intent
        intent_res = classify_incoming_intent(inc, "")

        # Retrieve top 5 similar pairs (excluding exact match)
        retrieved = search_similar_pairs(inc, profile_id="default", top_k=5)
        retrieved_filtered = [p for p in retrieved if p.incoming.strip() != inc.strip()][:4]

        req = DraftRequest(context=inc, intent="", length_preference="short", style_intensity="moderate", num_drafts=2)
        sys_p = build_system_prompt(profile, req.style_intensity, inc, "")
        usr_p = build_user_prompt(req, profile=profile, retrieved_pairs=retrieved_filtered, detected_intent=intent_res.category, intent_instruction=intent_res.instruction)

        try:
            raw = await provider.generate_reply(sys_p, usr_p, temperature=0.65, max_tokens=60, format_json=True)
            drafts = parse_draft_options(raw, expected_count=2, incoming_text=inc, profile=profile, intent_category=intent_res.category)
            gen_rep = drafts[0] if drafts else "No draft"
        except Exception as e:
            gen_rep = f"Error: {e}"

        # Evaluate intent match
        gen_intent = classify_incoming_intent(gen_rep, "").category
        is_match = (gen_intent == intent_res.category) or (intent_res.category in ("activity_check", "wellbeing", "casual_chat") and gen_intent in ("activity_check", "wellbeing", "casual_chat"))
        if is_match:
            intent_matches += 1

        # Lengths
        real_w = len(real_rep.split())
        gen_w = len(gen_rep.split())
        real_lengths.append(real_w)
        gen_lengths.append(gen_w)

        # Slang & emojis
        real_slang_count += extract_slang_tokens(real_rep, slang_targets)
        gen_slang_count += extract_slang_tokens(gen_rep, slang_targets)
        real_emoji_count += count_emojis(real_rep)
        gen_emoji_count += count_emojis(gen_rep)

        results.append({
            "incoming": inc.replace("\n", " ")[:40],
            "real_reply": real_rep.replace("\n", " ")[:35],
            "gen_reply": gen_rep.replace("\n", " ")[:35],
            "intent": intent_res.category,
        })
        print(f"[{idx}/{len(eval_subset)}] Tested: \"{inc[:25]}...\" -> Gen: \"{gen_rep[:25]}...\"")

    total_eval_time = time.time() - start_time
    intent_match_rate = (intent_matches / len(eval_subset)) * 100
    avg_real_len = sum(real_lengths) / len(real_lengths)
    avg_gen_len = sum(gen_lengths) / len(gen_lengths)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS REPORT")
    print("=" * 70)
    print(f"Intent Match Rate: {intent_match_rate:.1f}%")
    print(f"Average Words/Reply: Real = {avg_real_len:.1f} | Generated = {avg_gen_len:.1f}")
    print(f"Learned Slang/Spelling Tokens Used: Real = {real_slang_count} | Generated = {gen_slang_count}")
    print(f"Total Emojis: Real = {real_emoji_count} | Generated = {gen_emoji_count}")
    print(f"Avg Latency per generation: {total_eval_time/len(eval_subset):.2f}s")

    print("\n" + "=" * 70)
    print("SAMPLE EVALUATION TABLE (Holdout Set)")
    print("=" * 70)
    print(f"{'Incoming Message':<35} | {'My Real Reply':<30} | {'Generated Reply':<30} | {'Intent':<15}")
    print("-" * 115)
    for r in results[:10]:
        print(f"{r['incoming']:<35} | {r['real_reply']:<30} | {r['gen_reply']:<30} | {r['intent']:<15}")

    # 3. Hand Tests
    print("\n" + "=" * 70)
    print("HAND TESTS (Specified Test Cases)")
    print("=" * 70)
    hand_tests = [
        ("kya kar rha h?", ""),
        ("kaha h?", ""),
        ("khana khaya?", ""),
        ("aa rha h kya?", ""),
        ("kal free h?", ""),
        ("bhai call utha", ""),
        ("sab thik?", ""),
        ("kya haal", ""),
    ]

    hand_results = []
    for q, intent in hand_tests:
        i_res = classify_incoming_intent(q, intent)
        ret = search_similar_pairs(q, profile_id="default", top_k=5)
        req = DraftRequest(context=q, intent=intent, length_preference="short", style_intensity="moderate", num_drafts=2)
        sys_p = build_system_prompt(profile, req.style_intensity, q, intent)
        usr_p = build_user_prompt(req, profile=profile, retrieved_pairs=ret, detected_intent=i_res.category, intent_instruction=i_res.instruction)

        try:
            raw = await provider.generate_reply(sys_p, usr_p, temperature=0.65, max_tokens=60, format_json=True)
            drafts = parse_draft_options(raw, expected_count=2, incoming_text=q, profile=profile, intent_category=i_res.category)
            reply_opt = drafts[0] if drafts else "No draft"
            reply_opt2 = drafts[1] if len(drafts) > 1 else ""
        except Exception as e:
            reply_opt = f"Error: {e}"
            reply_opt2 = ""

        hand_results.append((q, i_res.category, reply_opt, reply_opt2))

    print(f"{'Hand Query':<20} | {'Intent Category':<18} | {'Draft 1':<30} | {'Draft 2':<30}")
    print("-" * 105)
    for q, cat, d1, d2 in hand_results:
        print(f"{q:<20} | {cat:<18} | {d1:<30} | {d2:<30}")
    print("=" * 105)


if __name__ == "__main__":
    asyncio.run(evaluate_system())
