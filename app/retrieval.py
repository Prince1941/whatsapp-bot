import re
import math
import httpx
from typing import List, Optional, Tuple, Dict
from app.models import ChatMessage, ChatPair
from app.storage import storage as default_storage


def clean_message_text(text: str) -> str:
    """Cleans message text by removing excess whitespace and non-printable characters."""
    return re.sub(r"\s+", " ", text).strip()


def extract_chat_pairs(messages: List[ChatMessage], target_user: str) -> List[ChatPair]:
    """
    Parses a sequence of ChatMessage objects into consecutive (incoming_message -> user_reply) pairs.
    Handles multiline messages, clusters consecutive turns, and ignores media and system messages.
    """
    pairs: List[ChatPair] = []
    if not messages or not target_user:
        return pairs

    # Normalize target user for case-insensitive matching
    norm_target = target_user.strip().lower()

    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]

        # Skip system or media messages
        if msg.is_system or msg.is_media or not msg.content.strip():
            i += 1
            continue

        # Check if this message is from an external sender (incoming message)
        sender_norm = msg.sender.strip().lower()
        if sender_norm != norm_target:
            other_sender = msg.sender.strip()
            incoming_parts = [msg.content.strip()]
            ts = msg.timestamp_str

            # Cluster any consecutive incoming messages from the same or other external senders
            i += 1
            while i < n:
                next_msg = messages[i]
                if next_msg.is_system or next_msg.is_media:
                    i += 1
                    continue
                next_sender_norm = next_msg.sender.strip().lower()
                if next_sender_norm != norm_target:
                    if next_msg.content.strip():
                        incoming_parts.append(next_msg.content.strip())
                    i += 1
                else:
                    break

            # Now look for the user's immediate reply
            reply_parts = []
            while i < n:
                next_msg = messages[i]
                if next_msg.is_system or next_msg.is_media:
                    i += 1
                    continue
                next_sender_norm = next_msg.sender.strip().lower()
                if next_sender_norm == norm_target:
                    if next_msg.content.strip():
                        reply_parts.append(next_msg.content.strip())
                    i += 1
                else:
                    break

            if incoming_parts and reply_parts:
                incoming_text = " \n ".join(incoming_parts)
                reply_text = " \n ".join(reply_parts)

                # Filter out obvious media placeholders, code blocks, or very long spam
                if len(incoming_text.split()) < 80 and len(reply_text.split()) < 80:
                    pairs.append(
                        ChatPair(
                            incoming=incoming_text,
                            reply=reply_text,
                            other_sender=other_sender,
                            timestamp_str=ts,
                        )
                    )
        else:
            i += 1

    return pairs


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def token_overlap_similarity(q: str, target: str) -> float:
    """Fallback lexical similarity using token overlap and character n-grams."""
    q_tokens = set(re.findall(r"\b[A-Za-z0-9]+\b", q.lower()))
    target_tokens = set(re.findall(r"\b[A-Za-z0-9]+\b", target.lower()))
    if not q_tokens or not target_tokens:
        return 0.0
    intersection = q_tokens & target_tokens
    union = q_tokens | target_tokens
    return len(intersection) / len(union)


def embed_texts(
    texts: List[str],
    base_url: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
    batch_size: int = 32,
    timeout: float = 30.0,
) -> List[Optional[List[float]]]:
    """
    Batches texts and queries Ollama's /api/embed endpoint.
    Returns a list of float vectors (or None for failed items).
    """
    if not texts:
        return []

    url = f"{base_url.rstrip('/')}/api/embed"
    results: List[Optional[List[float]]] = [None] * len(texts)

    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx : start_idx + batch_size]
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json={"model": model, "input": batch})
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    for i, emb in enumerate(embeddings):
                        results[start_idx + i] = emb
                else:
                    # Fallback single embedding if batching fails
                    for i, t in enumerate(batch):
                        sub_resp = client.post(url, json={"model": model, "input": t})
                        if sub_resp.status_code == 200:
                            sub_data = sub_resp.json()
                            results[start_idx + i] = sub_data.get("embeddings", [None])[0]
        except Exception:
            # Server offline or timeout: leave as None to allow fallback to token overlap
            pass

    return results


def index_chat_export_pairs(
    messages: List[ChatMessage],
    target_user: str,
    profile_id: str = "default",
    storage_instance=None,
    model: str = "nomic-embed-text",
    max_pairs: int = 500,
) -> int:
    """
    Extracts authentic pairs, embeds incoming messages via Ollama, and persists in SQLite.
    Limits to the most recent `max_pairs` to maintain fast indexing and retrieval.
    """
    s = storage_instance or default_storage
    pairs = extract_chat_pairs(messages, target_user)
    if not pairs:
        return 0

    # Pick the most recent pairs if more than max_pairs
    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]

    # Embed incoming messages
    incoming_texts = [p.incoming for p in pairs]
    embeddings = embed_texts(incoming_texts, model=model)

    # Save to storage
    s.delete_chat_pairs(profile_id)
    pairs_with_embs = list(zip(pairs, embeddings))
    s.save_chat_pairs(profile_id, pairs_with_embs)
    return len(pairs)


def search_similar_pairs(
    query: str,
    profile_id: str = "default",
    top_k: int = 6,
    contact_filter: Optional[str] = None,
    storage_instance=None,
    model: str = "nomic-embed-text",
) -> List[ChatPair]:
    """
    Retrieves the top-k most similar (incoming_message -> reply) pairs from storage.
    Uses cosine vector similarity when embeddings are available, boosting matches from the same contact.
    Falls back to token-overlap lexical similarity if embeddings are unavailable.
    """
    s = storage_instance or default_storage
    stored = s.get_chat_pairs(profile_id)
    if not stored:
        return []

    # Attempt to embed the query
    q_emb_list = embed_texts([query], model=model)
    q_vec = q_emb_list[0] if q_emb_list else None

    scored_pairs: List[Tuple[float, ChatPair]] = []
    contact_norm = contact_filter.strip().lower() if contact_filter else None

    for pair, vec in stored:
        score = 0.0
        if q_vec and vec:
            score = cosine_similarity(q_vec, vec)
        else:
            score = token_overlap_similarity(query, pair.incoming)

        # Contact match bonus (prefer pairs from same contact if present)
        if contact_norm and pair.other_sender and pair.other_sender.strip().lower() == contact_norm:
            score *= 1.2

        pair_copy = ChatPair(
            id=pair.id,
            profile_id=pair.profile_id,
            incoming=pair.incoming,
            reply=pair.reply,
            other_sender=pair.other_sender,
            timestamp_str=pair.timestamp_str,
            similarity_score=round(score, 4),
        )
        scored_pairs.append((score, pair_copy))

    # Sort descending by score
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate replies to ensure diverse few-shot examples
    seen_replies = set()
    results: List[ChatPair] = []
    for score, pair in scored_pairs:
        norm_r = pair.reply.strip().lower()
        if norm_r not in seen_replies and score > 0.15:
            seen_replies.add(norm_r)
            results.append(pair)
            if len(results) >= top_k:
                break

    # If too few above threshold, take top matches regardless of score
    if len(results) < min(3, len(scored_pairs)):
        for score, pair in scored_pairs:
            norm_r = pair.reply.strip().lower()
            if norm_r not in seen_replies:
                seen_replies.add(norm_r)
                results.append(pair)
                if len(results) >= top_k:
                    break

    return results
