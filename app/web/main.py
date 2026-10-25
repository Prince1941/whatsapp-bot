import os
import re
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import config
from app.models import (
    StyleProfile,
    PrivacyFilterConfig,
    DraftRequest,
    DraftResponse,
    OllamaConfig,
)
from app.parser import parse_whatsapp_export
from app.privacy import PrivacyFilter
from app.analyzer import build_style_profile
from app.prompt import (
    build_system_prompt,
    build_user_prompt,
    parse_draft_options,
    is_factual_personal_question,
    extract_incoming_and_history,
)
from app.retrieval import search_similar_pairs, index_chat_export_pairs
from app.classifier import classify_incoming_intent
from app.providers import get_llm_provider
from app.ollama import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from app.storage import storage
from app.language import resolve_target_language_script, validate_script


app = FastAPI(title="WhatsApp Style AI Writing Assistant", version="1.0.0")

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
(TEMPLATES_DIR / "partials").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory transient store for raw upload parsed session (avoiding writing raw file to disk)
# session_id -> { "text": str, "parse_result": ParseResult }
UPLOAD_SESSIONS = {}


def get_ollama_client() -> OllamaClient:
    cfg = storage.get_ollama_config()
    return OllamaClient(cfg)


def is_model_available(target_model: str, available_models: List[str]) -> bool:
    """Checks if target_model exists in available_models (exact or base name match)."""
    if not target_model or not available_models:
        return False
    target_clean = target_model.strip().lower()
    for m in available_models:
        m_clean = m.strip().lower()
        if target_clean == m_clean:
            return True
        if target_clean.split(":")[0] == m_clean.split(":")[0]:
            return True
    return False


def sync_and_validate_ollama_model(
    ollama_cfg: OllamaConfig,
    ollama_status: OllamaStatus,
) -> Tuple[OllamaConfig, bool, Optional[str]]:
    """
    Validates that ollama_cfg.model is present in OllamaStatus.available_models.
    If not installed and other models are available, auto-selects the best installed
    chat model (preferring qwen2.5:7b or first non-embedding model), saves it,
    and returns a warning.
    """
    if not ollama_status.is_connected:
        return ollama_cfg, False, None

    available = ollama_status.available_models
    if not available:
        return ollama_cfg, True, f"No models installed in Ollama. Run 'ollama pull {ollama_cfg.model}'."

    if is_model_available(ollama_cfg.model, available):
        return ollama_cfg, False, None

    # Model is missing! Find best available model
    old_model = ollama_cfg.model
    preferred = [m for m in available if "qwen" in m.lower() and "embed" not in m.lower()]
    chat_models = [m for m in available if "embed" not in m.lower()]

    if preferred:
        chosen = preferred[0]
    elif chat_models:
        chosen = chat_models[0]
    else:
        chosen = available[0]

    ollama_cfg.model = chosen
    storage.save_ollama_config(ollama_cfg)
    warning = f"Model '{old_model}' was not found in Ollama. Auto-switched to installed model '{chosen}'."
    return ollama_cfg, False, warning


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders the main dashboard page."""
    profile = storage.get_profile()
    ollama_cfg = storage.get_ollama_config()
    priv_cfg = storage.get_privacy_config()
    client = OllamaClient(ollama_cfg)
    ollama_status = await client.check_health()
    ollama_cfg, model_not_found, warning = sync_and_validate_ollama_model(ollama_cfg, ollama_status)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "profile": profile,
            "ollama_cfg": ollama_cfg,
            "priv_cfg": priv_cfg,
            "ollama_status": ollama_status,
            "model_not_found": model_not_found,
            "warning_message": warning,
        },
    )


@app.get("/api/ollama/status", response_class=HTMLResponse)
async def check_ollama_status(request: Request):
    """Pings Ollama and returns an updated status badge."""
    client = get_ollama_client()
    ollama_status = await client.check_health()
    ollama_cfg = storage.get_ollama_config()
    ollama_cfg, model_not_found, warning = sync_and_validate_ollama_model(ollama_cfg, ollama_status)

    return templates.TemplateResponse(
        request=request,
        name="partials/ollama_status.html",
        context={
            "ollama_status": ollama_status,
            "ollama_cfg": ollama_cfg,
            "model_not_found": model_not_found,
            "warning_message": warning,
        },
    )


@app.post("/api/chat/preview", response_class=HTMLResponse)
async def preview_chat_export(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Accepts an uploaded WhatsApp export (.txt), validates size, parses senders and counts,
    and returns a preview without saving anything permanently to disk.
    """
    if not file.filename.lower().endswith((".txt", ".text")):
        return HTMLResponse(
            "<div class='alert alert-error'>Please upload a valid WhatsApp text export file (.txt).</div>",
            status_code=400,
        )

    content_bytes = await file.read()
    if len(content_bytes) > config.max_upload_bytes:
        return HTMLResponse(
            f"<div class='alert alert-error'>File size ({len(content_bytes)/(1024*1024):.1f} MB) exceeds maximum allowed size of {config.max_upload_bytes/(1024*1024):.0f} MB.</div>",
            status_code=400,
        )

    # Decode with fallback encoding
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode("latin-1")
        except Exception:
            return HTMLResponse(
                "<div class='alert alert-error'>Failed to decode file encoding. WhatsApp text exports should be UTF-8.</div>",
                status_code=400,
            )

    parse_result = parse_whatsapp_export(text)

    if not parse_result.senders:
        return HTMLResponse(
            "<div class='alert alert-warning'>No messages or participants could be detected in this file. Please verify the WhatsApp export format.</div>",
            status_code=400,
        )

    import uuid
    session_id = str(uuid.uuid4())
    UPLOAD_SESSIONS[session_id] = {
        "text": text,
        "parse_result": parse_result,
    }

    # Clean up old upload sessions if too many
    if len(UPLOAD_SESSIONS) > 10:
        oldest_key = next(iter(UPLOAD_SESSIONS))
        del UPLOAD_SESSIONS[oldest_key]

    return templates.TemplateResponse(
        request=request,
        name="partials/chat_preview.html",
        context={
            "session_id": session_id,
            "filename": file.filename,
            "result": parse_result,
        },
    )


@app.post("/api/profile/build", response_class=HTMLResponse)
async def build_profile(
    request: Request,
    session_id: str = Form(...),
    sender_name: str = Form(...),
    profile_name: str = Form("My WhatsApp Style"),
    redact_phones: bool = Form(True),
    redact_emails: bool = Form(True),
    redact_urls: bool = Form(True),
    redact_financial: bool = Form(True),
    filter_action: str = Form("redact"),  # "redact" or "drop"
    custom_terms: str = Form(""),
    save_raw: bool = Form(False),
):
    """
    Builds style profile from user's selected sender identity.
    Sanitizes PII, runs stylistic extraction, saves profile to SQLite,
    and cleans in-memory raw session.
    """
    session = UPLOAD_SESSIONS.get(session_id)
    if not session:
        return HTMLResponse(
            "<div class='alert alert-error'>Upload session expired. Please re-upload your chat export.</div>",
            status_code=400,
        )

    banned_list = [t.strip() for t in custom_terms.split(",") if t.strip()]
    priv_cfg = PrivacyFilterConfig(
        redact_phone_numbers=redact_phones,
        redact_emails=redact_emails,
        redact_urls=redact_urls,
        redact_financial=redact_financial,
        custom_banned_terms=banned_list,
        action=filter_action if filter_action in ("redact", "drop") else "redact",
    )

    p_filter = PrivacyFilter(priv_cfg)
    user_messages = p_filter.filter_messages(session["parse_result"].messages, sender_name)

    if not user_messages:
        return HTMLResponse(
            f"<div class='alert alert-warning'>No usable messages found for sender '{sender_name}' after privacy filtering.</div>",
            status_code=400,
        )

    # Build the profile
    profile = build_style_profile(
        messages=user_messages,
        sender_name=sender_name,
        profile_name=profile_name.strip() or "My WhatsApp Style",
    )

    # Save to SQLite
    storage.save_profile(profile)
    storage.save_privacy_config(priv_cfg)

    # Index authentic chat pairs for few-shot retrieval
    try:
        index_chat_export_pairs(
            session["parse_result"].messages,
            target_user=sender_name,
            profile_id=profile.id,
        )
    except Exception:
        pass

    # Handle opt-in raw storage
    if save_raw:
        storage.save_messages_opt_in([m.model_dump() for m in user_messages])

    # Clean session
    UPLOAD_SESSIONS.pop(session_id, None)


    return templates.TemplateResponse(
        request=request,
        name="partials/profile_view.html",
        context={
            "profile": profile,
            "success_message": f"Successfully built style profile from {len(user_messages)} messages!",
        },
    )


@app.get("/api/profile/view", response_class=HTMLResponse)
async def view_profile(request: Request):
    """Returns the current profile HTML component."""
    profile = storage.get_profile()
    return templates.TemplateResponse(
        request=request,
        name="partials/profile_view.html",
        context={
            "profile": profile,
        },
    )


@app.post("/api/profile/update", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    profile_name: str = Form(...),
    custom_notes: str = Form(""),
    slang_csv: str = Form(""),
    emojis_str: str = Form(""),
    snippets_text: str = Form(""),
    spelling_prefs_text: str = Form(""),
    phrases_text: str = Form(""),
):
    """Allows manual editing of profile parameters."""
    profile = storage.get_profile()
    if not profile:
        return HTMLResponse(
            "<div class='alert alert-warning'>No active profile to update.</div>",
            status_code=400,
        )

    profile.profile_name = profile_name.strip()
    profile.custom_notes = custom_notes.strip()

    # Parse slang
    slang_list = [s.strip() for s in slang_csv.split(",") if s.strip()]
    profile.slang_and_phrases = slang_list

    # Parse emojis
    emoji_list = [e.strip() for e in emojis_str.split() if e.strip()]
    profile.emojis.top_emojis = emoji_list

    # Parse sample snippets
    snippets = [line.strip() for line in snippets_text.splitlines() if line.strip()]
    profile.sample_snippets = snippets

    # Parse spelling preferences: "theek=thik, hai=h, hoon=hu" or newline separated
    new_prefs = {}
    for item in re.split(r"[,\n]", spelling_prefs_text):
        if "=" in item:
            k, v = item.split("=", 1)
            if k.strip() and v.strip():
                new_prefs[k.strip().lower()] = v.strip().lower()
    profile.spelling_preferences = new_prefs

    # Parse frequent phrases
    phrases = [line.strip() for line in phrases_text.splitlines() if line.strip()]
    if phrases:
        profile.frequent_short_phrases = phrases

    storage.save_profile(profile)

    return templates.TemplateResponse(
        request=request,
        name="partials/profile_view.html",
        context={
            "profile": profile,
            "success_message": "Style profile successfully updated!",
        },
    )


@app.post("/api/profile/delete", response_class=HTMLResponse)
async def delete_profile(request: Request):
    """Deletes the active profile."""
    storage.delete_profile("default")
    return templates.TemplateResponse(
        request=request,
        name="partials/profile_view.html",
        context={
            "profile": None,
            "info_message": "Style profile has been removed.",
        },
    )


@app.post("/api/draft", response_class=HTMLResponse)
async def draft_replies(
    request: Request,
    context: str = Form(...),
    intent: Optional[str] = Form(""),
    length_preference: str = Form("medium"),
    style_intensity: str = Form("moderate"),
    num_drafts: int = Form(2),
    model_name: Optional[str] = Form(None),
    provider: Optional[str] = Form("ollama"),
    language_script: str = Form("auto"),
):
    """
    Drafts replies using locally running Ollama model (or configured provider),
    guided by retrieved authentic few-shot pairs and detected communicative intent.
    """
    if not context.strip():
        return HTMLResponse(
            "<div class='alert alert-error'>Please provide the received message or conversation context.</div>",
            status_code=400,
        )

    if len(context) > config.max_context_chars:
        return HTMLResponse(
            f"<div class='alert alert-error'>Context exceeds maximum limit ({config.max_context_chars} chars).</div>",
            status_code=400,
        )

    clean_lang_script = language_script.strip().lower() if language_script else "auto"
    if clean_lang_script not in ("auto", "hinglish", "devanagari", "english"):
        clean_lang_script = "auto"

    draft_req = DraftRequest(
        context=context,
        intent=intent,
        length_preference=length_preference if length_preference in ("short", "medium", "long") else "medium",
        style_intensity=style_intensity if style_intensity in ("subtle", "moderate", "strong") else "moderate",
        num_drafts=max(1, min(5, num_drafts)),
        model_name=model_name.strip() if model_name and model_name.strip() else None,
        provider=provider.strip() if provider and provider.strip() else "ollama",
        language_script=clean_lang_script,
    )

    profile = storage.get_profile()
    if not profile:
        profile = StyleProfile(
            profile_name="Generic Friendly Style",
            sender_name="Me",
            custom_notes="Casual, friendly WhatsApp reply.",
        )

    client = get_ollama_client()
    target_model = draft_req.model_name or client.config.model

    history, incoming = extract_incoming_and_history(draft_req.context)
    target_incoming = incoming if incoming else draft_req.context.strip()

    # Resolve target language and script (Intent first, then incoming message)
    target_script, script_reason = resolve_target_language_script(
        draft_req.language_script,
        intent=draft_req.intent or "",
        latest_incoming=target_incoming,
    )
    # Log non-sensitive diagnostics only (never private chat text)
    print(f"[LANGUAGE/SCRIPT] Mode: {draft_req.language_script} -> Selected: {target_script} ({script_reason})")

    # Detect contact if available in conversation context
    contact_name = None
    for line in (history + "\n" + draft_req.context).splitlines():
        line_clean = line.strip()
        m = re.match(r"^([^:\n]{1,30}):\s*", line_clean)
        if m and m.group(1).lower() not in ("me", "you"):
            contact_name = m.group(1).strip()
            break

    # 1. Intent Classification
    intent_res = classify_incoming_intent(target_incoming, draft_req.intent or "")

    # 2. Check if factual personal question requires explicit fact
    requires_fact, fact_topic = is_factual_personal_question(target_incoming, history)
    if (requires_fact or intent_res.needs_user_facts) and not (draft_req.intent and draft_req.intent.strip()):
        topic_desc = fact_topic or intent_res.category.replace("_", " ")
        response_data = DraftResponse(
            drafts=[
                "Please specify your answer in the Intent field (e.g. 'haan padh liya', 'nahi padha', or 'thoda baaki hai') so the assistant doesn't guess."
            ],
            model_used=target_model,
            confidence_level=profile.confidence_level,
            detected_intent=intent_res.category,
            needs_user_input=True,
            user_input_prompt=f"The incoming message asks a personal factual question ({topic_desc}). Please provide your factual answer in the Intent field above to avoid guessing.",
            language_script_used=target_script,
        )
        return templates.TemplateResponse(
            request=request,
            name="partials/draft_results.html",
            context={
                "result": response_data,
                "context": context,
                "intent": intent,
                "profile": profile,
            },
        )

    # 3. Retrieve authentic few-shot pairs
    retrieved_pairs = search_similar_pairs(
        target_incoming,
        profile_id=profile.id,
        top_k=6,
        contact_filter=contact_name,
    )

    # 4. Construct prompts with explicit target_script
    system_prompt = build_system_prompt(
        profile,
        draft_req.style_intensity,
        incoming_text=draft_req.context,
        user_intent=draft_req.intent or "",
        target_script=target_script,
    )
    user_prompt = build_user_prompt(
        draft_req,
        profile=profile,
        retrieved_pairs=retrieved_pairs,
        detected_intent=intent_res.category,
        intent_instruction=intent_res.instruction,
        target_script=target_script,
    )

    try:
        max_tokens = 60 if draft_req.length_preference == "short" else 140
        temp = 0.5 if draft_req.style_intensity == "strong" else 0.65

        is_cloud = False
        raw_output = ""

        # Generation with retry loop (max 2 retries)
        for attempt in range(2):
            try:
                if draft_req.provider and draft_req.provider != "ollama":
                    llm = get_llm_provider(draft_req.provider, model_name=target_model)
                    raw_output = await llm.generate_reply(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temp,
                        max_tokens=max_tokens,
                        format_json=True,
                    )
                    is_cloud = llm.is_cloud
                else:
                    raw_output = await client.generate_chat(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=target_model,
                        temperature=temp,
                        max_tokens=max_tokens,
                        format_json=True,
                    )
                    is_cloud = False

                drafts = parse_draft_options(
                    raw_output,
                    draft_req.num_drafts,
                    incoming_text=draft_req.context,
                    profile=profile,
                    user_intent=draft_req.intent or "",
                    intent_category=intent_res.category,
                    length_preference=draft_req.length_preference,
                    target_script=target_script,
                )

                if drafts and drafts != ["Unable to generate draft. Please try again."]:
                    break
            except (OllamaModelNotFoundError, OllamaConnectionError):
                raise
            except Exception as e:
                print(f"[DRAFT ATTEMPT {attempt} ERROR] {e}")
                if attempt == 1:
                    raise
                temp = max(0.3, temp - 0.15)

        # 5. Validate generated drafts against selected script
        invalid_script = [d for d in drafts if not validate_script(d, target_script)[0]]
        if invalid_script:
            print(f"[SCRIPT MISMATCH] Output script invalid for target '{target_script}'. Retrying with corrective prompt...")
            corrective_user_prompt = (
                f"{user_prompt}\n\n"
                f"CRITICAL CORRECTION: You generated output in the WRONG script! "
                f"You MUST generate all reply drafts strictly in {target_script.upper()} script. "
                f"Do NOT use any other script. Output only the options in {target_script.upper()}."
            )
            try:
                if draft_req.provider and draft_req.provider != "ollama":
                    llm = get_llm_provider(draft_req.provider, model_name=target_model)
                    retry_raw = await llm.generate_reply(
                        system_prompt=system_prompt,
                        user_prompt=corrective_user_prompt,
                        temperature=0.3,
                        max_tokens=max_tokens,
                        format_json=True,
                    )
                else:
                    retry_raw = await client.generate_chat(
                        system_prompt=system_prompt,
                        user_prompt=corrective_user_prompt,
                        model=target_model,
                        temperature=0.3,
                        max_tokens=max_tokens,
                        format_json=True,
                    )

                retry_drafts = parse_draft_options(
                    retry_raw,
                    draft_req.num_drafts,
                    incoming_text=draft_req.context,
                    profile=profile,
                    user_intent=draft_req.intent or "",
                    intent_category=intent_res.category,
                    length_preference=draft_req.length_preference,
                    target_script=target_script,
                )
                still_invalid = [d for d in retry_drafts if not validate_script(d, target_script)[0]]
                if not still_invalid and retry_drafts:
                    drafts = retry_drafts
                else:
                    # Still invalid after retry: show clear error allowing regeneration
                    return HTMLResponse(
                        f"""
                        <div class='alert alert-warning' style='border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.15); padding: 1.25rem; border-radius: 8px; color: #fef08a;'>
                            <strong style='font-size: 1.05rem;'>⚠️ Script Generation Mismatch</strong>
                            <p style='margin: 0.5rem 0 0.8rem 0; font-size: 0.9rem; line-height: 1.4;'>
                                The model was unable to generate drafts matching the requested <strong>{target_script.title()}</strong> script.
                            </p>
                            <div style='display: flex; gap: 0.5rem;'>
                                <button type='button' class='btn btn-secondary btn-sm' onclick='document.getElementById("draft-form").dispatchEvent(new Event("submit"))'>
                                    Regenerate Draft
                                </button>
                            </div>
                        </div>
                        """,
                        status_code=422,
                    )
            except Exception as retry_err:
                print(f"[RETRY ERROR] {retry_err}")

        # Track which authentic learned phrases were applied
        applied = []
        if profile and profile.frequent_short_phrases:
            for p in profile.frequent_short_phrases:
                for d in drafts:
                    if p.lower() in d.lower() and p not in applied:
                        applied.append(p)

        retrieved_summary = [
            f'"{p.incoming}" -> "{p.reply}"' for p in retrieved_pairs[:4]
        ]

        response_data = DraftResponse(
            drafts=drafts,
            model_used=target_model,
            confidence_level=profile.confidence_level,
            learned_phrases_applied=applied[:3],
            detected_intent=intent_res.category,
            retrieved_examples=retrieved_summary,
            is_cloud=is_cloud,
            language_script_used=target_script,
        )

        return templates.TemplateResponse(
            request=request,
            name="partials/draft_results.html",
            context={
                "result": response_data,
                "context": context,
                "intent": intent,
                "profile": profile,
            },
        )


    except OllamaConnectionError as e:
        print(f"[API ERROR] OllamaConnectionError: {e}")
        return HTMLResponse(
            f"""
            <div class='alert alert-error' style='border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.15); padding: 1.25rem; border-radius: 8px; color: #fca5a5;'>
                <strong style='font-size: 1.05rem;'>⚠️ Ollama Connection Error</strong>
                <p style='margin: 0.5rem 0 0.8rem 0; font-size: 0.9rem; line-height: 1.4;'>{e}</p>
                <div style='background: rgba(0,0,0,0.4); padding: 0.6rem 0.8rem; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #ffffff;'>
                    Run in your terminal: <strong>ollama serve</strong>
                </div>
            </div>
            """,
            status_code=503,
        )
    except OllamaModelNotFoundError as e:
        print(f"[API ERROR] OllamaModelNotFoundError: {e}")
        return HTMLResponse(
            f"""
            <div class='alert alert-warning' style='border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.15); padding: 1.25rem; border-radius: 8px; color: #fef08a;'>
                <strong style='font-size: 1.05rem;'>⚠️ Model Not Found</strong>
                <p style='margin: 0.5rem 0 0.8rem 0; font-size: 0.9rem; line-height: 1.4;'>{e}</p>
                <div style='background: rgba(0,0,0,0.4); padding: 0.6rem 0.8rem; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #ffffff;'>
                    Run in your terminal: <strong>ollama pull {target_model}</strong> or switch to an installed model in Settings.
                </div>
            </div>
            """,
            status_code=404,
        )
    except OllamaTimeoutError as e:
        print(f"[API ERROR] OllamaTimeoutError: {e}")
        return HTMLResponse(
            f"""
            <div class='alert alert-error' style='border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.15); padding: 1.25rem; border-radius: 8px; color: #fca5a5;'>
                <strong style='font-size: 1.05rem;'>⏱️ Request Timeout</strong>
                <p style='margin: 0.5rem 0 0.8rem 0; font-size: 0.9rem; line-height: 1.4;'>{e}</p>
                <span style='font-size: 0.85rem; opacity: 0.9;'>7B models may take longer on initial load. You can increase the timeout in the Settings tab.</span>
            </div>
            """,
            status_code=504,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            f"""
            <div class='alert alert-error' style='border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.15); padding: 1.25rem; border-radius: 8px; color: #fca5a5;'>
                <strong style='font-size: 1.05rem;'>❌ Draft Generation Failed</strong>
                <p style='margin: 0.5rem 0 0.8rem 0; font-size: 0.9rem; line-height: 1.4;'>{str(e)}</p>
                <span style='font-size: 0.85rem;'>Check console output for detailed request and response logs.</span>
            </div>
            """,
            status_code=500,
        )


@app.post("/api/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    ollama_url: str = Form(...),
    ollama_model: str = Form(...),
    ollama_timeout: float = Form(120.0),
    custom_banned_terms: str = Form(""),
):
    """Updates Ollama and privacy configuration."""
    cfg = OllamaConfig(
        base_url=ollama_url.strip(),
        model=ollama_model.strip(),
        timeout_seconds=max(5.0, ollama_timeout),
    )
    storage.save_ollama_config(cfg)

    # Update privacy banned terms
    priv_cfg = storage.get_privacy_config()
    banned = [t.strip() for t in custom_banned_terms.split(",") if t.strip()]
    priv_cfg.custom_banned_terms = banned
    storage.save_privacy_config(priv_cfg)

    # Recheck health and validate model
    client = OllamaClient(cfg)
    status = await client.check_health()
    cfg, model_not_found, warning = sync_and_validate_ollama_model(cfg, status)

    return templates.TemplateResponse(
        request=request,
        name="partials/ollama_status.html",
        context={
            "ollama_status": status,
            "ollama_cfg": cfg,
            "model_not_found": model_not_found,
            "warning_message": warning,
            "saved_message": "Settings saved successfully!",
        },
    )


@app.post("/api/purge", response_class=HTMLResponse)
async def purge_all_data(request: Request):
    """Permanently deletes all profiles, configuration, and data."""
    storage.purge_all_data()
    UPLOAD_SESSIONS.clear()
    return HTMLResponse(
        """
        <div class='alert alert-success'>
            <strong>All Data Purged:</strong> Style profiles, settings, and imported messages have been permanently deleted from this machine.
        </div>
        """
    )


@app.get("/api/profile/export")
async def export_profile():
    """Download profile as JSON file."""
    profile = storage.get_profile()
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found to export.")

    filename = f"whatsapp_style_profile_{profile.sender_name or 'me'}.json"
    content = profile.model_dump_json(indent=2)
    return JSONResponse(
        content=profile.model_dump(),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/profile/import", response_class=HTMLResponse)
async def import_profile(
    request: Request,
    file: UploadFile = File(...),
):
    """Import a previously exported JSON profile."""
    try:
        content = await file.read()
        profile_data = StyleProfile.model_validate_json(content)
        storage.save_profile(profile_data)
        return templates.TemplateResponse(
            request=request,
            name="partials/profile_view.html",
            context={
                "profile": profile_data,
                "success_message": "Profile imported successfully!",
            },
        )
    except Exception as e:
        return HTMLResponse(
            f"<div class='alert alert-error'>Failed to import profile: {str(e)}</div>",
            status_code=400,
        )
