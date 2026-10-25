import os
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
from app.prompt import build_system_prompt, build_user_prompt, parse_draft_options
from app.ollama import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from app.storage import storage

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


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders the main dashboard page."""
    profile = storage.get_profile()
    ollama_cfg = storage.get_ollama_config()
    priv_cfg = storage.get_privacy_config()
    client = OllamaClient(ollama_cfg)
    ollama_status = await client.check_health()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "profile": profile,
            "ollama_cfg": ollama_cfg,
            "priv_cfg": priv_cfg,
            "ollama_status": ollama_status,
        },
    )


@app.get("/api/ollama/status", response_class=HTMLResponse)
async def check_ollama_status(request: Request):
    """Pings Ollama and returns an updated status badge / model dropdown."""
    client = get_ollama_client()
    ollama_status = await client.check_health()
    ollama_cfg = storage.get_ollama_config()

    return templates.TemplateResponse(
        request=request,
        name="partials/ollama_status.html",
        context={
            "ollama_status": ollama_status,
            "ollama_cfg": ollama_cfg,
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
):
    """
    Drafts replies using locally running Ollama model and style profile.
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

    draft_req = DraftRequest(
        context=context,
        intent=intent,
        length_preference=length_preference if length_preference in ("short", "medium", "long") else "medium",
        style_intensity=style_intensity if style_intensity in ("subtle", "moderate", "strong") else "moderate",
        num_drafts=max(1, min(5, num_drafts)),
        model_name=model_name.strip() if model_name and model_name.strip() else None,
    )

    profile = storage.get_profile()
    if not profile:
        # Fallback to an empty/generic profile if user hasn't trained one yet
        profile = StyleProfile(
            profile_name="Generic Friendly Style",
            sender_name="Me",
            custom_notes="Casual, friendly WhatsApp reply.",
        )

    # Construct prompts
    system_prompt = build_system_prompt(profile, draft_req.style_intensity)
    user_prompt = build_user_prompt(draft_req)

    client = get_ollama_client()
    target_model = draft_req.model_name or client.config.model

    try:
        raw_output = await client.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=target_model,
        )

        drafts = parse_draft_options(raw_output, draft_req.num_drafts)

        response_data = DraftResponse(
            drafts=drafts,
            model_used=target_model,
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
        return HTMLResponse(
            f"""
            <div class='alert alert-error'>
                <strong>Ollama Connection Error:</strong><br>
                {e}<br><br>
                <em>Check that Ollama is installed and running locally with <code>ollama serve</code>.</em>
            </div>
            """,
            status_code=503,
        )
    except OllamaModelNotFoundError as e:
        return HTMLResponse(
            f"""
            <div class='alert alert-warning'>
                <strong>Model Missing:</strong><br>
                {e}<br><br>
                <em>Pull the model in your terminal using <code>ollama pull {target_model}</code> or choose another model in Settings.</em>
            </div>
            """,
            status_code=404,
        )
    except OllamaTimeoutError as e:
        return HTMLResponse(
            f"""
            <div class='alert alert-error'>
                <strong>Request Timeout:</strong><br>
                {e}<br>
                <em>You can increase the timeout in the Settings tab.</em>
            </div>
            """,
            status_code=504,
        )
    except Exception as e:
        return HTMLResponse(
            f"""
            <div class='alert alert-error'>
                <strong>Draft Generation Failed:</strong><br>
                {str(e)}
            </div>
            """,
            status_code=500,
        )


@app.post("/api/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    ollama_url: str = Form(...),
    ollama_model: str = Form(...),
    ollama_timeout: float = Form(60.0),
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

    # Recheck health
    client = OllamaClient(cfg)
    status = await client.check_health()

    return templates.TemplateResponse(
        request=request,
        name="partials/ollama_status.html",
        context={
            "ollama_status": status,
            "ollama_cfg": cfg,
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
