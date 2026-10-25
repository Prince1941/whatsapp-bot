# WhatsApp Style AI Assistant (Local-First)

A local-first, privacy-preserving writing copilot that analyzes your authorized WhatsApp chat exports (`.txt`), learns your personal writing cadence (brevity, lowercase habits, punctuation, top emojis, and colloquial slang), and drafts reply options using a local **Ollama** model.

> **Privacy & Human Review Mandate:**
> - **100% Local Processing:** Runs on your machine with Ollama. Zero external API calls, zero telemetry, zero cloud data transfer.
> - **Strict Human-in-the-Loop:** Does NOT send messages automatically. There is no WhatsApp Web automation, session scraping, or background messaging. Every draft must be manually reviewed, edited, and approved by you.
> - **Authorization Required:** Only import chat exports you have explicit permission to use. All messages from other conversation participants are immediately and permanently discarded.

---

## Architecture & Modular Design

```
my-bot/
├── app/
│   ├── analyzer.py       # Stylistic metric extraction (length, punctuation, capitalization, slang, emojis)
│   ├── classifier.py     # Incoming communicative intent classification (status, activity, availability)
│   ├── config.py         # App configuration, limits, and paths
│   ├── language.py       # Script detection, resolution (Intent > Message), and script validation
│   ├── models.py         # Pydantic v2 schemas for chat data, profiles, and requests
│   ├── ollama.py         # Local Ollama HTTP API client with dynamic model detection & robust retries
│   ├── parser.py         # WhatsApp export parser (Android, iOS bracketed, multiline, media)
│   ├── privacy.py        # PII redaction (phones, emails, URLs, credit cards, banned keywords)
│   ├── prompt.py         # Script-aware system & user prompt builder with strict anti-hallucination rules
│   ├── providers.py      # LLM provider abstractions (Local Ollama, Gemini, Groq, OpenAI)
│   ├── retrieval.py      # Semantic vector search for authentic conversational few-shot pairs
│   ├── storage.py        # Local SQLite repository for style profiles, settings, and chat pairs
│   ├── validator.py      # Script conformity validation, semantic intent verification & deduping
│   └── web/              # FastAPI application + Jinja2 + HTMX frontend
│       ├── main.py
│       ├── templates/
│       └── static/
├── scripts/
│   └── eval_reply_system.py # Comprehensive evaluation harness for intent matching & vocabulary fidelity
├── tests/                # Test suite (unit, integration, and language/script control tests)
├── run.py                # Server runner script
├── run.sh                # Shell launcher script
└── requirements.txt      # Project dependencies
```

---

## Key Features

### 1. Language & Script Control (Strict Fidelity)
- **Script Prioritization**: The reply's language and script strictly follows your explicit **Intent** first. When Intent is blank, it automatically follows the latest incoming message's language and script.
- **Explicit Selector**: Choose between `Auto (follow my intent/message)`, `Romanized Hinglish`, `Devanagari Hindi (हिन्दी)`, and `English`. Default is `Auto`.
- **Exact Spelling Preservation**: Never transliterates, translates, or "corrects" user-provided spellings. If you write `"thik hu bhai"`, the assistant will not alter it to `"theek hoon bhai"` or `"ठीक हूँ भाई"`.
- **Output Script Validation & Corrective Retry**: Every generated draft is validated against the selected script. If the model produces the wrong script, it retries once with a corrective prompt. If it still fails, a clear warning allows regeneration or editing.
- **Privacy-Preserving Diagnostics**: Logs only non-sensitive language/script mode changes; never logs private chat text.

### 2. Chat Import & Preview
- **Broad WhatsApp Format Support**: Parses Android standard 12h/24h timestamps (`15/01/24, 10:14 am - Name: ...`) and iOS bracketed format (`[15/01/2024, 10:14:22] Name: ...`).
- **Multiline & Media Handling**: Accurately aggregates multiline messages and recognizes attachment notices (`<Media omitted>`, `image omitted`, `video omitted`).
- **Sender Identification**: Detects all conversation participants and lets you select your identity.
- **Participant Isolation**: Other participants' messages are excluded by default.
- **Privacy Filtering**: Redacts or drops phone numbers, emails, financial records, web URLs, and custom user-provided terms.
- **Zero Raw Storage by Default**: Raw chat files are parsed in-memory and discarded after style extraction unless you explicitly check the opt-in persistence box.

### 3. Inspectable Style Profile
- **Brevity & Length**: Tracks average words and characters per message, alongside short/medium/long message ratios.
- **Capitalization**: Distinguishes between lowercase typing (`"sounds good bro"`), sentence case, or mixed casing.
- **Punctuation Trends**: Measures period-ending habits (casual messaging often omits trailing periods), exclamation rates, question frequency, and ellipses.
- **Emoji Habits**: Extracts emoji frequency and identifies your most frequently used emojis.
- **Slang & Distinctive Vocabulary**: Discovers conversational slang, contractions, common greetings, and sign-offs.
- **Editable & Inspectable**: Review, tweak parameters, edit sample snippets, or export/import profile backups as JSON.

### 4. Reply Drafting with Anti-Hallucination Controls
- **Human Intent First**: You provide what you want to say (e.g. *"agree to dinner, but after 7:30pm"*).
- **Anti-Hallucination Constraints**: The system prompt strictly prohibits inventing commitments, promises, appointments, or personal memories not present in the prompt.
- **No Verbatim Copying**: Adopts rhythm, brevity, and tone without copying previous training messages word-for-word.
- **Controls**: Adjust desired length (short, medium, long), style intensity (subtle, moderate, strong), and number of options.
- **In-Place Editing**: Edit generated drafts directly in the browser and copy with one click.
- **Prominent Review Badge**: Displays `⚠️ AI-Generated Draft — Review Before Use` alongside the active script badge.

### 5. Ollama Local Integration
- **Health & Dynamic Model Detection**: Auto-detects local Ollama instance, checks installed models via `/api/tags`, and populates model dropdowns dynamically.
- **Single Source of Truth**: Configurable model default (`qwen2.5:7b`), with graceful fallback and warnings if saved model is uninstalled.
- **Helpful Failure Modes**: Clear, actionable error messages for Ollama connection refusal, missing models, and request timeouts.

---

## Prerequisites & Installation

### 1. Environment
Python 3.11+ is required.

Clone and enter the directory:
```bash
cd "/Users/princekumar/Desktop/w bot/my-bot"
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Ollama & Model Setup

1. **Install Ollama**:
   - macOS: `brew install ollama` or download from [ollama.com](https://ollama.com).
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Windows: Download installer from [ollama.com](https://ollama.com).

2. **Start the local Ollama server**:
   ```bash
   ollama serve
   ```

3. **Pull Recommended Models**:
   For accurate Hinglish semantic understanding and vector retrieval, pull `qwen2.5:7b` (or `llama3.1:8b`) and `nomic-embed-text`:
   ```bash
   # Recommended generation model for Hinglish conversational reasoning:
   ollama pull qwen2.5:7b

   # Embedding model for vector semantic search across chat pairs:
   ollama pull nomic-embed-text

   # Fast lightweight alternatives:
   ollama pull qwen2.5:3b
   ollama pull llama3.1:8b
   ```

---

## Adding WhatsApp Exports & Indexing Pairs

1. **Export WhatsApp Chat**:
   - In WhatsApp, open any chat with a contact you talk to frequently.
   - Tap Chat Name -> **Export Chat** -> Choose **Without Media**.
   - Save the `.txt` file.

2. **Import & Learn**:
   - In the web app, navigate to the **Import Chats** tab.
   - Upload the `.txt` export file and enter your exact sender name (e.g., `Prince Kr`).
   - Click **Save & Learn Profile**.
   - The app automatically:
     - Extracts your individual message style patterns (slang, emojis, punctuation).
     - Extracts incoming -> outgoing conversational exchange pairs.
     - Automatically generates vector embeddings via `nomic-embed-text` and indexes them in SQLite for retrieval.

3. **Batch Indexing via Script (Optional)**:
   You can also index chat pairs directly from the command line:
   ```bash
   PYTHONPATH=. ./.venv/bin/python -c '
   from app.parser import parse_whatsapp_export
   from app.retrieval import extract_chat_pairs, embed_texts
   from app.storage import save_chat_pairs

   with open("/path/to/_chat.txt", "r", encoding="utf-8") as f:
       msgs = parse_whatsapp_export(f.read(), "My Profile")
   pairs = extract_chat_pairs(msgs, "Prince Kr", "My Profile")
   embeds = embed_texts([p.incoming_text for p in pairs])
   for p, e in zip(pairs, embeds):
       p.embedding = e
   save_chat_pairs(pairs)
   print(f"Indexed {len(pairs)} pairs!")
   '
   ```

---

## Switching Models & Providers

The system supports **Local Ollama** (default, 100% private) and **Optional Cloud Providers** (Gemini, Groq, OpenAI):

### 1. Local Ollama (Default)
In `app/config.py` or via the web UI Settings:
```python
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"      # Best for Hinglish semantic grounding
EMBEDDING_MODEL = "nomic-embed-text"    # Fast 768-dim embeddings
```

### 2. Cloud Providers (Optional)
If local hardware cannot run 7B+ models smoothly, configure optional cloud API keys in your environment:
```bash
# Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# Groq (Ultra-fast Llama-3.3-70b)
export GROQ_API_KEY="your-groq-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```
> [!NOTE]
> When using any cloud provider, **PII Masking** is automatically enforced (`app/providers.py:mask_sensitive_pii`). Phone numbers, email addresses, and Indian UPI IDs are redacted before prompts leave your machine. A warning badge is also displayed in the web UI.

---

## Running the Application

Launch the local web server:
```bash
./.venv/bin/python run.py --port 8000
```
Or with auto-reload for development:
```bash
./.venv/bin/python run.py --port 8000 --reload
```

Then open your browser at:
```
http://127.0.0.1:8000
```

---

## Running Evaluation Benchmarks

We provide an automated evaluation harness that tests semantic intent matching, Hinglish vocabulary fidelity, and length control against real holdout pairs and hand tests:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/eval_reply_system.py
```

This script:
1. Holds out 100 real chat pairs not seen in few-shot retrieval.
2. Evaluates the full pipeline: intent classification -> vector retrieval -> generation -> validation.
3. Computes intent accuracy, word count comparison, slang overlap, and emoji match rate.
4. Runs 8 critical hand tests (e.g., `"kya kar rha h?"`, `"kaha h?"`, `"kal free h?"`, `"bhai call utha"`) and prints side-by-side comparisons.

---

## Running the Tests

The test suite runs with fully isolated SQLite databases and mocked LLMs:
```bash
PYTHONPATH=. ./.venv/bin/pytest -v
```

---

## Privacy, Trust & Safety Guidelines

1. **Export Authorization**:
   Only import chats where all participants have given consent or where you are legally authorized. WhatsApp exports contain unencrypted personal conversations.
2. **Local-Only Guarantee**:
   All processing occurs on `localhost` (`127.0.0.1`) by default. No data is sent to external services unless you explicitly configure an optional cloud provider.
3. **Data Deletion & Purging**:
   - Delete individual style profiles directly from the **Style Profile** tab.
   - Clear indexed chat pairs or purge everything from **Settings & Privacy**.
   - Alternatively, delete the `data/` folder manually:
     ```bash
     rm -rf data/
     ```
4. **No Automated Sending**:
   This tool intentionally does not include any mechanism to send messages, connect to WhatsApp Web, or automate chat clients. All drafts require human review and approval.

---

## Known Limitations

- **Ollama Performance**: Generation speed depends on your local hardware (GPU/CPU/RAM) and the selected model size. On Apple Silicon (M1/M2/M3/M4), `qwen2.5:7b` runs in ~1.5 seconds.
- **Export Formats**: Standard iOS bracketed (`[DD/MM/YY, HH:MM:SS]`) and Android comma-separated (`DD/MM/YY, HH:MM -`) formats are supported.
- **Context Depth**: The system extracts 1-to-1 immediate reply exchanges. Multi-message bursts within 3 minutes are concatenated into single turns.
