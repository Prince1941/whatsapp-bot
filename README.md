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
│   ├── config.py         # App configuration, limits, and paths
│   ├── models.py         # Pydantic v2 schemas for chat data, profiles, and requests
│   ├── ollama.py         # Local Ollama HTTP API client with robust error handling
│   ├── parser.py         # WhatsApp export parser (Android, iOS bracketed, multiline, media)
│   ├── privacy.py        # PII redaction (phones, emails, URLs, credit cards, banned keywords)
│   ├── prompt.py         # System prompt builder with strict anti-hallucination constraints
│   ├── storage.py        # Local SQLite repository for style profiles and settings
│   └── web/              # FastAPI application + Jinja2 + HTMX frontend
│       ├── main.py
│       ├── templates/
│       └── static/
├── tests/                # Comprehensive test suite (35+ unit and integration tests)
├── run.py                # Server runner script
└── requirements.txt      # Project dependencies
```

---

## Key Features

### 1. Chat Import & Preview
- **Broad WhatsApp Format Support**: Parses Android standard 12h/24h timestamps (`15/01/24, 10:14 am - Name: ...`) and iOS bracketed format (`[15/01/2024, 10:14:22] Name: ...`).
- **Multiline & Media Handling**: Accurately aggregates multiline messages and recognizes attachment notices (`<Media omitted>`, `image omitted`, `video omitted`).
- **Sender Identification**: Detects all conversation participants and lets you select your identity.
- **Participant Isolation**: Other participants' messages are excluded by default.
- **Privacy Filtering**: Redacts or drops phone numbers, emails, financial records, web URLs, and custom user-provided terms.
- **Zero Raw Storage by Default**: Raw chat files are parsed in-memory and discarded after style extraction unless you explicitly check the opt-in persistence box.

### 2. Inspectable Style Profile
- **Brevity & Length**: Tracks average words and characters per message, alongside short/medium/long message ratios.
- **Capitalization**: Distinguishes between lowercase typing (`"sounds good bro"`), sentence case, or mixed casing.
- **Punctuation Trends**: Measures period-ending habits (casual messaging often omits trailing periods), exclamation rates, question frequency, and ellipses.
- **Emoji Habits**: Extracts emoji frequency and identifies your most frequently used emojis.
- **Slang & Distinctive Vocabulary**: Discovers conversational slang, contractions, common greetings, and sign-offs.
- **Editable & Inspectable**: Review, tweak parameters, edit sample snippets, or export/import profile backups as JSON.

### 3. Reply Drafting with Anti-Hallucination Controls
- **Human Intent First**: You provide what you want to say (e.g. *"agree to dinner, but after 7:30pm"*).
- **Anti-Hallucination Constraints**: The system prompt strictly prohibits inventing commitments, promises, appointments, or personal memories not present in the prompt.
- **No Verbatim Copying**: Adopts rhythm, brevity, and tone without copying previous training messages word-for-word.
- **Controls**: Adjust desired length (short, medium, long), style intensity (subtle, moderate, strong), and number of options.
- **In-Place Editing**: Edit generated drafts directly in the browser and copy with one click.
- **Prominent Review Badge**: Displays `⚠️ AI-Generated Draft — Review Before Use`.

### 4. Ollama Local Integration
- **Health & Model Detection**: Auto-detects local Ollama instance and displays installed models.
- **Customizable Host & Model**: Configure Ollama URL and preferred model in the Settings tab.
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

## Ollama Setup

1. **Install Ollama**:
   - macOS: `brew install ollama` or download from [ollama.com](https://ollama.com).
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Windows: Download installer from [ollama.com](https://ollama.com).

2. **Start the local Ollama server**:
   ```bash
   ollama serve
   ```

3. **Pull a model of your choice**:
   Fast, lightweight models ideal for chat replies:
   ```bash
   ollama pull llama3.2
   # Or alternatives:
   ollama pull mistral
   ollama pull qwen2.5:3b
   ```

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

## Running the Tests

The test suite thoroughly mocks Ollama, so tests run quickly without needing Ollama active:
```bash
PYTHONPATH=. ./.venv/bin/pytest -v tests
```

---

## Privacy, Trust & Safety Guidelines

1. **Export Authorization**:
   Only import chats where all participants have given consent or where you are legally authorized. WhatsApp exports contain unencrypted personal conversations.
2. **Local-Only Guarantee**:
   All processing occurs on `localhost` (`127.0.0.1`). No cloud APIs or third-party analytical trackers are used.
3. **Data Deletion & Purging**:
   - Delete individual style profiles directly from the **Style Profile** tab.
   - To completely remove all data (profiles, settings, and database), navigate to **Settings & Privacy** and click **Purge All Data From This Machine**.
   - Alternatively, delete the `data/` folder manually:
     ```bash
     rm -rf data/
     ```
4. **No Automated Sending**:
   This tool intentionally does not include any mechanism to send messages, connect to WhatsApp Web, or automate chat clients.

---

## Known Limitations

- **Ollama Performance**: Generation speed depends on your local hardware (GPU/CPU/RAM) and the selected model size.
- **Export Formats**: Standard iOS bracketed and Android comma-separated formats are supported; unusual third-party modifications or third-party WhatsApp clones may require custom delimiters.
- **Language Nuance**: Style extraction works across languages with Unicode emoji support, but slang candidate dictionaries are currently optimized for English and common conversational slang.
