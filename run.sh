#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
fi

export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
echo "Starting WhatsApp Style AI Assistant on http://127.0.0.1:8000 ..."
PYTHONPATH=. ./.venv/bin/python run.py "$@"
