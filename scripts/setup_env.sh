#!/bin/bash
# One-shot environment setup for macOS Apple Silicon.
# Run from repo root: bash scripts/setup_env.sh

set -e

echo "==> Checking prerequisites..."
command -v python3.11 >/dev/null 2>&1 || { echo "Install Python 3.11: brew install python@3.11"; exit 1; }
command -v java >/dev/null 2>&1 || { echo "Install Java 17: brew install openjdk@17"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Install Docker Desktop from docker.com"; exit 1; }

echo "==> Creating Python venv..."
if [ ! -d .venv ]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate

echo "==> Installing Python deps..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Setting JAVA_HOME for current shell..."
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
echo "JAVA_HOME=$JAVA_HOME"

echo ""
echo "Setup complete."
echo "Activate the venv anytime with: source .venv/bin/activate"
echo "Then start Kafka with:           cd docker && docker compose up -d"