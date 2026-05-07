#!/usr/bin/env bash
set -euo pipefail

# Unified installer for the Private Agent Assistant repo.
# Installs Python dependencies for:
# - main agent backend
# - desktop GUI
# - optional seeding utilities
# - vendored external qwen3-tts-api package deps

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_SEEDING_DEPS="${INSTALL_SEEDING_DEPS:-1}"
INSTALL_QWEN_TTS_PKG="${INSTALL_QWEN_TTS_PKG:-0}"

info() { echo "[install] $*"; }
warn() { echo "[install][warn] $*"; }
fail() { echo "[install][error] $*" >&2; exit 1; }

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"

if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  info "Using existing virtual environment at $VENV_DIR"
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
  # Linux/macOS/WSL
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV_DIR/Scripts/activate" ]; then
  # Git Bash on Windows
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV_DIR/Scripts/activate.bat" ]; then
  fail "Found Windows venv only (.bat). Run this script in Git Bash/WSL, or use PowerShell activation manually."
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  fail "Could not activate venv from $VENV_DIR"
fi

info "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

info "Installing main project dependencies"
python -m pip install -r requirements.txt

info "Installing GUI dependencies"
python -m pip install -r requirements-gui.txt

if [ "$INSTALL_SEEDING_DEPS" = "1" ] && [ -f "requirements-seeding.txt" ]; then
  info "Installing optional seeding dependencies"
  python -m pip install -r requirements-seeding.txt
else
  warn "Skipping seeding dependencies (INSTALL_SEEDING_DEPS=$INSTALL_SEEDING_DEPS)"
fi

if [ -f "external/qwen3-tts-api/requirements.txt" ]; then
  info "Installing vendored qwen3-tts-api dependencies"
  python -m pip install -r external/qwen3-tts-api/requirements.txt
else
  warn "external/qwen3-tts-api/requirements.txt not found; skipping"
fi

if [ "$INSTALL_QWEN_TTS_PKG" = "1" ]; then
  info "Installing optional qwen-tts package"
  python -m pip install -U qwen-tts
else
  warn "qwen-tts package not auto-installed (set INSTALL_QWEN_TTS_PKG=1 to enable)"
fi

echo
echo "Install complete."
echo
echo "Next:"
echo "1) Follow step 2, 3 in README.md to ensure correct local .env values"
echo "2) Start backend services:"
echo "   docker compose up --build"
echo "3) Verify Ollama models exist (step 5)"
echo "4) Run GUI (new shell, same venv):"
echo "   python -m src.GUI.gui_main"
echo
echo "If qwen3-tts fails with missing model/tokenizer assets, set HF_TOKEN before docker compose:"
echo "  export HF_TOKEN=\"hf_xxx\"        # bash"
echo "  # or in PowerShell:"
echo "  # \$env:HF_TOKEN=\"hf_xxx\""
