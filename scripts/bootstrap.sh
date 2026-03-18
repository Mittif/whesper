#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CONFIG_EXAMPLE="${ROOT_DIR}/whesper.example.toml"
CONFIG_FILE="${ROOT_DIR}/whesper.toml"

LOCAL_MODEL="${WHESPER_LOCAL_MODEL:-}"
KIMI_API_KEY="${WHESPER_KIMI_API_KEY:-}"

log() {
  printf '[whesper-bootstrap] %s\n' "$1"
}

fail() {
  printf '[whesper-bootstrap] ERROR: %s\n' "$1" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

python_major_minor() {
  "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

version_ge() {
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

select_python() {
  local candidates=("python3.14" "python3.13" "python3.12" "python3")
  local candidate
  local version

  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    version="$(python_major_minor "$candidate")"
    if version_ge "$version" "3.12"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

detect_local_model() {
  local detected_model

  if [ -n "$LOCAL_MODEL" ]; then
    return 0
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    return 0
  fi

  detected_model="$(ollama list 2>/dev/null | awk 'NR > 1 && $1 != "" { print $1; exit }')"
  if [ -n "$detected_model" ]; then
    LOCAL_MODEL="$detected_model"
    log "Detected local Ollama model: ${LOCAL_MODEL}"
  fi
}

write_config_from_example() {
  cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"

  if [ -n "$LOCAL_MODEL" ]; then
    perl -0pi -e 's/model = "your-model"/model = $ENV{WHESPER_LOCAL_MODEL_REPLACE}/' \
      "$CONFIG_FILE"
    log "Created whesper.toml and set local model to: ${LOCAL_MODEL}"
  else
    log "Created whesper.toml from template"
  fi
}

show_next_steps() {
  cat <<EOF

Whesper is ready to start.

Project root:
  ${ROOT_DIR}

Run the CLI:
  .venv/bin/python -m whesper --config whesper.toml chat

Helpful environment variables:
  WHESPER_LOCAL_MODEL   Set this before running bootstrap to replace the example model
  WHESPER_KIMI_API_KEY  Optional, enables the Kimi provider

EOF

  if [ -z "$LOCAL_MODEL" ] && grep -q 'model = "your-model"' "$CONFIG_FILE"; then
    cat <<EOF
Before first chat, edit whesper.toml and replace:
  model = "your-model"
with your local Ollama model name, for example:
  model = "qwen3.5:14b"

Or rerun bootstrap like this:
  WHESPER_LOCAL_MODEL="qwen3.5:14b" ./scripts/bootstrap.sh

EOF
  fi

  if [ -z "$KIMI_API_KEY" ]; then
    cat <<EOF
Kimi is not configured yet.
If you want to use the remote Kimi model, export:
  export WHESPER_KIMI_API_KEY="your-api-key"

EOF
  fi
}

main() {
  require_command cp
  require_command grep
  require_command perl

  local python_bin
  python_bin="$(select_python)" || fail "Python 3.12+ is required"
  log "Using Python: ${python_bin} ($(python_major_minor "$python_bin"))"

  detect_local_model

  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment"
    "$python_bin" -m venv "$VENV_DIR"
  else
    log "Virtual environment already exists"
  fi

  log "Installing package into virtual environment"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -e "$ROOT_DIR"

  if [ ! -f "$CONFIG_FILE" ]; then
    export WHESPER_LOCAL_MODEL_REPLACE="\"${LOCAL_MODEL}\""
    write_config_from_example
  else
    log "Keeping existing whesper.toml"
  fi

  show_next_steps
}

main "$@"
