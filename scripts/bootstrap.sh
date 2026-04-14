#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CONFIG_EXAMPLE="${ROOT_DIR}/whesper.example.toml"
CONFIG_FILE="${ROOT_DIR}/whesper.toml"

LOCAL_MODEL="${WHESPER_LOCAL_MODEL:-}"
KIMI_API_KEY="${WHESPER_KIMI_API_KEY:-}"
SILICONFLOW_API_KEY="${WHESPER_SILICONFLOW_API_KEY:-}"
SKIP_INSTALL="${WHESPER_SKIP_INSTALL:-0}"
BOOTSTRAP_PROFILE=""
REMOTE_PROFILE=""
REMOTE_MODEL_ALIAS="local_chat"
PYTHON_BIN=""

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
    log "Using local model from WHESPER_LOCAL_MODEL: ${LOCAL_MODEL}"
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

remote_runtime_label() {
  case "$1" in
    kimi)
      printf 'Kimi'
      ;;
    siliconflow)
      printf 'SiliconFlow'
      ;;
    *)
      printf 'remote'
      ;;
  esac
}

select_bootstrap_profile() {
  detect_local_model

  if [ -n "$LOCAL_MODEL" ]; then
    BOOTSTRAP_PROFILE="local"
    if [ -n "$KIMI_API_KEY" ]; then
      REMOTE_PROFILE="kimi"
      REMOTE_MODEL_ALIAS="kimi-k2.5"
    elif [ -n "$SILICONFLOW_API_KEY" ]; then
      REMOTE_PROFILE="siliconflow"
      REMOTE_MODEL_ALIAS="siliconflow-qwen3-8b"
    fi
    return 0
  fi

  if [ -n "$KIMI_API_KEY" ]; then
    BOOTSTRAP_PROFILE="kimi"
    REMOTE_PROFILE="kimi"
    REMOTE_MODEL_ALIAS="kimi-k2.5"
    log "No local Ollama model detected; configuring Kimi as the default runtime"
    return 0
  fi

  if [ -n "$SILICONFLOW_API_KEY" ]; then
    BOOTSTRAP_PROFILE="siliconflow"
    REMOTE_PROFILE="siliconflow"
    REMOTE_MODEL_ALIAS="siliconflow-qwen3-8b"
    log "No local Ollama model detected; configuring SiliconFlow as the default runtime"
    return 0
  fi

  fail \
"Could not detect a local Ollama model and neither WHESPER_KIMI_API_KEY nor WHESPER_SILICONFLOW_API_KEY is set.
To complete one-click deployment, do one of the following before rerunning bootstrap:
  1. Install/start Ollama and pull a model, then rerun ./scripts/bootstrap.sh
  2. Export WHESPER_LOCAL_MODEL=\"your-local-model\"
  3. Export WHESPER_KIMI_API_KEY=\"your-api-key\"
  4. Export WHESPER_SILICONFLOW_API_KEY=\"your-api-key\""
}

write_config_from_example() {
  cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"

  WHESPER_BOOTSTRAP_PROFILE="$BOOTSTRAP_PROFILE" \
  WHESPER_BOOTSTRAP_LOCAL_MODEL="$LOCAL_MODEL" \
  WHESPER_BOOTSTRAP_REMOTE_MODEL="$REMOTE_MODEL_ALIAS" \
  "$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
from pathlib import Path
import os
import sys

config_path = Path(sys.argv[1])
text = config_path.read_text(encoding="utf-8")
profile = os.environ["WHESPER_BOOTSTRAP_PROFILE"]
local_model = os.environ.get("WHESPER_BOOTSTRAP_LOCAL_MODEL", "").strip()
remote_model = os.environ.get("WHESPER_BOOTSTRAP_REMOTE_MODEL", "local_chat").strip() or "local_chat"

if profile == "local":
    if not local_model:
        raise SystemExit("missing local model for local bootstrap profile")
    text = text.replace('model = "your-model"', f'model = "{local_model}"', 1)
    text = text.replace('search_model = "kimi-k2.5"', f'search_model = "{remote_model}"', 1)
elif profile in {"kimi", "siliconflow"}:
    text = text.replace('chat_model = "local_chat"', f'chat_model = "{remote_model}"', 1)
    text = text.replace('reasoning_model = "local_chat"', f'reasoning_model = "{remote_model}"', 1)
    text = text.replace('search_model = "kimi-k2.5"', f'search_model = "{remote_model}"', 1)
    text = text.replace('model = "your-model"', 'model = "local-model-not-configured"', 1)
else:
    raise SystemExit(f"unknown bootstrap profile: {profile}")

config_path.write_text(text, encoding="utf-8")
PY

  case "$BOOTSTRAP_PROFILE" in
    local)
      if [ -n "$REMOTE_PROFILE" ]; then
        log "Created whesper.toml for local chat with $(remote_runtime_label "$REMOTE_PROFILE") search fallback"
      else
        log "Created whesper.toml for fully local use"
      fi
      ;;
    kimi)
      log "Created whesper.toml for Kimi-only runtime"
      ;;
    siliconflow)
      log "Created whesper.toml for SiliconFlow-only runtime"
      ;;
  esac
}

validate_generated_config() {
  PYTHONPATH="${ROOT_DIR}/src" "$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import sys

from whesper.config import load_config

load_config(sys.argv[1])
PY
}

install_package() {
  if [ "$SKIP_INSTALL" = "1" ]; then
    log "Skipping package installation because WHESPER_SKIP_INSTALL=1"
    return 0
  fi

  log "Installing package into virtual environment"
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "${VENV_DIR}/bin/python" -m pip install --no-build-isolation -e "$ROOT_DIR"
}

show_next_steps() {
  cat <<EOF

Whesper is ready to start.

Project root:
  ${ROOT_DIR}

Run the CLI:
  .venv/bin/python -m whesper --config whesper.toml chat

Helpful environment variables:
  WHESPER_OLLAMA_BASE_URL  Optional, override the Ollama base URL (default: http://localhost:11434)
  WHESPER_LOCAL_MODEL   Force the local Ollama model name bootstrap should write
  WHESPER_KIMI_API_KEY  Optional, enables or becomes the default Kimi runtime
  WHESPER_SILICONFLOW_API_KEY  Optional, enables or becomes the default SiliconFlow runtime
  WHESPER_SKIP_INSTALL  Optional, set to 1 to skip pip install during debugging

EOF

  case "$BOOTSTRAP_PROFILE" in
    local)
      cat <<EOF
Configured runtime:
  chat_model      = local_chat
  reasoning_model = local_chat
  search_model    = ${REMOTE_MODEL_ALIAS}

Selected local model:
  ${LOCAL_MODEL}

EOF
      ;;
    kimi)
      cat <<EOF
Configured runtime:
  chat_model      = ${REMOTE_MODEL_ALIAS}
  reasoning_model = ${REMOTE_MODEL_ALIAS}
  search_model    = ${REMOTE_MODEL_ALIAS}

Kimi-only bootstrap was selected because no local Ollama model was detected.

EOF
      ;;
    siliconflow)
      cat <<EOF
Configured runtime:
  chat_model      = ${REMOTE_MODEL_ALIAS}
  reasoning_model = ${REMOTE_MODEL_ALIAS}
  search_model    = ${REMOTE_MODEL_ALIAS}

SiliconFlow-only bootstrap was selected because no local Ollama model was detected.

EOF
      ;;
  esac

  if [ "$BOOTSTRAP_PROFILE" = "kimi" ] || [ "$BOOTSTRAP_PROFILE" = "siliconflow" ]; then
    cat <<EOF
If you later add a local Ollama model, rerun bootstrap like this:
  export WHESPER_OLLAMA_BASE_URL="http://localhost:11434"
  WHESPER_LOCAL_MODEL="your-local-model" ./scripts/bootstrap.sh

EOF
  fi

  if [ -z "$KIMI_API_KEY" ]; then
    cat <<EOF
Kimi is not configured yet.
If you want to use the remote Kimi model, export:
  export WHESPER_KIMI_API_KEY="your-api-key"

EOF
  fi

  if [ -z "$SILICONFLOW_API_KEY" ]; then
    cat <<EOF
SiliconFlow is not configured yet.
If you want to use the remote SiliconFlow model, export:
  export WHESPER_SILICONFLOW_API_KEY="your-api-key"

EOF
  fi
}

main() {
  require_command cp

  PYTHON_BIN="$(select_python)" || fail "Python 3.12+ is required"
  log "Using Python: ${PYTHON_BIN} ($(python_major_minor "$PYTHON_BIN"))"

  select_bootstrap_profile

  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    log "Virtual environment already exists"
  fi

  install_package

  if [ ! -f "$CONFIG_FILE" ]; then
    write_config_from_example
    validate_generated_config
  else
    log "Keeping existing whesper.toml"
  fi

  show_next_steps
}

main "$@"
