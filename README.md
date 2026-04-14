# Whesper

Whesper is a Python-first CLI companion focused on natural conversation, configurable personas, memory, and multi-model routing.

The project is local-first: you can run it with Ollama on your machine, with Kimi or SiliconFlow, or with a local chat model plus a remote fallback.

## What It Can Do

- Interactive CLI chat with streaming responses
- Session persistence, session switching, rename, retry, and transcript history
- Per-session model pinning with `/model` and route-mode override with `/mode`
- Local memory capture and recall with `/memory`, `/remember`, and `/forget`
- Search / live-data assisted turns for things like weather, news, docs, and URLs
- Config-driven providers through `TOML`
- Local Ollama support and optional Moonshot/Kimi or SiliconFlow support

## Quick Start

### Recommended: one-command bootstrap

Whesper ships with a bootstrap script that creates a virtual environment, installs the package, and writes a usable `whesper.toml`.

```bash
./scripts/bootstrap.sh
```

The script will choose a runnable setup automatically:

- If you already have a local Ollama model, it writes a local-first config
- If no local model is found but `WHESPER_KIMI_API_KEY` is set, it writes a Kimi-only config
- If no local model is found but `WHESPER_SILICONFLOW_API_KEY` is set, it writes a SiliconFlow-only config
- If neither is available, it stops with clear next steps instead of generating a broken config

Useful environment variables:

```bash
WHESPER_LOCAL_MODEL="qwen3.5:14b" ./scripts/bootstrap.sh
WHESPER_KIMI_API_KEY="your-api-key" ./scripts/bootstrap.sh
WHESPER_SILICONFLOW_API_KEY="your-api-key" ./scripts/bootstrap.sh
```

After bootstrap:

```bash
.venv/bin/python -m whesper --config whesper.toml chat
```

### Manual setup

If you prefer to configure things yourself:

1. Create a virtual environment and install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

2. Create your local config:

```bash
cp whesper.example.toml whesper.toml
```

3. Update `whesper.toml`:

- Set `[models.local_chat].model` to a real Ollama model name if you want local chat
- Export `WHESPER_KIMI_API_KEY` or `WHESPER_SILICONFLOW_API_KEY` if you want a remote model
- If you are running fully local, set `search_model = "local_chat"` under `[scheduler]`

4. Start the CLI:

```bash
.venv/bin/python -m whesper --config whesper.toml chat
```

## Common Commands

- `/models`
- `/model local_chat`
- `/model kimi-k2.5`
- `/model siliconflow-qwen3-8b`
- `/model auto`
- `/mode auto`
- `/mode reasoning`
- `/search latest AI news`
- `/trace`
- `/status`
- `/info`
- `/retry`
- `/copy-last`
- `/rename test-session`
- `/delete-session old-session`
- `/history`
- `/memory`
- `/remember I like jasmine tea`
- `/forget profile-memory-id`
- `/new test-session`
- `/clear`
- `/exit`

## Configuration

Tracked example config:

- [whesper.example.toml](whesper.example.toml)

Real local config:

- `whesper.toml`

`whesper.toml` is ignored by git so local secrets and internal endpoints stay out of version control.

Default model aliases in the example config:

- `local_chat`: local Ollama chat model
- `kimi-k2.5`: Moonshot/Kimi via `WHESPER_KIMI_API_KEY`
- `siliconflow-qwen3-8b`: SiliconFlow via `WHESPER_SILICONFLOW_API_KEY`

Optional live-data endpoint sections are also supported in config:

- `live_context.status_api.*`
- `live_context.custom_api.*`
- `live_context.search_api`

`live_context.search_api` defaults to Brave web search with no API key required. `serpapi` is still supported if you want it.

Leave them empty if you want a narrow default setup.

## Runtime Notes

- Python `3.12+` is required
- Ollama is optional, but recommended for local-first usage
- Kimi and SiliconFlow are optional remote fallbacks
- SiliconFlow uses the OpenAI-compatible API; Whesper maps `think` to SiliconFlow's `enable_thinking` field automatically
- Session and memory data are stored under `.whesper/`
- `whesper.toml` is local-only and should not be committed

## Project Status

Whesper is still in an MVP / prototype phase. The current focus is:

- making the conversation loop and tool path solid
- improving CLI usability and deployment ergonomics
- validating local-first and hybrid model routing before expanding beyond CLI

## Development

Local development notes can live in an ignored file such as:

- `DEVELOPMENT_NOTES.local.md`

The current CLI task list lives in:

- [CLI_TODO.md](CLI_TODO.md)
