# Whesper

Whesper is a Python-first companion agent prototype focused on natural conversation, configurable personas, and multi-model routing.

The current repository ships a local CLI experience first, so the core dialogue loop can be tested quickly before moving to richer surfaces such as a web app.

## Current Scope

- Interactive CLI chat experience
- Configurable persona and system prompt
- Multi-model routing with provider abstraction
- Local Ollama support
- Remote Kimi support
- Session persistence and command-driven model switching

## Features

- Claude Code-inspired terminal UX with history, completion, and slash commands
- Streaming responses with a lightweight `thinking...` state
- Per-session model selection via `/model`
- Config-driven provider setup through `TOML`
- Python-first implementation built around a small, inspectable codebase

## Quick Start

1. Create a virtual environment and install the package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

2. Create your local config from the example:

```bash
cp whesper.example.toml whesper.toml
```

3. Export your Kimi API key if you want to use the Kimi model:

```bash
export WHESPER_KIMI_API_KEY="your-api-key"
```

4. Start the CLI:

```bash
.venv/bin/python -m whesper --config whesper.toml chat
```

## Common Commands

- `/models`
- `/model local_chat`
- `/model kimi-k2.5`
- `/model auto`
- `/mode auto`
- `/mode reasoning`
- `/status`
- `/info`
- `/retry`
- `/copy-last`
- `/rename test-session`
- `/delete-session old-session`
- `/history`
- `/new test-session`
- `/clear`
- `/exit`

## Configuration

The repository includes a tracked example config:

- [whesper.example.toml](whesper.example.toml)

Your real local config should live in:

- `whesper.toml`

`whesper.toml` is ignored by git so local secrets and internal endpoints stay out of version control.

## Providers

- `local_chat` is configured for an Ollama-compatible local or private endpoint
- `kimi-k2.5` is configured for Moonshot/Kimi via `WHESPER_KIMI_API_KEY`

## Project Status

Whesper is currently in an MVP / prototype phase. The main focus right now is:

- making the conversation loop solid
- improving CLI usability
- validating multi-model behavior before expanding into a web application

## Development

This README is intentionally written as a project-facing overview for GitHub readers.

Local development notes can live in an ignored file such as:

- `DEVELOPMENT_NOTES.local.md`

The current CLI task list lives in:

- [CLI_TODO.md](CLI_TODO.md)
