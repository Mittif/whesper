# Whesper CLI Todo

This file tracks the current CLI surface so it is easy to see what is already implemented and what still needs work.

## Completed

- Interactive chat loop with `prompt_toolkit`
- Input history with `Up/Down`
- Reverse history search with `Ctrl+R`
- Slash command parsing and `Tab` completion
- Session persistence in `.whesper/`
- Multi-model routing with config-driven providers and models
- Streaming output with animated `thinking...` state
- First-token timing in assistant output
- Better message rendering with dividers and color hierarchy
- Friendly config and model alias errors without exiting the CLI
- Auto-reset when a saved session contains an invalid pinned model
- Runtime model switching with `/model` and `/use`
- Routing override with `/mode`
- Session switching with `/new`
- Current screen clear with `/clear`
- Session rename with `/rename <session_id>`
- Session deletion with `/delete-session <session_id>`
- Runtime status view with `/status`
- Active model inspection with `/info`
- Retry the previous assistant turn with `/retry`
- Print the latest assistant reply as plain text with `/copy-last`
- Interrupt generation with `Ctrl+C` without leaving the CLI
- Rich reply footer meta with total time, first-token time, model, provider, route, and reply size
- Improved `/history` rendering with role-based colors and clearer layout

## Good Next Steps

- Add `/recent` for recently active sessions
- Add `/route` as a clearer alias for `/mode`
- Improve command descriptions in the completion menu

## Candidate Features

- `/test-model <alias>` to verify a model quickly
- `/provider` to inspect configured providers
- `/fallback <alias>` for automatic retry on provider failure
- `/export` to save the current session as `md` or `json`
- `/transcript` for a cleaner plain-text conversation dump
- `/theme` for alternate terminal color styles
- `/compact on|off` for dense versus card-style rendering
- Temporary prompt editing commands such as `/system` and `/persona`

## Notes

- Keep `README.md` project-facing for GitHub readers.
- Put local-only notes in ignored files such as `DEVELOPMENT_NOTES.local.md`.
- Keep private endpoints and secrets out of `whesper.example.toml`.
