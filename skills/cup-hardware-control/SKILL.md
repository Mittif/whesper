---
name: cup-hardware-control
description: Use this skill when working in the whesper repository and the user wants the agent to inspect or control the CUP hardware through the local client API at http://localhost:3001/api/*. Covers status checks, motor speed changes, stop requests, LED mood adjustments, and relative intensity nudges such as “再刺激一点” or “温柔一点”.
---

# CUP Hardware Control

This repo already exposes the device through the `control_cup` tool when `[hardware.cup]` is enabled in config.

The runtime path should go through the local client API at `http://localhost:3001/api/*`. In config, prefer `base_url = "http://localhost:3001"`. The code also tolerates `http://localhost:3001/api`.

## Use This Skill When

- The user explicitly asks to inspect or control the CUP hardware
- The conversation is already in a CUP-control context and the user gives a relative follow-up like `再刺激一点`、`慢一点`、`停一下`
- You need to map natural-language intensity requests onto motor speed plus LED cues

## Source Of Truth

- Treat `hardware.md` as the primary device spec
- [`hardware/http_api.md`](../../hardware/http_api.md) mirrors the current HTTP section

## Workflow

1. Use `control_cup` with `action="status"` when device state is unknown.
2. Use `action="nudge_intensity"` for relative changes like `再刺激一点` or `温柔一点`.
3. Use `action="apply_scene"` for broad mode shifts like `gentle`, `steady`, `intense`, or `cooldown`.
4. Use `action="stop"` immediately for pause, stop, or safety-oriented requests.
5. Use `action="set_led"` or `action="set_motor"` only when the user wants a more direct/manual adjustment.

## Intent Mapping

- `再刺激一点` -> `{"action":"nudge_intensity","direction":"up"}`
- `温柔一点` -> `{"action":"nudge_intensity","direction":"down"}`
- `切到温柔模式` -> `{"action":"apply_scene","scene":"gentle"}`
- `切到强一点的模式` -> `{"action":"apply_scene","scene":"intense"}`
- `停一下` -> `{"action":"stop"}`

## Guardrails

- Only perform real device actions when the user clearly asked for them, or when the current conversation obviously continues an active CUP-control flow
- Do not invent undocumented vibration-pattern endpoints; the documented API exposes motor speed, stop, LED settings, status, config, and OTA
- If the user asks for more advanced patterns, approximate them with scene changes or repeated intensity nudges, and say that the current firmware API does not expose native pattern primitives
