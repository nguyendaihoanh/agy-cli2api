# agy-cli2api Hermes Model-Provider Plugin

Native **Hermes Agent** (`kind: model-provider`) plugin packaging the multi-account failover and protobuf SQLite execution engine from [`nguyendaihoanh/agy-cli2api`](https://github.com/nguyendaihoanh/agy-cli2api).

> **Disclosure:** This plugin executes your local Google Antigravity `agy` CLI binary under the Google account profiles stored on your own machine (or optionally forwards calls to your local `agy-cli2api` HTTP gateway via `AGY_CLI2API_BASE_URL`). Automated subscription usage may be subject to Google Antigravity's Terms of Service — review them before running continuous unattended workloads.

---

## Key Features

- **Zero External Python Dependencies:** Self-contained stdlib-only plugin (`python_dependencies: []`); works immediately via `hermes plugins install` without setting up FastAPI or uvicorn.
- **Multi-Account Auto-Swap (0.2s Failover):** Automatically rotates across your local `~/.gemini_profiles/*` and `~/.gemini/antigravity-cli/antigravity-oauth-token.*` profiles upon `429 (RESOURCE_EXHAUSTED)` or `403 (PERMISSION_DENIED)` errors.
- **Protobuf SQLite Reader:** Decodes Antigravity CLI SQLite conversation payloads directly for complete answers and reasoning extraction.
- **Hermes Host Tool Calling:** Exposes Hermes tools cleanly to `agy` and parses `<tool_call>` blocks back into OpenAI-compatible `tool_calls`.
- **Optional HTTP Proxy Mode:** Set `AGY_CLI2API_BASE_URL=http://127.0.0.1:7862/v1` (and optional `AGY2API_KEY`) to route through a standalone `agy-cli2api` server instead of spawning `agy` per request.

---

## Installation

### From GitHub directly
```bash
hermes plugins install https://github.com/nguyendaihoanh/agy-cli2api#hermes-plugin/agy-cli2api
hermes plugins enable agy-cli2api
```

### Local development / validation
```bash
hermes plugins validate /path/to/agy-cli2api/hermes-plugin/agy-cli2api --install-deps
```

---

## Usage in Hermes

One-off test:
```bash
hermes chat -q "Hello" --provider agy-cli2api --model gemini-3.8-flash-medium
```

Or configure `~/.hermes/config.yaml`:
```yaml
model:
  provider: agy-cli2api
  default: gemini-3.8-flash-medium
```

Optional environment variables:
- `AGY_PATH`: Explicit path to the `agy` binary.
- `AGY_PROFILES_DIR`: Custom profile pool directory (default `~/.gemini_profiles`).
- `AGY_PREFERRED_ACCOUNTS`: Comma-separated email priority list.
- `AGY_EXCLUDED_ACCOUNTS`: Comma-separated email blocklist.
- `AGY2API_STATEFUL`: Set `true` to enable per-account `--conversation` delta resume.
- `AGY_CLI2API_BASE_URL` / `AGY2API_KEY`: Optional local OpenAI gateway override.

---

## Sample Marketplace Catalog Entry (`plugin-catalog/agy-cli2api.yaml`)

When opening a PR to `NousResearch/hermes-agent`, replace `<40-CHAR-COMMIT-SHA>` with the exact merged commit hash:

```yaml
name: agy-cli2api
title: Agy CLI2API (Multi-Account Antigravity)
maintainer: nguyendaihoanh
tier: community
category: models
repo: https://github.com/nguyendaihoanh/agy-cli2api
subdir: hermes-plugin/agy-cli2api
sha: <40-CHAR-COMMIT-SHA>
version: "1.0.0"
requires_hermes: ">=0.21.4"
platforms:
  - linux
  - macos
  - windows
description: >-
  Google Antigravity CLI multi-account provider with 0.2s quota auto-swap,
  protobuf SQLite extraction, and Hermes tool calling. Disclosure: executes
  local agy binary under the user's own signed-in profiles.
capabilities:
  provides_tools: []
  provides_hooks: []
  provides_middleware: []
  requires_env: []
```
