<p align="center">
  <img src="assets/banner.svg" alt="agy-cli2api Banner" width="100%">
</p>

# agy-cli2api (Antigravity CLI to OpenAI API Gateway)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)

**agy-cli2api** (formerly agy2api) transforms the official Google Antigravity CLI (`agy`) into a lightweight, high-performance, drop-in replacement for OpenAI API endpoints (`/v1/chat/completions`, `/v1/models`).

---

## Key Features

- **Minimalist Pure-Python Architecture:** Ultra-lightweight (~80 KB), zero heavy dependencies, runs natively anywhere with Python 3.9+.
- **Multi-Account Auto-Swap (0.2s Failover):** Automatically detects `429 (Resource Exhausted)` or `403 (Permission Denied)` quota limits and hot-swaps tokens/profiles in 0.2 seconds without aborting client requests.
- **Protobuf SQLite Binary Parser:** Directly decodes Antigravity SQLite internal binary payloads for rapid response extraction.
- **Hybrid Stateful Session Store:** Supports `--conversation <id>` context reuse backed by SQLite isolation, with an instant fallback to stateless full-prompt if a turn is corrupted.
- **Reasoning Leakage Suppression:** Suppresses raw thinking tokens when model answer is empty to prevent UI glitching.
- **Headless Updater Quiet Mode:** Auto-touches `last_check.timestamp` across all profile roots to suppress background updater popups in headless environments.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/nguyendaihoanh/agy-cli2api.git
cd agy-cli2api
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

### 3. Run the Server

```bash
uvicorn server:app --host 127.0.0.1 --port 7862
```

### 4. Test with cURL

```bash
curl http://127.0.0.1:7862/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_strong_secret_api_key_here" \
  -d '{
    "model": "gemini-3.5-flash",
    "messages": [{"role": "user", "content": "Hello Antigravity!"}]
  }'
```

---

## Maintainers & Credits

- **Maintainer:** [@nguyendaihoanh](https://github.com/nguyendaihoanh)
- **Base Upstream Credit:** GhostXia ([GhostXia/agy2api](https://github.com/GhostXia/agy2api))

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
