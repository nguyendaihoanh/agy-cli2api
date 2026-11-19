"""Offline unit tests for the agy-cli2api Hermes model-provider plugin."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _install_stub_providers() -> list:
    registered = []
    providers_mod = types.ModuleType("providers")
    providers_base_mod = types.ModuleType("providers.base")

    @dataclass
    class ProviderProfile:
        name: str
        api_mode: str = "chat_completions"
        aliases: tuple = ()
        display_name: str = ""
        description: str = ""
        signup_url: str = ""
        hidden: bool = False
        env_vars: tuple = ()
        base_url: str = ""
        models_url: str = ""
        auth_type: str = "api_key"
        supports_streaming: bool | None = None
        supports_health_check: bool = True
        supports_model_listing: bool = True
        supports_vision: bool = False
        default_aux_model: str = ""
        process_command: str = ""
        process_args: tuple = ()
        process_command_env_vars: tuple = ()
        fallback_models: tuple = ()
        model_aliases: dict = field(default_factory=dict)
        unsupported_response_formats: tuple = ()
        static_models: tuple = ()
        base_url_env_var: str = ""

    def register_provider(profile):
        registered.append(profile)
        return profile

    providers_mod.register_provider = register_provider
    providers_base_mod.ProviderProfile = ProviderProfile
    sys.modules["providers"] = providers_mod
    sys.modules["providers.base"] = providers_base_mod
    return registered


def _load_plugin_pkg():
    registered = _install_stub_providers()
    spec = importlib.util.spec_from_file_location(
        "agy_cli2api_plugin",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agy_cli2api_plugin"] = mod
    spec.loader.exec_module(mod)
    return mod, registered


class AgyCli2ApiPluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin_mod, cls.registered = _load_plugin_pkg()

    def test_provider_registered(self):
        self.assertEqual(len(self.registered), 1)
        prof = self.registered[0]
        self.assertEqual(prof.name, "agy-cli2api")
        self.assertEqual(prof.auth_type, "external_process")
        self.assertIn("gemini-4-hydrogen-xhigh", prof.fallback_models)

    def test_tool_call_extraction_with_nested_closing_tag_text(self):
        from agy_cli2api_plugin.prompt_tools import extract_tool_calls

        raw = (
            "Checking now.\n"
            '<tool_call>{"name":"read_file","arguments":{"path":"/a","note":"example </tool_call> inside"}}</tool_call>\n'
            '<tool_call>{"function":{"name":"terminal","arguments":{"command":"echo ok"}}}</tool_call>'
        )
        clean, calls = extract_tool_calls(raw)
        self.assertEqual(clean, "Checking now.")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].function.name, "read_file")
        self.assertEqual(calls[1].function.name, "terminal")
        self.assertEqual(calls[0].index, 0)
        self.assertEqual(calls[1].index, 1)

    def test_account_pool_rotation_and_masking(self):
        from agy_cli2api_plugin import account_manager as am

        self.assertEqual(am.mask_email("example.user@example.com"), "ex***@example.com")
        with tempfile.TemporaryDirectory() as td:
            old_prof = os.environ.get("AGY_PROFILES_DIR")
            old_cli = os.environ.get("AGY_CLI_DIR")
            old_pref = os.environ.get("AGY_PREFERRED_ACCOUNTS")
            old_excl = os.environ.get("AGY_EXCLUDED_ACCOUNTS")
            old_state = os.environ.get("AGY_STATE_FILE")
            try:
                for email in ("first@example.com", "second@example.com", "skip@example.com"):
                    tok = Path(td) / "profiles" / email / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
                    tok.parent.mkdir(parents=True, exist_ok=True)
                    tok.write_text('{"token":"placeholder"}', encoding="utf-8")
                os.environ["AGY_PROFILES_DIR"] = str(Path(td) / "profiles")
                os.environ["AGY_CLI_DIR"] = str(Path(td) / "cli")
                os.environ["AGY_PREFERRED_ACCOUNTS"] = "first@example.com,second@example.com"
                os.environ["AGY_EXCLUDED_ACCOUNTS"] = "skip@example.com"
                os.environ["AGY_STATE_FILE"] = ".test-active-acc"
                avail = am.get_available_accounts()
                self.assertEqual(avail, ["first@example.com", "second@example.com"])
                self.assertEqual(am.get_active_account(), "first@example.com")
                nxt = am.rotate_to_next_account("first@example.com")
                self.assertEqual(nxt, "second@example.com")
            finally:
                for k, v in (
                    ("AGY_PROFILES_DIR", old_prof),
                    ("AGY_CLI_DIR", old_cli),
                    ("AGY_PREFERRED_ACCOUNTS", old_pref),
                    ("AGY_EXCLUDED_ACCOUNTS", old_excl),
                    ("AGY_STATE_FILE", old_state),
                ):
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_client_sync_async_and_stream_contract(self):
        import asyncio
        from unittest import mock
        from agy_cli2api_plugin.conversation_reader import AgResponse
        from agy_cli2api_plugin.runner import AgyRunResult

        fake_result = AgyRunResult(
            db_path=Path("conv-123.db"),
            response=AgResponse(
                answer='Done.<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>',
                reasoning="thought step",
            ),
            account="default",
        )
        prof = self.registered[0]
        client = prof.create_client()
        self.assertTrue(client.HERMES_SKIP_TRANSPORT_WRAP)
        self.assertTrue(client.HERMES_SKIP_ASYNC_WRAP)

        with mock.patch("agy_cli2api_plugin.client.run_agy_with_autoswap", return_value=fake_result):
            resp = client.chat.completions.create(
                model="gemini-3.8-flash-medium",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
            )
            self.assertEqual(resp.choices[0].message.content, "Done.")
            self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
            self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "read_file")

            async def _async_check():
                awaited = await client.chat.completions.create(
                    model="gemini-4-hydrogen-xhigh",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=False,
                )
                self.assertEqual(awaited.choices[0].message.content, "Done.")
                stream = await client.chat.completions.create(
                    model="gemini-3.8-flash-medium",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
                collected = [chunk async for chunk in stream]
                self.assertTrue(any(c.choices[0].delta.content == "Done." for c in collected))
                self.assertEqual(collected[-1].choices[0].finish_reason, "tool_calls")

            asyncio.run(_async_check())

    def test_sanitize_error_redacts_pii(self):
        from agy_cli2api_plugin.client import sanitize_error_text

        cleaned = sanitize_error_text("Error reading /Users/alice/secret from bob@example.com with Bearer abc123==")
        self.assertNotIn("alice", cleaned)
        self.assertNotIn("bob@example.com", cleaned)
        self.assertNotIn("abc123==", cleaned)

    def test_session_store_account_scoping(self):
        from agy_cli2api_plugin.session_store import SessionStore, fingerprint

        store = SessionStore(10)
        sigs_a = [fingerprint("user", "hi a")]
        sigs_b = [fingerprint("user", "hi b")]
        store.remember("cid-a", "acc1@example.com", sigs_a)
        store.remember("cid-b", "acc2@example.com", sigs_b)
        store.forget_account("acc1@example.com")
        self.assertIsNone(store.lookup(sigs_a + [fingerprint("user", "next")], "acc1@example.com").conversation_id)
        self.assertEqual(
            store.lookup(sigs_b + [fingerprint("user", "next")], "acc2@example.com").conversation_id,
            "cid-b",
        )

    def test_dynamic_model_discovery_accepts_new_future_models(self):
        from unittest import mock
        from agy_cli2api_plugin import models as pm

        pm.clear_model_cache()
        fake_stdout = (
            "Fetching available models...\n"
            "gemini-4-neon-xhigh\tGemini 4 Neon (XHigh)\n"
            "gemini-5-nova-ultra\tGemini 5 Nova (Ultra)\n"
        )
        fake_proc = mock.Mock(returncode=0, stdout=fake_stdout)
        try:
            with mock.patch("agy_cli2api_plugin.models.os.path.exists", return_value=True), mock.patch(
                "agy_cli2api_plugin.models.subprocess.run", return_value=fake_proc
            ):
                ids = pm.discover_models("/fake/agy", force_refresh=True)
            self.assertEqual(ids, ["gemini-4-neon-xhigh", "gemini-5-nova-ultra"])
            self.assertEqual(pm.get_model_label("gemini-5-nova-ultra"), "Gemini 5 Nova (Ultra)")
            self.assertEqual(pm.resolve_agy_label("gemini-5-nova-ultra"), "Gemini 5 Nova (Ultra)")
            self.assertEqual(pm.resolve_agy_label("neon"), "Gemini 4 Neon (XHigh)")
            self.assertIn("xhigh", pm.supported_efforts_for("gemini-4-neon-high") or ())
        finally:
            pm.clear_model_cache()


if __name__ == "__main__":
    unittest.main()
