"""Unit and API tests for dynamic `agy models` discovery in agy-cli2api."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
import server  # noqa: E402


SAMPLE_AGY_OUTPUT = """Fetching available models...
gemini-4-neon-xhigh\tGemini 4 Neon (XHigh)
gemini-4-neon-high\tGemini 4 Neon (High)
gemini-pro-orchestrator-neon\tOrchestrator (Neon)
gemini-5-nova-ultra\tGemini 5 Nova (Ultra)
gemini-5-pulse-fast   Gemini 5 Pulse (Fast)
claude-opus-5-5-high\tClaude Opus 5.5 (High)
"""


class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        config.clear_model_cache()

    def tearDown(self) -> None:
        config.clear_model_cache()

    def test_parse_agy_models_output_handles_new_unknown_models(self) -> None:
        parsed = config.parse_agy_models_output(SAMPLE_AGY_OUTPUT)
        self.assertEqual(
            list(parsed.keys()),
            [
                "gemini-4-neon-xhigh",
                "gemini-4-neon-high",
                "gemini-pro-orchestrator-neon",
                "gemini-5-nova-ultra",
                "gemini-5-pulse-fast",
                "claude-opus-5-5-high",
            ],
        )
        self.assertEqual(parsed["gemini-5-nova-ultra"], "Gemini 5 Nova (Ultra)")
        self.assertEqual(parsed["gemini-5-pulse-fast"], "Gemini 5 Pulse (Fast)")
        self.assertNotIn("Fetching available models...", parsed)

    def test_discover_caches_and_refreshes_and_resolves_new_labels(self) -> None:
        first_catalog = {"gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)"}
        second_catalog = {
            "gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)",
            "gemini-5-nova-ultra": "Gemini 5 Nova (Ultra)",
        }
        with mock.patch.object(
            config, "_run_agy_models_cli", side_effect=[first_catalog, second_catalog]
        ) as cli_mock:
            models1 = config.get_exposed_models(force_refresh=False)
            models2 = config.get_exposed_models(force_refresh=False)
            self.assertEqual(models1, ["gemini-4-neon-xhigh"])
            self.assertEqual(models2, ["gemini-4-neon-xhigh"])
            self.assertEqual(cli_mock.call_count, 1)

            models3 = config.get_exposed_models(force_refresh=True)
            self.assertEqual(models3, ["gemini-4-neon-xhigh", "gemini-5-nova-ultra"])
            self.assertEqual(cli_mock.call_count, 2)
            self.assertEqual(config.resolve_model("gemini-5-nova-ultra"), "Gemini 5 Nova (Ultra)")
            self.assertEqual(config.resolve_model("brand-new-unlisted-id"), "brand-new-unlisted-id")

    def test_fallback_on_cli_failure_with_backoff(self) -> None:
        with mock.patch.object(config, "_run_agy_models_cli", return_value=None) as cli_mock:
            models1 = config.get_exposed_models(force_refresh=False)
            models2 = config.get_exposed_models(force_refresh=False)
            self.assertIn("gemini-4-neon-xhigh", models1)
            self.assertIn("gemini-4-hydrogen-xhigh", models1)
            self.assertEqual(models1, models2)
            # Second call hits fallback backoff instead of stalling again
            self.assertEqual(cli_mock.call_count, 1)

            # Explicit refresh retries immediately
            config.get_exposed_models(force_refresh=True)
            self.assertEqual(cli_mock.call_count, 2)

    def test_models_override_env_skips_cli(self) -> None:
        patched_settings = config.Settings(models_override=["custom-a", "custom-b"])
        object.__setattr__(patched_settings, "models_override", ["custom-a", "custom-b"])
        with mock.patch.object(config, "settings", patched_settings), mock.patch.object(
            config, "_run_agy_models_cli"
        ) as cli_mock:
            self.assertEqual(config.get_exposed_models(force_refresh=True), ["custom-a", "custom-b"])
            cli_mock.assert_not_called()

    def test_fastapi_v1_models_endpoint_and_refresh_auth(self) -> None:
        async def _run() -> None:
            test_settings = config.Settings(api_key="test-secret-key", require_auth=True)
            live_a = {"gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)"}
            live_b = {
                "gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)",
                "gemini-5-nova-ultra": "Gemini 5 Nova (Ultra)",
            }
            with mock.patch.object(server, "settings", test_settings), mock.patch.object(
                config, "settings", test_settings
            ), mock.patch.object(config, "_run_agy_models_cli", side_effect=[live_a, live_b]):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server.app), base_url="http://test"
                ) as client:
                    unauth = await client.get("/v1/models")
                    self.assertEqual(unauth.status_code, 401)

                    headers = {"Authorization": "Bearer test-secret-key"}
                    r1 = await client.get("/v1/models", headers=headers)
                    self.assertEqual(r1.status_code, 200)
                    self.assertEqual([m["id"] for m in r1.json()["data"]], ["gemini-4-neon-xhigh"])

                    r2 = await client.get("/v1/models?refresh=1", headers=headers)
                    self.assertEqual(r2.status_code, 200)
                    self.assertEqual(
                        [m["id"] for m in r2.json()["data"]],
                        ["gemini-4-neon-xhigh", "gemini-5-nova-ultra"],
                    )

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
