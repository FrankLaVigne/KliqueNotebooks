"""Offline contracts for the shared transport and single-turn UI callback."""
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError, URLError

from gateway_client import GatewayClient, GatewayConfig, GatewayError, _NoRedirect
from playground import FixtureClient, answer, build_demo


class Opener:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.payload).encode())


def config(**changes):
    values = dict(base_url="https://example.test/llm", profile="lab", models=("alias-a", "alias-b"), token="secret-test-token")
    values.update(changes)
    return GatewayConfig(**values)


class GatewayTests(unittest.TestCase):
    def test_env_only_and_explicit_aliases(self):
        env = {"KLIQUE_BASE_URL": "https://example.test/endpoint/v1", "KLIQUE_PROFILE": "lab",
               "KLIQUE_GATEWAY_TOKEN": "private", "KLIQUE_MODEL_ALIASES": "one, two,one"}
        cfg = GatewayConfig.from_environment(env, "/tmp/klique-no-such-env-05")
        self.assertEqual(cfg.base_url, "https://example.test/llm")
        self.assertEqual(cfg.models, ("one", "two"))
        self.assertNotIn("private", repr(cfg))
        env.pop("KLIQUE_MODEL_ALIASES")
        env["KLIQUE_MODEL"] = "direct-endpoint-not-an-alias"
        with self.assertRaises(ValueError):
            GatewayConfig.from_environment(env, "/tmp/klique-no-such-env-05")

    def test_env_precedence_quotes_and_no_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / ".env"
            p.write_text('KLIQUE_BASE_URL=https://example.test/v1\nKLIQUE_PROFILE="file-profile"\n'
                         "KLIQUE_GATEWAY_TOKEN='file-secret'\nKLIQUE_MODEL_ALIASES=one\n")
            cfg = GatewayConfig.from_environment({"KLIQUE_PROFILE": "injected"}, p)
            self.assertEqual(cfg.profile, "injected")
            self.assertEqual(cfg.token, "file-secret")

    def test_unsafe_destinations_rejected(self):
        for base in ("http://example.test/llm", "https://user:secret@example.test/llm",
                     "https://example.test/llm?key=secret", "https://example.test/not-llm"):
            with self.assertRaises(ValueError):
                GatewayConfig.from_environment({"KLIQUE_LLM_GATEWAY_URL": base}, "/tmp/klique-no-such-env-05")
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test"))

    def test_exact_route_headers_and_unknown_usage(self):
        opener = Opener({"choices": [{"message": {"content": "hello"}}]})
        result = GatewayClient(config(), opener=opener).complete([{"role": "user", "content": "hi"}], model="alias-b")
        req, timeout = opener.calls[0]
        self.assertEqual(req.full_url, "https://example.test/llm/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), "Bearer secret-test-token")
        self.assertEqual(req.get_header("X-clearml-profile"), "lab")
        self.assertEqual(json.loads(req.data)["model"], "alias-b")
        self.assertNotIn("usage", result)
        self.assertEqual(timeout, 60)

    def test_error_classification_no_retries_no_body_leaks(self):
        for code, label, expected in [(429, "budget-code", "budget"), (429, "rate-code", "rate_limit"),
                                      (429, "unknown", "ambiguous_limit"), (401, "secret-test-token", "authentication"),
                                      (503, "secret-test-token", "upstream"), (302, "", "http_error")]:
            error = HTTPError("https://example.test", code, "error", {},
                              io.BytesIO(json.dumps({"error": {"code": label}}).encode()))
            opener = Opener(error=error)
            with self.assertRaises(GatewayError) as caught:
                GatewayClient(config(rate_limit_code="rate-code", budget_code="budget-code"), opener=opener).complete([{}])
            self.assertEqual(caught.exception.kind, expected)
            self.assertNotIn("secret-test-token", str(caught.exception))
            self.assertEqual(len(opener.calls), 1)

    def test_identical_codes_stay_ambiguous(self):
        error = HTTPError("https://example.test", 429, "limit", {}, io.BytesIO(b'{"error":{"code":"same"}}'))
        with self.assertRaises(GatewayError) as caught:
            GatewayClient(config(rate_limit_code="same", budget_code="same"), opener=Opener(error=error)).complete([{}])
        self.assertEqual(caught.exception.kind, "ambiguous_limit")

    def test_invalid_responses_and_timeout(self):
        for payload in ({}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]}):
            with self.assertRaises(GatewayError) as caught:
                GatewayClient(config(), opener=Opener(payload)).complete([{}])
            self.assertEqual(caught.exception.kind, "invalid_response")
        opener = Opener(error=URLError("secret-test-token"))
        with self.assertRaises(GatewayError) as caught:
            GatewayClient(config(), opener=opener).complete([{}])
        self.assertEqual(caught.exception.kind, "transport")
        self.assertNotIn("secret-test-token", str(caught.exception))

    def test_request_bounds(self):
        client = GatewayClient(config(), opener=Opener())
        for kwargs in ({"model": "unlisted"}, {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": 2049},
                       {"temperature": float("nan")}, {"temperature": 3}):
            with self.assertRaises(ValueError):
                client.complete([{}], **kwargs)


class PlaygroundTests(unittest.TestCase):
    def test_fixture_and_validation(self):
        result, meta = answer(FixtureClient(), ("demo",), "hello", "demo", 0, 128)
        self.assertIn("OFFLINE FIXTURE", result)
        self.assertTrue(meta["offline_fixture"])
        self.assertIsNone(meta["usage_reported"])
        for prompt, alias in [("", "demo"), ("x" * 4001, "demo"), ("hello", "unlisted")]:
            self.assertFalse(answer(FixtureClient(), ("demo",), prompt, alias, 0, 128)[1]["sent"])

    def test_error_callback(self):
        class RefusingClient:
            def complete(self, *args, **kwargs):
                raise GatewayError("Gateway refused", status_code=429, kind="budget")
        _, meta = answer(RefusingClient(), ("demo",), "hello", "demo", 0, 128)
        self.assertEqual(meta["error_kind"], "budget")

    def test_ui_builds(self):
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("Install requirements.txt to check actual Gradio construction")
        demo = build_demo(FixtureClient(), ("demo",))
        self.assertEqual(demo.title, "Klique: Governed AI Playground")


if __name__ == "__main__":
    unittest.main()
