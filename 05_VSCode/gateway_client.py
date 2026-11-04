"""Small, dependency-free Klique LLM Gateway client for the Part 05 labs.

No import-time requests, automatic retries, secret logging, or alias discovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


class GatewayError(RuntimeError):
    def __init__(self, message, *, status_code=None, kind="request_error"):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind


def read_env_file(path):
    """Read simple KEY=value lines; not a shell and never expands variables."""
    values = {}
    if not path.exists():
        return values
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid KEY=value syntax at .env line {line_number}; values were not printed.")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unmatched quote at .env line {line_number}.")
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class GatewayConfig:
    base_url: str
    profile: str
    models: tuple[str, ...]
    token: str = field(repr=False)
    rate_limit_code: str | None = None
    budget_code: str | None = None

    @classmethod
    def from_environment(cls, env=None, env_file=None):
        values = read_env_file(Path(env_file) if env_file is not None else ROOT / ".env")
        values.update(dict(os.environ if env is None else env))
        base = values.get("KLIQUE_LLM_GATEWAY_URL", "").rstrip("/")
        if not base:
            endpoint = urlsplit(values.get("KLIQUE_BASE_URL", ""))
            if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError("KLIQUE_BASE_URL must not contain credentials, a query, or a fragment.")
            base = f"{endpoint.scheme}://{endpoint.netloc}/llm"
        url = urlsplit(base)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path != "/llm"):
            raise ValueError("Set an HTTPS KLIQUE_LLM_GATEWAY_URL ending /llm, or KLIQUE_BASE_URL from Section 1.")
        profile = values.get("KLIQUE_PROFILE", "").strip()
        token = values.get("KLIQUE_GATEWAY_TOKEN", "").strip()
        aliases = values.get("KLIQUE_MODEL_ALIASES", "") or values.get("KLIQUE_GATEWAY_MODEL", "")
        models = tuple(dict.fromkeys(a.strip() for a in aliases.split(",") if a.strip()))
        if not profile or not token or not models:
            raise ValueError("Set KLIQUE_PROFILE, KLIQUE_MODEL_ALIASES, and KLIQUE_GATEWAY_TOKEN; no values were printed.")
        if any("\n" in v or "\r" in v for v in (profile, token, *models)):
            raise ValueError("Configuration values cannot contain newlines.")
        return cls(base, profile, models, token, values.get("KLIQUE_RATE_LIMIT_CODE") or None,
                   values.get("KLIQUE_BUDGET_CODE") or None)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a Bearer token to a redirected destination.


class GatewayClient:
    def __init__(self, config, *, timeout=60, opener=None):
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("timeout must be between 0 and 120 seconds")
        self.config = config
        self.timeout = timeout
        self._opener = opener or build_opener(_NoRedirect())

    def complete(self, messages, *, model=None, max_tokens=256, temperature=0, tools=None, tool_choice=None):
        model = model or self.config.models[0]
        if model not in self.config.models:
            raise ValueError("Choose an alias from KLIQUE_MODEL_ALIASES.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be an integer from 1 to 2048")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        request = Request(self.config.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + self.config.token,
                                   "X-ClearML-Profile": self.config.profile, "Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(4_000_001)
        except HTTPError as exc:
            code = None
            try:
                error = json.loads(exc.read(8192)).get("error", {})
                if isinstance(error, dict):
                    code = error.get("code") or error.get("type")
            except (ValueError, AttributeError):
                pass
            kind = "http_error"
            if exc.code == 429:
                kind = "ambiguous_limit"
                if self.config.rate_limit_code != self.config.budget_code:
                    if self.config.budget_code and code == self.config.budget_code:
                        kind = "budget"
                    elif self.config.rate_limit_code and code == self.config.rate_limit_code:
                        kind = "rate_limit"
            elif exc.code in {401, 403}:
                kind = "authentication"
            elif exc.code in {502, 503, 504}:
                kind = "upstream"
            raise GatewayError(f"Gateway HTTP {exc.code} ({kind}); no retry was sent. Check profile, alias and AI Usage.",
                               status_code=exc.code, kind=kind) from None
        except (URLError, TimeoutError, OSError):
            raise GatewayError("Request failed or timed out. It may have reached the gateway; inspect AI Usage before retrying.",
                               kind="transport") from None
        if len(raw) > 4_000_000:
            raise GatewayError("Gateway response exceeded the lab's size limit.", kind="invalid_response")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise GatewayError("Gateway returned non-JSON content.", kind="invalid_response") from None
        if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
            raise GatewayError("Gateway returned no choices.", kind="invalid_response")
        first = result["choices"][0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise GatewayError("Gateway returned no assistant message.", kind="invalid_response")
        return result
