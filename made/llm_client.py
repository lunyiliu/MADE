"""OpenAI-compatible chat client used by every MADE agent.

Connection settings are read from environment variables:

    MADE_API_KEY     API key for an OpenAI-compatible endpoint (required)
    MADE_BASE_URL    Base URL of the endpoint (default: https://api.openai.com/v1)
    MADE_MODEL       Model name (default: gemini-3-flash)
    MADE_TEMPERATURE Sampling temperature (default: 0.1)
    MADE_MAX_TOKENS  Default max output tokens (default: 4096)

The client exposes the small surface the pipeline relies on:

    client.config.max_tokens / .temperature / .model / .stage
    client.chat(messages, sample_count=1, tools=None) -> dict

`chat` returns a dict with keys: text, tool_calls, finish_reason, usage,
status, error, error_type, sample_count.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

MAX_RETRIES = 4
REQUEST_TIMEOUT = 900

@dataclass
class LLMClientConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gemini-3-flash"
    temperature: float = 0.1
    max_tokens: int = 4096
    stage: str = ""
    caller: str = "made"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

class LLMClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        sample_count: int = 1,
        tools: list[dict[str, Any]] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        result: dict[str, Any] = {
            "status": None,
            "usage": None,
            "text": "",
            "tool_calls": None,
            "finish_reason": None,
            "error": None,
            "error_type": None,
            "sample_count": sample_count,
        }

        last_err: Exception | None = None
        resp = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    self.config.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if 500 <= resp.status_code < 600:
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                        resp = None
                        continue
                last_err = None
                break
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        if resp is None:
            result["error_type"] = type(last_err).__name__ if last_err else "RequestError"
            result["error"] = str(last_err) if last_err else "request failed"
            return result

        result["status"] = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}

        if isinstance(body, dict):
            result["usage"] = body.get("usage")
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                text = message.get("content")
                if not text:
                    text = message.get("reasoning_content") or message.get("reasoning") or ""
                result["text"] = text or ""
                result["tool_calls"] = message.get("tool_calls")
                result["finish_reason"] = choices[0].get("finish_reason")

        if not (200 <= resp.status_code < 300):
            result["error_type"] = "HTTPError"
            api_err = ""
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                api_err = str(body["error"].get("message") or "")
            result["error"] = f"HTTP {resp.status_code}" + (f": {api_err}" if api_err else "")

        return result

def client_from_env(caller: str = "made", model: str | None = None) -> LLMClient:
    api_key = os.environ.get("MADE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MADE_API_KEY is not set. Export it (and optionally MADE_BASE_URL / "
            "MADE_MODEL) to point MADE at an OpenAI-compatible endpoint."
        )
    config = LLMClientConfig(
        api_key=api_key,
        base_url=os.environ.get("MADE_BASE_URL", "https://api.openai.com/v1"),
        model=model or os.environ.get("MADE_MODEL", "gemini-3-flash"),
        temperature=float(os.environ.get("MADE_TEMPERATURE", "0.1")),
        max_tokens=int(os.environ.get("MADE_MAX_TOKENS", "4096")),
        caller=caller,
    )
    return LLMClient(config)
