"""vLLM OpenAI-compatible chat client.

This client talks to a running ``vllm serve`` process through the
OpenAI-compatible ``/v1/chat/completions`` endpoint. It intentionally uses the
Python standard library so local development does not need the OpenAI SDK.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from vmp_memos.llm.base import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeFloat, SchemaModel


class VLLMClientConfig(SchemaModel):
    """Connection settings for a vLLM OpenAI-compatible server."""

    base_url: NonEmptyStr = "http://127.0.0.1:8000/v1"
    model: NonEmptyStr = "Qwen/Qwen2.5-7B-Instruct"
    api_key: NonEmptyStr | None = None
    timeout_seconds: NonNegativeFloat = 120.0
    max_retries: int = Field(default=2, ge=0)
    retry_sleep_seconds: NonNegativeFloat = 1.0
    generation: LLMGenerationConfig = Field(default_factory=LLMGenerationConfig)

    @classmethod
    def from_env(cls) -> "VLLMClientConfig":
        """Build config from environment variables used by the server scripts."""

        default_base_url = str(cls.model_fields["base_url"].default)
        default_model = str(cls.model_fields["model"].default)
        return cls(
            base_url=os.getenv("VMP_LLM_BASE_URL", default_base_url),
            model=os.getenv("VMP_LLM_MODEL", default_model),
            api_key=os.getenv("VMP_LLM_API_KEY") or None,
            timeout_seconds=float(os.getenv("VMP_LLM_TIMEOUT_SECONDS", "120")),
        )


class VLLMClient:
    """Synchronous vLLM chat client."""

    provider = "vllm"

    def __init__(self, config: VLLMClientConfig | None = None) -> None:
        self.config = config or VLLMClientConfig.from_env()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        """Call ``/v1/chat/completions`` and return normalized text."""

        if not messages:
            raise ValueError("at least one chat message is required")
        generation = generation or self.config.generation
        payload = {
            "model": self.config.model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "stream": False,
        }
        if generation.stop:
            payload["stop"] = list(generation.stop)
        raw_response = self._post_json(self._chat_url(), payload)
        return _parse_chat_response(
            raw_response,
            provider=self.provider,
            model=self.config.model,
        )

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str = "You are a concise, helpful assistant.",
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        """Convenience wrapper for a single user prompt."""

        return self.chat(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=prompt),
            ],
            generation=generation,
        )

    def list_models(self) -> dict[str, Any]:
        """Call ``/v1/models`` to verify server connectivity."""

        request = urllib.request.Request(
            self._models_url(),
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(
            request,
            timeout=float(self.config.timeout_seconds),
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=float(self.config.timeout_seconds),
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                message = _http_error_message(exc)
                if exc.code < 500 or attempt >= self.config.max_retries:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
            time.sleep(float(self.config.retry_sleep_seconds))
        raise RuntimeError(f"vLLM request failed after retries: {last_error}") from last_error

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _chat_url(self) -> str:
        return f"{_normalize_base_url(self.config.base_url)}/chat/completions"

    def _models_url(self) -> str:
        return f"{_normalize_base_url(self.config.base_url)}/models"


def load_vllm_config(path: str | Path) -> VLLMClientConfig:
    """Load a flat YAML config file for the vLLM client."""

    config_path = Path(path).expanduser().resolve()
    values = _load_yaml_mapping(config_path)
    values.pop("provider", None)
    api_key_env = str(values.pop("api_key_env", "") or "")
    if api_key_env and not values.get("api_key"):
        values["api_key"] = os.getenv(api_key_env) or None
    generation_keys = {"max_tokens", "temperature", "top_p", "stop"}
    generation_values = {
        key: values.pop(key)
        for key in list(values)
        if key in generation_keys
    }
    if generation_values:
        values["generation"] = LLMGenerationConfig.model_validate(generation_values)
    return VLLMClientConfig.model_validate(values)


def _parse_chat_response(
    raw_response: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> LLMResponse:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"vLLM response did not include choices: {raw_response}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"vLLM choice has unexpected shape: {first_choice}")
    message = first_choice.get("message", {})
    text = ""
    if isinstance(message, dict):
        text = str(message.get("content", "") or "")
    if not text and "text" in first_choice:
        text = str(first_choice.get("text", "") or "")
    return LLMResponse(
        provider=provider,
        model=str(raw_response.get("model") or model),
        text=text,
        finish_reason=(
            str(first_choice["finish_reason"])
            if first_choice.get("finish_reason") is not None
            else None
        ),
        usage=raw_response.get("usage", {}) if isinstance(raw_response.get("usage"), dict) else {},
        raw_response=raw_response,
    )


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8")
    except Exception:
        body = ""
    return f"vLLM HTTP {exc.code}: {body or exc.reason}"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml_mapping(text)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"LLM config must be a mapping: {path}")
    return dict(loaded)


def _parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError("YAML list item without a preceding key")
            values.setdefault(current_list_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        current_list_key = None
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            values[key] = []
            current_list_key = key
        else:
            values[key] = _parse_scalar(raw_value)
    return values


def _parse_scalar(value: str) -> Any:
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
