"""Unified LLM client abstraction for Ollama and Claude backends.

Provides a common interface for chat completions with optional
structured output via official provider SDKs.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TypeVar

import anthropic
import ollama
from pydantic import BaseModel

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _build_example_hint(response_model: type[BaseModel]) -> str:
    """Build a JSON schema hint to append to user messages for Ollama."""
    schema = response_model.model_json_schema()
    return (
        f"\n\nRespond with a JSON object matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```"
    )


class LLMClient(ABC):
    """Abstract base for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
    ) -> str | T:
        """Send a chat completion request.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            response_model: Optional Pydantic model for structured output.

        Returns:
            Plain text string or validated Pydantic model instance.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the backend is reachable."""


class OllamaClient(LLMClient):
    """Async client for local Ollama API using the official SDK."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = ollama.AsyncClient(host=self.base_url)

    async def health_check(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            resp = await self._client.list()
            return any(m.model == self.model for m in resp.models)
        except Exception:
            return False

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
    ) -> str | T:
        """Send chat completion to Ollama via SDK."""
        call_messages = list(messages)
        fmt = None

        if response_model is not None:
            fmt = "json"
            hint = _build_example_hint(response_model)
            call_messages = [*messages[:-1], {
                "role": messages[-1]["role"],
                "content": messages[-1]["content"] + hint,
            }]

        resp = await self._client.chat(
            model=self.model,
            messages=call_messages,
            format=fmt,
        )

        text = resp.message.content or ""

        if response_model is None:
            return text

        data = json.loads(text)
        return response_model.model_validate(data)


class ClaudeClient(LLMClient):
    """Async client for Anthropic Claude API using the official SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("Claude API key is required")
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout,
        )

    async def health_check(self) -> bool:
        """Check if Anthropic API is reachable."""
        try:
            await self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
    ) -> str | T:
        """Send chat completion to Claude API via SDK."""
        # Separate system message from conversation
        system_text = ""
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                conversation.append(msg)

        if response_model is not None:
            schema = response_model.model_json_schema()
            schema_hint = (
                f"\n\nRespond ONLY with a JSON object matching this schema "
                f"(no markdown fences, no extra text):\n"
                f"{json.dumps(schema, indent=2)}"
            )
            if conversation:
                conversation[-1] = {
                    "role": conversation[-1]["role"],
                    "content": conversation[-1]["content"] + schema_hint,
                }

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": conversation,
        }
        if system_text:
            kwargs["system"] = system_text

        resp = await self._client.messages.create(**kwargs)

        # Extract text from content blocks
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text

        if response_model is None:
            return text

        data = json.loads(text)
        return response_model.model_validate(data)


def get_client(config: Settings | None = None) -> LLMClient:
    """Factory: return the configured LLM backend client.

    Args:
        config: Application settings. If None, loads from defaults.

    Returns:
        An LLMClient instance (OllamaClient or ClaudeClient).

    Raises:
        ValueError: If backend is 'claude' but no API key configured.
    """
    if config is None:
        config = get_settings()

    if config.ai_backend == "claude":
        if not config.claude_api_key:
            raise ValueError(
                "Claude backend selected but no API key configured. "
                "Set OPTION_ALPHA_CLAUDE_API_KEY or config.claude_api_key."
            )
        return ClaudeClient(api_key=config.claude_api_key)

    # Default: Ollama
    return OllamaClient(model=config.ollama_model)
