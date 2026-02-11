"""Unified LLM client abstraction for Ollama and Claude backends.

Provides a common interface for chat completions with optional
structured output via the instructor library or manual JSON parsing.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Try to import instructor for structured output extraction
_INSTRUCTOR_AVAILABLE = False
try:
    import instructor  # noqa: F401

    _INSTRUCTOR_AVAILABLE = True
    logger.debug("instructor library available for structured output")
except ImportError:
    logger.debug("instructor library not available; using manual JSON parsing")


def _extract_json_from_text(text: str) -> str:
    """Extract JSON object from LLM response text.

    Handles cases where the LLM wraps JSON in markdown code fences
    or includes extra text around it.
    """
    # Try to find JSON in code fences first
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + len("```")
        end = text.index("```", start)
        return text[start:end].strip()

    # Try to find a JSON object directly
    brace_start = text.find("{")
    if brace_start == -1:
        return text.strip()

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]

    # Fallback: return from first brace to end
    return text[brace_start:].strip()


def _parse_structured_output(text: str, response_model: type[T]) -> T:
    """Parse LLM text response into a Pydantic model via JSON extraction."""
    json_str = _extract_json_from_text(text)
    data = json.loads(json_str)
    return response_model.model_validate(data)


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
    """Async client for local Ollama API."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health_check(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
    ) -> str | T:
        """Send chat completion to Ollama.

        If response_model is provided and instructor is available, uses
        instructor for structured extraction. Otherwise falls back to
        manual JSON parsing.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if response_model is not None:
            # Ask for JSON output
            schema = response_model.model_json_schema()
            payload["format"] = "json"
            # Append schema hint to last message
            schema_hint = (
                f"\n\nRespond with a JSON object matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```"
            )
            payload["messages"] = [*messages[:-1], {
                "role": messages[-1]["role"],
                "content": messages[-1]["content"] + schema_hint,
            }]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        text = data.get("message", {}).get("content", "")

        if response_model is None:
            return text

        return _parse_structured_output(text, response_model)


class ClaudeClient(LLMClient):
    """Async client for Anthropic Claude API via httpx."""

    API_URL = "https://api.anthropic.com/v1/messages"
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

    async def health_check(self) -> bool:
        """Check if Anthropic API is reachable (simple connectivity test)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://api.anthropic.com/")
                # Any non-connection-error response means reachable
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
    ) -> str | T:
        """Send chat completion to Claude API.

        Converts messages to Anthropic format (system prompt separate).
        """
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

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": conversation,
        }
        if system_text:
            payload["system"] = system_text

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.API_URL,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

        data = resp.json()
        # Extract text from first content block
        content_blocks = data.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        if response_model is None:
            return text

        return _parse_structured_output(text, response_model)


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
