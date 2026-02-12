"""Error checking and messaging utilities for the web dashboard.

Provides health checks for external dependencies (Ollama, yfinance, Claude API)
and structured error/warning messages for the dashboard UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from option_alpha.config import Settings

logger = logging.getLogger(__name__)


class CheckSeverity(str, Enum):
    """Severity levels for health check results."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class HealthCheck:
    """Result of a single dependency health check."""

    name: str
    severity: CheckSeverity = CheckSeverity.OK
    message: str = ""
    detail: str = ""


@dataclass
class SystemStatus:
    """Aggregated system health status for dashboard display."""

    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(c.severity == CheckSeverity.ERROR for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.severity == CheckSeverity.WARNING for c in self.checks)

    @property
    def errors(self) -> list[HealthCheck]:
        return [c for c in self.checks if c.severity == CheckSeverity.ERROR]

    @property
    def warnings(self) -> list[HealthCheck]:
        return [c for c in self.checks if c.severity == CheckSeverity.WARNING]


async def check_ollama(settings: Settings) -> HealthCheck:
    """Check if Ollama is running by hitting its health endpoint.

    Only relevant when ai_backend is set to 'ollama'.
    """
    if settings.ai_backend != "ollama":
        return HealthCheck(name="ollama", severity=CheckSeverity.OK)

    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                return HealthCheck(
                    name="ollama",
                    severity=CheckSeverity.OK,
                    message="Ollama is running",
                )
    except Exception:
        pass

    return HealthCheck(
        name="ollama",
        severity=CheckSeverity.WARNING,
        message="Ollama not detected",
        detail=(
            "Ollama is not running on localhost:11434. "
            "Install from ollama.ai or switch to Claude API in Settings."
        ),
    )


def check_claude_api_key(settings: Settings) -> HealthCheck:
    """Check if Claude API key is configured when Claude backend is selected."""
    if settings.ai_backend != "claude":
        return HealthCheck(name="claude_api_key", severity=CheckSeverity.OK)

    if settings.claude_api_key:
        return HealthCheck(
            name="claude_api_key",
            severity=CheckSeverity.OK,
            message="Claude API key configured",
        )

    return HealthCheck(
        name="claude_api_key",
        severity=CheckSeverity.ERROR,
        message="Claude API key not configured",
        detail="Claude backend is selected but no API key is set. Add it in Settings.",
    )


def check_yfinance() -> HealthCheck:
    """Check if yfinance can reach Yahoo Finance servers."""
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY")
        info = spy.fast_info
        if hasattr(info, "last_price") and info.last_price is not None:
            return HealthCheck(
                name="yfinance",
                severity=CheckSeverity.OK,
                message="yfinance connectivity OK",
            )
    except Exception:
        pass

    return HealthCheck(
        name="yfinance",
        severity=CheckSeverity.WARNING,
        message="Market data unavailable",
        detail="Cannot connect to Yahoo Finance. Check your internet connection.",
    )


async def run_health_checks(
    settings: Settings,
    include_yfinance: bool = False,
) -> SystemStatus:
    """Run all relevant health checks and return aggregated status.

    Args:
        settings: Current application settings.
        include_yfinance: Whether to check yfinance connectivity
            (expensive, skip on normal page loads).

    Returns:
        SystemStatus with all check results.
    """
    status = SystemStatus()

    # Always check AI backend config
    ollama_check = await check_ollama(settings)
    status.checks.append(ollama_check)

    claude_check = check_claude_api_key(settings)
    status.checks.append(claude_check)

    # Only check yfinance when explicitly requested (e.g., on scan trigger)
    if include_yfinance:
        yf_check = check_yfinance()
        status.checks.append(yf_check)

    return status


def format_scan_error(error: Exception) -> str:
    """Format a scan failure into a user-friendly error message.

    Avoids leaking API keys or internal details in error messages.
    """
    msg = str(error)

    # Sanitize potential secrets from error messages
    if "api_key" in msg.lower() or "api-key" in msg.lower():
        return "Scan failed: authentication error. Check your API key in Settings."

    if "connect" in msg.lower() or "timeout" in msg.lower():
        return "Scan failed: connection error. Check your internet connection and Ollama status."

    if "rate limit" in msg.lower() or "429" in msg:
        return "Scan failed: rate limited. Wait a moment and try again."

    # Generic fallback -- truncate to avoid leaking internal details
    safe_msg = msg[:200] if len(msg) > 200 else msg
    return f"Scan failed: {safe_msg}"
