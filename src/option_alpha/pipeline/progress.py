"""Progress reporting models for scan pipeline execution.

Provides structured progress tracking for each phase and overall scan,
with an async callback type for real-time updates (e.g. WebSocket).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel, Field


class PhaseStatus(str, Enum):
    """Status of an individual pipeline phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PhaseProgress(BaseModel):
    """Progress state for a single pipeline phase."""

    phase_name: str
    status: PhaseStatus = PhaseStatus.PENDING
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    ticker_count: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""


class ScanProgress(BaseModel):
    """Overall scan progress aggregating all phases."""

    phases: list[PhaseProgress] = Field(default_factory=list)
    overall_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_phase: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    elapsed_total: float = 0.0


# Async callback type for progress updates (e.g. WebSocket push).
ProgressCallback = Callable[[ScanProgress], Awaitable[None]]
