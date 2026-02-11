"""Shared Pydantic models for Option Alpha."""

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Trade direction."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ScanStatus(str, Enum):
    """Status of a scan run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class FetchErrorType(str, Enum):
    """Classification of ticker fetch failures."""

    DELISTED = "delisted"
    RATE_LIMITED = "rate_limited"
    NETWORK = "network"
    INSUFFICIENT_DATA = "insufficient_data"
    UNKNOWN = "unknown"


class TickerData(BaseModel):
    """OHLCV data container for a single ticker."""

    symbol: str
    dates: list[datetime] = Field(default_factory=list)
    open: list[float] = Field(default_factory=list)
    high: list[float] = Field(default_factory=list)
    low: list[float] = Field(default_factory=list)
    close: list[float] = Field(default_factory=list)
    volume: list[int] = Field(default_factory=list)
    last_price: Optional[float] = None
    avg_volume: Optional[float] = None


class ScoreBreakdown(BaseModel):
    """Per-indicator score breakdown with full transparency."""

    name: str
    raw_value: float
    normalized: float = Field(ge=0, le=100, description="Percentile rank 0-100")
    weight: float = Field(ge=0, le=1)
    contribution: float = Field(description="normalized * weight")


class TickerScore(BaseModel):
    """Composite score for a single ticker with full breakdown."""

    symbol: str
    composite_score: float = Field(ge=0, le=100)
    breakdown: list[ScoreBreakdown] = Field(default_factory=list)
    direction: Direction = Direction.NEUTRAL
    last_price: Optional[float] = None
    avg_volume: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentResponse(BaseModel):
    """Response from a single AI agent (bull, bear, or risk)."""

    role: str = Field(description="Agent role: bull, bear, or risk")
    analysis: str = Field(description="Free-text analysis from the agent")
    key_points: list[str] = Field(
        default_factory=list, description="Bullet-point key arguments"
    )
    conviction: Optional[int] = Field(
        default=None, ge=1, le=10, description="Conviction score 1-10"
    )


class TradeThesis(BaseModel):
    """Structured trade thesis from AI debate."""

    symbol: str
    direction: Direction
    conviction: int = Field(ge=1, le=10)
    entry_rationale: str
    risk_factors: list[str] = Field(default_factory=list)
    recommended_action: str = Field(
        description="e.g. 'Buy AAPL 180C 30DTE' or 'No trade'"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DebateResult(BaseModel):
    """Complete multi-agent debate result for a ticker."""

    symbol: str
    bull: AgentResponse
    bear: AgentResponse
    risk: AgentResponse
    final_thesis: TradeThesis


class OptionsRecommendation(BaseModel):
    """Specific options contract recommendation with Greeks."""

    symbol: str
    contract_symbol: Optional[str] = None
    direction: Direction
    option_type: str = Field(description="'call' or 'put'")
    strike: float
    expiry: datetime
    dte: int = Field(description="Days to expiration")
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid_price: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    underlying_price: Optional[float] = None


class ScanResult(BaseModel):
    """Complete scan output with all tickers, scores, and theses."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ticker_scores: list[TickerScore] = Field(default_factory=list)
    debate_results: list[DebateResult] = Field(default_factory=list)
    options_recommendations: list[OptionsRecommendation] = Field(default_factory=list)
    total_tickers_scanned: int = 0
    top_n_scored: int = 0
    top_n_debated: int = 0


class ScanRun(BaseModel):
    """Metadata about a scan run."""

    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ticker_count: int = 0
    duration_seconds: Optional[float] = None
    status: ScanStatus = ScanStatus.PENDING
    error_message: Optional[str] = None
    scores_computed: int = 0
    debates_completed: int = 0
    options_analyzed: int = 0
