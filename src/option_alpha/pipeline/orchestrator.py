"""Sequential pipeline orchestrator for the full scan workflow.

Runs six phases in order:
  1. Data Fetch  - Get universe, fetch OHLCV, cache
  2. Scoring     - Compute composite scores for all tickers
  3. Catalysts   - Fetch earnings dates, merge catalyst scores
  4. Options     - Fetch chains + recommend contracts for top N
  5. Persist     - Checkpoint save scores with PARTIAL status
  6. AI Debate   - Run multi-agent debate for top N, finalize scan

Handles partial failures gracefully: if individual tickers fail in any
phase the pipeline continues with available data. Phase timing is
tracked via time.perf_counter().
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Optional

import pandas as pd

from option_alpha.ai.clients import get_client
from option_alpha.ai.debate import DebateManager
from option_alpha.catalysts.earnings import batch_earnings_info, merge_catalyst_scores
from option_alpha.config import Settings, get_settings
from option_alpha.data.cache import load_batch, load_failure_cache, record_failures, save_batch
from option_alpha.data.fetcher import fetch_batch
from option_alpha.data.universe import get_scan_universe
from option_alpha.data.watchlists import get_active_watchlist
from option_alpha.models import (
    DebateResult,
    FetchErrorType,
    OptionsRecommendation,
    ScanResult,
    ScanRun,
    ScanStatus,
    TickerData,
    TickerScore,
)
from option_alpha.options.chains import fetch_chains_for_tickers
from option_alpha.options.recommender import recommend_for_scored_tickers
from option_alpha.persistence.database import initialize_db
from option_alpha.persistence.repository import (
    save_ai_theses,
    save_scan_run,
    save_ticker_scores,
    update_scan_run,
)
from option_alpha.pipeline.progress import (
    PhaseProgress,
    PhaseStatus,
    ProgressCallback,
    ScanProgress,
)

logger = logging.getLogger(__name__)

# Ordered phase names for progress tracking.
PHASE_NAMES = [
    "data_fetch",
    "scoring",
    "catalysts",
    "options",
    "persist",
    "ai_debate",
]


def _ticker_data_to_dataframe(td: TickerData) -> pd.DataFrame:
    """Convert a TickerData model to a pandas DataFrame with OHLCV columns.

    The DataFrame index is set to the dates and columns are capitalized
    ('Open', 'High', 'Low', 'Close', 'Volume') to match yfinance format
    expected by score_universe().
    """
    return pd.DataFrame(
        {
            "Open": td.open,
            "High": td.high,
            "Low": td.low,
            "Close": td.close,
            "Volume": td.volume,
        },
        index=pd.DatetimeIndex(td.dates, name="Date"),
    )


class ScanOrchestrator:
    """Orchestrates the full scan pipeline with progress reporting."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._phase_timings: dict[str, float] = {}

    async def run_scan(
        self,
        on_progress: Optional[ProgressCallback] = None,
        universe_override: Optional[list[str]] = None,
    ) -> ScanResult:
        """Execute the complete scan pipeline.

        Args:
            on_progress: Optional async callback invoked at each phase
                start and completion with the current ScanProgress.
            universe_override: Optional list of ticker symbols to scan
                instead of the configured universe. Useful for one-off
                custom scans (e.g., quick-add from dashboard).

        Returns:
            ScanResult containing scores, options recs, and debate results.
        """
        run_id = uuid.uuid4().hex[:12]
        scan_start = time.perf_counter()
        started_at = datetime.now(UTC)

        # Initialize progress state for all phases.
        progress = ScanProgress(
            phases=[PhaseProgress(phase_name=name) for name in PHASE_NAMES],
            started_at=started_at,
        )

        # Mutable containers for pipeline data flowing between phases.
        ohlcv_data: dict[str, TickerData] = {}
        ohlcv_frames: dict[str, pd.DataFrame] = {}
        ticker_scores: list[TickerScore] = []
        options_recs: list[OptionsRecommendation] = []
        debate_results: list[DebateResult] = []
        errors: list[str] = []

        # --- Phase 1: Data Fetch ---
        ohlcv_data, ohlcv_frames = await self._phase_data_fetch(
            progress, on_progress, scan_start, errors, universe_override,
        )

        # --- Phase 2: Scoring ---
        ticker_scores = await self._phase_scoring(
            ohlcv_frames, progress, on_progress, scan_start, errors,
        )

        # --- Phase 3: Catalysts ---
        ticker_scores = await self._phase_catalysts(
            ticker_scores, progress, on_progress, scan_start, errors,
        )

        # --- Phase 4: Options ---
        options_recs = await self._phase_options(
            ticker_scores, progress, on_progress, scan_start, errors,
        )

        # --- Phase 5: Persist (checkpoint) ---
        scan_db_id = await self._phase_persist(
            run_id, scan_start, ticker_scores, options_recs,
            progress, on_progress, errors,
        )

        # --- Phase 6: AI Debate ---
        debate_results = await self._phase_ai_debate(
            ticker_scores, options_recs, scan_db_id,
            progress, on_progress, scan_start, errors,
        )

        # Build final result.
        total_duration = time.perf_counter() - scan_start
        progress.overall_percentage = 100.0
        progress.elapsed_total = total_duration
        progress.current_phase = None
        if on_progress:
            await on_progress(progress)

        result = ScanResult(
            timestamp=started_at,
            ticker_scores=ticker_scores,
            debate_results=debate_results,
            options_recommendations=options_recs,
            total_tickers_scanned=len(ohlcv_data),
            top_n_scored=len(ticker_scores),
            top_n_debated=len(debate_results),
        )

        logger.info(
            "Scan complete: %d tickers scored, %d options recs, %d debates in %.1fs",
            len(ticker_scores),
            len(options_recs),
            len(debate_results),
            total_duration,
        )

        return result

    # ------------------------------------------------------------------
    # Individual phase implementations
    # ------------------------------------------------------------------

    async def _phase_data_fetch(
        self,
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
        errors: list[str],
        universe_override: Optional[list[str]] = None,
    ) -> tuple[dict[str, TickerData], dict[str, pd.DataFrame]]:
        """Phase 1: Get universe, check cache, fetch OHLCV, cache results."""
        phase_idx = 0
        await self._start_phase(phase_idx, progress, on_progress, scan_start)

        ohlcv_data: dict[str, TickerData] = {}
        ohlcv_frames: dict[str, pd.DataFrame] = {}
        phase_start = time.perf_counter()

        try:
            # Get the ticker universe (override, or dynamic from settings).
            if universe_override:
                universe = sorted(universe_override)
            else:
                extra = get_active_watchlist()
                universe = get_scan_universe(
                    presets=self.settings.universe_presets,
                    sectors=self.settings.universe_sectors,
                    extra_tickers=extra,
                    settings=self.settings,
                )
            logger.info("Universe: %d tickers", len(universe))

            # Check cache first.
            cached = load_batch(universe, settings=self.settings)
            cache_hits = set(cached.keys())
            to_fetch = [t for t in universe if t not in cache_hits]

            # Filter out tickers in the failure cache (recently failed).
            failure_cache = load_failure_cache(
                ttl_hours=self.settings.failure_cache_ttl_hours,
                settings=self.settings,
            )
            if failure_cache:
                skipped = [t for t in to_fetch if t in failure_cache]
                if skipped:
                    logger.info(
                        "Skipping %d tickers from failure cache", len(skipped),
                    )
                    for t in skipped:
                        logger.debug(
                            "Skipping %s (failure cache: %s)",
                            t, failure_cache[t].get("error_type", "unknown"),
                        )
                    to_fetch = [t for t in to_fetch if t not in failure_cache]

            logger.info(
                "Cache: %d/%d tickers cached (%.0f%%), %d to fetch",
                len(cache_hits), len(universe),
                len(cache_hits) / len(universe) * 100 if universe else 0,
                len(to_fetch),
            )

            # Fetch missing tickers with configurable params.
            if to_fetch:
                failure_tracker: dict[str, FetchErrorType] = {}
                fetched = fetch_batch(
                    to_fetch,
                    batch_size=self.settings.fetch_batch_size,
                    max_workers=self.settings.fetch_max_workers,
                    max_retries=self.settings.fetch_max_retries,
                    retry_delays=self.settings.fetch_retry_delays,
                    failure_tracker=failure_tracker,
                )
                # Cache the newly fetched data.
                if fetched:
                    save_batch(fetched, settings=self.settings)
                # Record new failures to the failure cache.
                if failure_tracker:
                    new_failures = {
                        symbol: {
                            "error_type": error_type,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "message": f"Fetch failed: {error_type.value}",
                        }
                        for symbol, error_type in failure_tracker.items()
                    }
                    record_failures(new_failures, settings=self.settings)
                ohlcv_data = {**cached, **fetched}
            else:
                ohlcv_data = cached

            # Convert TickerData -> DataFrames for the scoring engine.
            for symbol, td in ohlcv_data.items():
                try:
                    ohlcv_frames[symbol] = _ticker_data_to_dataframe(td)
                except Exception as e:
                    logger.warning("Failed to convert %s to DataFrame: %s", symbol, e)
                    errors.append(f"data_fetch:{symbol}:{e}")

        except Exception as e:
            logger.error("Data fetch phase failed: %s", e)
            errors.append(f"data_fetch:phase:{e}")

        elapsed = time.perf_counter() - phase_start
        self._phase_timings["data_fetch"] = elapsed
        await self._complete_phase(
            phase_idx, progress, on_progress, scan_start,
            ticker_count=len(ohlcv_data), elapsed=elapsed,
            message=f"Fetched {len(ohlcv_data)} tickers",
        )
        return ohlcv_data, ohlcv_frames

    async def _phase_scoring(
        self,
        ohlcv_frames: dict[str, pd.DataFrame],
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
        errors: list[str],
    ) -> list[TickerScore]:
        """Phase 2: Run composite scoring on all OHLCV data."""
        from option_alpha.scoring.composite import score_universe

        phase_idx = 1
        await self._start_phase(phase_idx, progress, on_progress, scan_start)

        ticker_scores: list[TickerScore] = []
        phase_start = time.perf_counter()

        try:
            if ohlcv_frames:
                ticker_scores = score_universe(ohlcv_frames, settings=self.settings)
                logger.info("Scored %d tickers", len(ticker_scores))
            else:
                logger.warning("No OHLCV data available for scoring")
        except Exception as e:
            logger.error("Scoring phase failed: %s", e)
            errors.append(f"scoring:phase:{e}")

        elapsed = time.perf_counter() - phase_start
        self._phase_timings["scoring"] = elapsed
        await self._complete_phase(
            phase_idx, progress, on_progress, scan_start,
            ticker_count=len(ticker_scores), elapsed=elapsed,
            message=f"Scored {len(ticker_scores)} tickers",
        )
        return ticker_scores

    async def _phase_catalysts(
        self,
        ticker_scores: list[TickerScore],
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
        errors: list[str],
    ) -> list[TickerScore]:
        """Phase 3: Fetch earnings dates and merge catalyst scores."""
        phase_idx = 2
        await self._start_phase(phase_idx, progress, on_progress, scan_start)

        phase_start = time.perf_counter()

        try:
            if ticker_scores:
                symbols = [ts.symbol for ts in ticker_scores]
                earnings_info = batch_earnings_info(symbols)
                ticker_scores = merge_catalyst_scores(
                    ticker_scores, earnings_info, settings=self.settings,
                )
                logger.info(
                    "Catalyst scores merged for %d tickers", len(ticker_scores),
                )
            else:
                logger.warning("No scores available for catalyst merge")
        except Exception as e:
            logger.error("Catalysts phase failed: %s", e)
            errors.append(f"catalysts:phase:{e}")

        elapsed = time.perf_counter() - phase_start
        self._phase_timings["catalysts"] = elapsed
        await self._complete_phase(
            phase_idx, progress, on_progress, scan_start,
            ticker_count=len(ticker_scores), elapsed=elapsed,
            message=f"Catalysts merged for {len(ticker_scores)} tickers",
        )
        return ticker_scores

    async def _phase_options(
        self,
        ticker_scores: list[TickerScore],
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
        errors: list[str],
    ) -> list[OptionsRecommendation]:
        """Phase 4: Fetch option chains + recommend contracts for top N."""
        phase_idx = 3
        await self._start_phase(phase_idx, progress, on_progress, scan_start)

        options_recs: list[OptionsRecommendation] = []
        phase_start = time.perf_counter()

        try:
            top_n = self.settings.top_n_options
            candidates = ticker_scores[:top_n]
            if candidates:
                symbols = [ts.symbol for ts in candidates]
                chains = fetch_chains_for_tickers(symbols, settings=self.settings)
                options_recs = recommend_for_scored_tickers(
                    candidates, chains, settings=self.settings,
                )
                logger.info(
                    "Options: %d recommendations from %d candidates",
                    len(options_recs), len(candidates),
                )
            else:
                logger.warning("No candidates for options analysis")
        except Exception as e:
            logger.error("Options phase failed: %s", e)
            errors.append(f"options:phase:{e}")

        elapsed = time.perf_counter() - phase_start
        self._phase_timings["options"] = elapsed
        await self._complete_phase(
            phase_idx, progress, on_progress, scan_start,
            ticker_count=len(options_recs), elapsed=elapsed,
            message=f"{len(options_recs)} options recommendations",
        )
        return options_recs

    async def _phase_ai_debate(
        self,
        ticker_scores: list[TickerScore],
        options_recs: list[OptionsRecommendation],
        scan_db_id: int | None,
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
        errors: list[str],
    ) -> list[DebateResult]:
        """Phase 6: Run AI multi-agent debates for top N tickers."""
        phase_idx = 5
        await self._start_phase(phase_idx, progress, on_progress, scan_start)

        debate_results: list[DebateResult] = []
        phase_start = time.perf_counter()

        try:
            top_n = self.settings.top_n_ai_debate
            if ticker_scores:
                client = get_client(self.settings)

                # Health check gating
                if not await client.health_check():
                    logger.error("LLM health check failed — skipping AI debate phase")
                    # Phase completes with 0 debates, scan stays PARTIAL
                else:
                    manager = DebateManager(client)

                    # Build options_recs lookup dict for the debate manager.
                    recs_dict: dict[str, OptionsRecommendation] = {
                        rec.symbol: rec for rec in options_recs
                    }

                    debate_results = await manager.run_debates(
                        scores=ticker_scores,
                        options_recs=recs_dict,
                        top_n=top_n,
                        settings=self.settings,
                    )
                    logger.info("AI debates completed: %d", len(debate_results))
            else:
                logger.warning("No scores available for AI debate")
        except Exception as e:
            logger.error("AI debate phase failed: %s", e)
            errors.append(f"ai_debate:phase:{e}")

        # Post-debate: finalize scan run and save theses
        if scan_db_id is not None:
            try:
                conn = initialize_db(self.settings.db_path)
                total_duration = time.perf_counter() - scan_start
                if debate_results:
                    save_ai_theses(conn, scan_db_id, debate_results)
                update_scan_run(
                    conn,
                    scan_db_id,
                    status=ScanStatus.COMPLETED if not errors else ScanStatus.PARTIAL,
                    debates_completed=len(debate_results),
                    duration_seconds=round(total_duration, 2),
                )
                conn.close()
            except Exception as e:
                logger.error("Post-debate finalize failed: %s", e)
                errors.append(f"ai_debate:finalize:{e}")

        elapsed = time.perf_counter() - phase_start
        self._phase_timings["ai_debate"] = elapsed
        await self._complete_phase(
            phase_idx, progress, on_progress, scan_start,
            ticker_count=len(debate_results), elapsed=elapsed,
            message=f"{len(debate_results)} debates completed",
        )
        return debate_results

    async def _phase_persist(
        self,
        run_id: str,
        scan_start: float,
        ticker_scores: list[TickerScore],
        options_recs: list[OptionsRecommendation],
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        errors: list[str],
    ) -> int | None:
        """Phase 5: Checkpoint persist — save scores with PARTIAL status."""
        phase_idx = 4
        await self._start_phase(phase_idx, progress, on_progress, scan_start)

        phase_start = time.perf_counter()
        total_duration = time.perf_counter() - scan_start
        scan_db_id: int | None = None

        try:
            conn = initialize_db(self.settings.db_path)

            scan_run = ScanRun(
                run_id=run_id,
                timestamp=datetime.now(UTC),
                ticker_count=len(ticker_scores),
                duration_seconds=round(total_duration, 2),
                status=ScanStatus.PARTIAL,
                error_message="; ".join(errors[:5]) if errors else None,
                scores_computed=len(ticker_scores),
                debates_completed=0,
                options_analyzed=len(options_recs),
            )

            scan_db_id = save_scan_run(conn, scan_run)

            if ticker_scores:
                save_ticker_scores(conn, scan_db_id, ticker_scores)

            conn.close()
            logger.info("Checkpoint persisted scan run %s (db id=%d)", run_id, scan_db_id)

        except Exception as e:
            logger.error("Persist phase failed: %s", e)
            errors.append(f"persist:phase:{e}")

        elapsed = time.perf_counter() - phase_start
        self._phase_timings["persist"] = elapsed
        await self._complete_phase(
            phase_idx, progress, on_progress, scan_start,
            ticker_count=len(ticker_scores), elapsed=elapsed,
            message="Scan results checkpointed",
        )
        return scan_db_id

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    async def _start_phase(
        self,
        phase_idx: int,
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
    ) -> None:
        """Mark a phase as running and notify callback."""
        phase = progress.phases[phase_idx]
        phase.status = PhaseStatus.RUNNING
        phase.message = f"Starting {phase.phase_name}..."
        progress.current_phase = phase.phase_name
        progress.elapsed_total = time.perf_counter() - scan_start
        # Each phase is ~1/6 of total; calculate overall percentage.
        completed_count = sum(
            1 for p in progress.phases if p.status == PhaseStatus.COMPLETED
        )
        progress.overall_percentage = (completed_count / len(PHASE_NAMES)) * 100

        if on_progress:
            await on_progress(progress)

    async def _complete_phase(
        self,
        phase_idx: int,
        progress: ScanProgress,
        on_progress: Optional[ProgressCallback],
        scan_start: float,
        ticker_count: int = 0,
        elapsed: float = 0.0,
        message: str = "",
    ) -> None:
        """Mark a phase as completed and notify callback."""
        phase = progress.phases[phase_idx]
        phase.status = PhaseStatus.COMPLETED
        phase.percentage = 100.0
        phase.ticker_count = ticker_count
        phase.elapsed_seconds = round(elapsed, 3)
        phase.message = message
        progress.elapsed_total = time.perf_counter() - scan_start

        completed_count = sum(
            1 for p in progress.phases if p.status == PhaseStatus.COMPLETED
        )
        progress.overall_percentage = (completed_count / len(PHASE_NAMES)) * 100

        if on_progress:
            await on_progress(progress)

    @property
    def phase_timings(self) -> dict[str, float]:
        """Return recorded phase timing durations in seconds."""
        return dict(self._phase_timings)
