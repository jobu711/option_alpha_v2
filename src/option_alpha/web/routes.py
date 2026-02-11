"""Route handlers for the Option Alpha web dashboard."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from option_alpha.config import Settings, get_settings
from option_alpha.models import TickerScore
from option_alpha.persistence.database import initialize_db
from option_alpha.persistence.repository import (
    get_all_scans,
    get_latest_scan,
    get_scores_for_scan,
    get_ticker_history,
)
from option_alpha.web.websocket import broadcast_progress

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set by app factory.
templates: Optional[Jinja2Templates] = None

# Track whether a scan is currently running.
_scan_running = False


def _get_db_conn(request: Request):
    """Get a database connection using config from app state."""
    settings: Settings = request.app.state.settings
    return initialize_db(settings.db_path)


def _get_market_regime() -> dict:
    """Fetch VIX level and SPY trend for market regime header.

    Returns cached data to avoid slow yfinance calls on every page load.
    Uses a simple in-memory cache with 5-minute TTL.
    """
    import time

    cache = getattr(_get_market_regime, "_cache", None)
    now = time.time()

    if cache and (now - cache["time"]) < 300:
        return cache["data"]

    data = {"vix": None, "spy_trend": None, "spy_price": None}

    try:
        import yfinance as yf

        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1d")
        if not vix_hist.empty:
            data["vix"] = round(float(vix_hist["Close"].iloc[-1]), 2)

        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="5d")
        if not spy_hist.empty:
            data["spy_price"] = round(float(spy_hist["Close"].iloc[-1]), 2)
            if len(spy_hist) >= 2:
                prev = float(spy_hist["Close"].iloc[-2])
                curr = float(spy_hist["Close"].iloc[-1])
                data["spy_trend"] = "up" if curr > prev else "down"
            else:
                data["spy_trend"] = "flat"
    except Exception as e:
        logger.warning("Failed to fetch market data: %s", e)

    _get_market_regime._cache = {"time": now, "data": data}
    return data


async def _run_scan_task(settings: Settings) -> None:
    """Background task to run a full scan with progress broadcasting."""
    global _scan_running

    from option_alpha.pipeline.orchestrator import ScanOrchestrator

    _scan_running = True
    try:
        orchestrator = ScanOrchestrator(settings=settings)
        await orchestrator.run_scan(on_progress=broadcast_progress)
    except Exception as e:
        logger.error("Scan failed: %s", e)
    finally:
        _scan_running = False


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page with latest scan results and market regime."""
    conn = _get_db_conn(request)
    try:
        latest_scan = get_latest_scan(conn)
        scores: list[TickerScore] = []
        if latest_scan:
            scores = get_scores_for_scan(conn, latest_scan.run_id)
            # get_scores_for_scan uses run_id but actually needs the DB id
            # Re-query using the scan_runs table id
            row = conn.execute(
                "SELECT id FROM scan_runs WHERE run_id = ?",
                (latest_scan.run_id,),
            ).fetchone()
            if row:
                scores = get_scores_for_scan(conn, row["id"])
    finally:
        conn.close()

    # Calculate staleness.
    stale = False
    if latest_scan:
        age = datetime.now(UTC) - latest_scan.timestamp
        stale = age > timedelta(hours=24)

    market = _get_market_regime()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "latest_scan": latest_scan,
            "scores": scores,
            "stale": stale,
            "scan_running": _scan_running,
            "market": market,
        },
    )


@router.get("/ticker/{symbol}", response_class=HTMLResponse)
async def ticker_detail(request: Request, symbol: str):
    """Ticker detail page with full score breakdown, options, and AI debate."""
    conn = _get_db_conn(request)
    try:
        latest_scan = get_latest_scan(conn)
        score: Optional[TickerScore] = None
        debate = None
        options_rec = None

        if latest_scan:
            row = conn.execute(
                "SELECT id FROM scan_runs WHERE run_id = ?",
                (latest_scan.run_id,),
            ).fetchone()
            if row:
                scan_db_id = row["id"]
                scores = get_scores_for_scan(conn, scan_db_id)
                for s in scores:
                    if s.symbol == symbol:
                        score = s
                        break

                # Fetch AI debate result.
                debate_row = conn.execute(
                    "SELECT * FROM ai_theses WHERE scan_run_id = ? AND ticker = ?",
                    (scan_db_id, symbol),
                ).fetchone()
                if debate_row:
                    debate = {
                        "bull_thesis": debate_row["bull_thesis"],
                        "bear_thesis": debate_row["bear_thesis"],
                        "risk_synthesis": debate_row["risk_synthesis"],
                        "conviction": debate_row["conviction"],
                        "recommendation": debate_row["recommendation"],
                        "direction": debate_row["direction"],
                    }

                # Fetch options recommendation from ticker_scores JSON.
                score_row = conn.execute(
                    "SELECT options_recommendation_json FROM ticker_scores "
                    "WHERE scan_run_id = ? AND ticker = ?",
                    (scan_db_id, symbol),
                ).fetchone()
                if score_row and score_row["options_recommendation_json"]:
                    import json

                    options_rec = json.loads(score_row["options_recommendation_json"])
    finally:
        conn.close()

    return templates.TemplateResponse(
        request,
        "ticker_detail.html",
        {
            "symbol": symbol,
            "score": score,
            "debate": debate,
            "options_rec": options_rec,
        },
    )


@router.post("/scan", response_class=HTMLResponse)
async def trigger_scan(request: Request, background_tasks: BackgroundTasks):
    """Trigger a new scan run. Returns HTMX partial showing progress."""
    global _scan_running

    if _scan_running:
        return templates.TemplateResponse(
            request,
            "_progress.html",
            {
                "message": "A scan is already running...",
                "running": True,
                "phases": [],
                "overall_percentage": 0,
            },
        )

    settings: Settings = request.app.state.settings
    background_tasks.add_task(_run_scan_task, settings)

    return templates.TemplateResponse(
        request,
        "_progress.html",
        {
            "message": "Scan started...",
            "running": True,
            "phases": [
                {"name": "data_fetch", "status": "pending"},
                {"name": "scoring", "status": "pending"},
                {"name": "catalysts", "status": "pending"},
                {"name": "options", "status": "pending"},
                {"name": "ai_debate", "status": "pending"},
                {"name": "persist", "status": "pending"},
            ],
            "overall_percentage": 0,
        },
    )


@router.get("/candidates", response_class=HTMLResponse)
async def candidates_table(
    request: Request,
    sort_by: str = Query(default="composite_score"),
    order: str = Query(default="desc"),
):
    """HTMX partial: ranked candidates table with sorting."""
    conn = _get_db_conn(request)
    try:
        latest_scan = get_latest_scan(conn)
        scores: list[TickerScore] = []
        if latest_scan:
            row = conn.execute(
                "SELECT id FROM scan_runs WHERE run_id = ?",
                (latest_scan.run_id,),
            ).fetchone()
            if row:
                scores = get_scores_for_scan(conn, row["id"])
    finally:
        conn.close()

    # Sort by requested field.
    valid_fields = {"composite_score", "symbol", "direction", "last_price"}
    if sort_by not in valid_fields:
        sort_by = "composite_score"

    reverse = order == "desc"
    if sort_by == "symbol":
        scores.sort(key=lambda s: s.symbol, reverse=reverse)
    elif sort_by == "composite_score":
        scores.sort(key=lambda s: s.composite_score, reverse=reverse)
    elif sort_by == "direction":
        scores.sort(key=lambda s: s.direction.value, reverse=reverse)
    elif sort_by == "last_price":
        scores.sort(key=lambda s: s.last_price or 0, reverse=reverse)

    return templates.TemplateResponse(
        request,
        "_candidates_table.html",
        {
            "scores": scores,
            "sort_by": sort_by,
            "order": order,
        },
    )


@router.get("/history/{symbol}")
async def ticker_history(
    request: Request,
    symbol: str,
    days: int = Query(default=30),
):
    """JSON endpoint: score history data for chart."""
    conn = _get_db_conn(request)
    try:
        history = get_ticker_history(conn, symbol, days=days)
    finally:
        conn.close()

    data = [
        {
            "timestamp": h.timestamp.isoformat(),
            "composite_score": h.composite_score,
            "direction": h.direction.value,
        }
        for h in history
    ]
    return JSONResponse(content={"symbol": symbol, "history": data})
