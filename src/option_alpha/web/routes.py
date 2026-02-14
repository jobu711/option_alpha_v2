"""Route handlers for the Option Alpha web dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from option_alpha.ai.clients import get_client
from option_alpha.ai.debate import DebateManager
from option_alpha.config import DEFAULT_SCORING_WEIGHTS, Settings, get_settings
from option_alpha.models import DebateResult, TickerScore
from option_alpha.data.universe_service import get_all_tags, get_tickers_by_tag
from option_alpha.persistence.database import initialize_db
from option_alpha.persistence.repository import (
    get_all_scans,
    get_latest_scan,
    get_scores_for_scan,
    get_ticker_history,
    save_ai_theses,
)
from option_alpha.web.errors import format_scan_error, run_health_checks
from option_alpha.web.websocket import broadcast_progress

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set by app factory.
templates: Optional[Jinja2Templates] = None

# Track whether a scan or debate is currently running.
_scan_running = False
_debate_running = False
_last_scan_error: Optional[str] = None


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


async def _run_scan_task(
    settings: Settings,
    ticker_subset: list[str] | None = None,
) -> None:
    """Background task to run a scan with progress broadcasting.

    Args:
        settings: Application settings.
        ticker_subset: Optional list of symbols to scan instead of full universe.
    """
    global _scan_running, _last_scan_error

    from option_alpha.pipeline.orchestrator import ScanOrchestrator

    _scan_running = True
    _last_scan_error = None
    try:
        orchestrator = ScanOrchestrator(settings=settings)
        await orchestrator.run_scan(
            ticker_subset=ticker_subset,
            on_progress=broadcast_progress,
        )
    except Exception as e:
        logger.error("Scan failed: %s", e)
        _last_scan_error = format_scan_error(e)
    finally:
        _scan_running = False


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page with latest scan results and market regime."""
    settings: Settings = request.app.state.settings

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

        # Ticker counts for dashboard display.
        active_ticker_count = conn.execute(
            "SELECT COUNT(*) FROM universe_tickers WHERE is_active = 1"
        ).fetchone()[0]

        # Tickers with existing debate results (for "debated" badge).
        debated_tickers: set[str] = set()
        if latest_scan:
            scan_row = conn.execute(
                "SELECT id FROM scan_runs WHERE run_id = ?",
                (latest_scan.run_id,),
            ).fetchone()
            if scan_row:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM ai_theses WHERE scan_run_id = ?",
                    (scan_row["id"],),
                ).fetchall()
                debated_tickers = {r["ticker"] for r in rows}

        # Tags for scan-by-tag dropdown.
        tags = get_all_tags(conn)
    finally:
        conn.close()

    # Calculate staleness.
    stale = False
    if latest_scan:
        age = datetime.now(UTC) - latest_scan.timestamp
        stale = age > timedelta(hours=24)

    market = _get_market_regime()

    # Run health checks for dashboard warnings/errors.
    system_status = await run_health_checks(settings)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "latest_scan": latest_scan,
            "scores": scores,
            "stale": stale,
            "scan_running": _scan_running,
            "debate_running": _debate_running,
            "market": market,
            "system_status": system_status,
            "last_scan_error": _last_scan_error,
            "active_ticker_count": active_ticker_count,
            "debated_tickers": debated_tickers,
            "tags": tags,
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
    """Trigger a new scan run. Returns HTMX partial showing progress.

    Accepts optional JSON body to scan a subset:
      - ``{symbols: ["AAPL", ...]}`` — scan only these tickers
      - ``{tag: "tag-slug"}`` — resolve tag to tickers server-side
      - No body / empty body — scan all active tickers (default)
    """
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

    # Parse optional subset from JSON body.
    symbols: list[str] | None = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
        symbols = body.get("symbols")
        tag = body.get("tag")
        if tag and not symbols:
            settings_tmp: Settings = request.app.state.settings
            conn = initialize_db(settings_tmp.db_path)
            try:
                symbols = get_tickers_by_tag(conn, tag)
            finally:
                conn.close()
        if symbols:
            symbols = [s.upper().strip() for s in symbols if isinstance(s, str) and s.strip()]
            if not symbols:
                symbols = None

    settings: Settings = request.app.state.settings
    background_tasks.add_task(_run_scan_task, settings, ticker_subset=symbols)

    scope_msg = f"Scanning {len(symbols)} tickers..." if symbols else "Scan started..."

    return templates.TemplateResponse(
        request,
        "_progress.html",
        {
            "message": scope_msg,
            "running": True,
            "phases": [
                {"name": "data_fetch", "status": "pending"},
                {"name": "scoring", "status": "pending"},
                {"name": "catalysts", "status": "pending"},
                {"name": "options", "status": "pending"},
                {"name": "persist", "status": "pending"},
            ],
            "overall_percentage": 0,
        },
    )


@router.post("/debate", response_class=HTMLResponse)
async def run_debate(request: Request):
    """Run on-demand AI debates for selected tickers. Returns HTMX partial."""
    global _debate_running

    if _scan_running:
        return JSONResponse(
            status_code=409,
            content={"detail": "A scan is currently running. Try again after it completes."},
        )

    if _debate_running:
        return JSONResponse(
            status_code=409,
            content={"detail": "A debate is already running."},
        )

    # Parse symbols from JSON body.
    try:
        body = await request.json()
        symbols = body.get("symbols", [])
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body."})

    if not symbols or not isinstance(symbols, list):
        return JSONResponse(status_code=400, content={"detail": "No symbols provided."})

    symbols = [s.upper().strip() for s in symbols if isinstance(s, str) and s.strip()]
    if not symbols:
        return JSONResponse(status_code=400, content={"detail": "No valid symbols provided."})

    # Get latest scan and validate symbols.
    settings: Settings = request.app.state.settings
    conn = _get_db_conn(request)
    try:
        latest_scan = get_latest_scan(conn)
        if not latest_scan:
            return JSONResponse(status_code=400, content={"detail": "No scan data available. Run a scan first."})

        row = conn.execute(
            "SELECT id FROM scan_runs WHERE run_id = ?",
            (latest_scan.run_id,),
        ).fetchone()
        if not row:
            return JSONResponse(status_code=400, content={"detail": "Scan data not found."})

        scan_db_id = row["id"]
        all_scores = get_scores_for_scan(conn, scan_db_id)
        scores_by_symbol = {s.symbol: s for s in all_scores}

        # Validate all requested symbols exist in latest scan.
        invalid = [s for s in symbols if s not in scores_by_symbol]
        if invalid:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Symbols not found in latest scan: {', '.join(invalid)}"},
            )
    finally:
        conn.close()

    # Run debates.
    _debate_running = True
    try:
        client = get_client(settings)
        manager = DebateManager(client)
        results: list[DebateResult] = []

        for symbol in symbols:
            ticker_score = scores_by_symbol[symbol]
            try:
                result = await manager.run_debate(ticker_score)
                results.append(result)
            except Exception as e:
                logger.error("Debate failed for %s: %s", symbol, e)

        # Persist: delete existing then insert fresh.
        if results:
            conn = _get_db_conn(request)
            try:
                placeholders = ",".join("?" for _ in symbols)
                conn.execute(
                    f"DELETE FROM ai_theses WHERE scan_run_id = ? AND ticker IN ({placeholders})",
                    [scan_db_id, *symbols],
                )
                save_ai_theses(conn, scan_db_id, results)
                conn.commit()
            finally:
                conn.close()
    finally:
        _debate_running = False

    # Build template data from results.
    template_results = []
    for r in results:
        template_results.append({
            "symbol": r.symbol,
            "bull_thesis": r.bull.analysis,
            "bear_thesis": r.bear.analysis,
            "risk_synthesis": r.risk.analysis,
            "conviction": r.final_thesis.conviction,
            "direction": r.final_thesis.direction.value,
            "recommendation": r.final_thesis.recommended_action,
        })

    return templates.TemplateResponse(
        request,
        "_debate_results.html",
        {"results": template_results},
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
        debated_tickers: set[str] = set()
        if latest_scan:
            row = conn.execute(
                "SELECT id FROM scan_runs WHERE run_id = ?",
                (latest_scan.run_id,),
            ).fetchone()
            if row:
                scores = get_scores_for_scan(conn, row["id"])
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM ai_theses WHERE scan_run_id = ?",
                    (row["id"],),
                ).fetchall()
                debated_tickers = {r["ticker"] for r in rows}
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
            "latest_scan": latest_scan,
            "debated_tickers": debated_tickers,
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


def _mask_key(key: Optional[str]) -> str:
    """Mask an API key, showing only the last 4 characters."""
    if not key:
        return "Not set"
    if len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page with current configuration values."""
    settings: Settings = request.app.state.settings

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "config": settings,
            "weights": settings.scoring_weights,
            "masked_claude_key": _mask_key(settings.claude_api_key),
            "masked_fred_key": _mask_key(settings.fred_api_key),
        },
    )


@router.post("/settings", response_class=HTMLResponse)
async def save_settings(request: Request):
    """Save settings from form data. Returns HTMX partial with status message."""
    settings: Settings = request.app.state.settings
    form = await request.form()

    errors: list[str] = []

    # Parse scoring weights.
    new_weights: dict[str, float] = {}
    for key in DEFAULT_SCORING_WEIGHTS:
        field_name = f"weight_{key}"
        raw = form.get(field_name)
        if raw is not None:
            try:
                val = float(raw)
                if val < 0:
                    errors.append(f"Weight '{key}' must be non-negative.")
                else:
                    new_weights[key] = val
            except (ValueError, TypeError):
                errors.append(f"Invalid value for weight '{key}'.")

    # Parse numeric fields with validation.
    field_defs: list[tuple[str, str, type, Optional[float], Optional[float]]] = [
        ("min_composite_score", "Min Composite Score", float, 0, 100),
        ("min_price", "Min Price", float, 0, None),
        ("min_avg_volume", "Min Avg Volume", int, 0, None),
        ("dte_min", "DTE Min", int, 1, None),
        ("dte_max", "DTE Max", int, 1, None),
        ("min_open_interest", "Min Open Interest", int, 0, None),
        ("max_bid_ask_spread_pct", "Max Bid-Ask Spread", float, 0, 1),
        ("min_option_volume", "Min Option Volume", int, 0, None),
    ]

    parsed: dict[str, float | int] = {}
    for field_name, label, typ, min_val, max_val in field_defs:
        raw = form.get(field_name)
        if raw is not None and raw != "":
            try:
                val = typ(raw)
                if min_val is not None and val < min_val:
                    errors.append(f"{label} must be >= {min_val}.")
                elif max_val is not None and val > max_val:
                    errors.append(f"{label} must be <= {max_val}.")
                else:
                    parsed[field_name] = val
            except (ValueError, TypeError):
                errors.append(f"Invalid value for {label}.")

    # Validate DTE range.
    dte_min = parsed.get("dte_min", settings.dte_min)
    dte_max = parsed.get("dte_max", settings.dte_max)
    if dte_min > dte_max:
        errors.append("DTE Min must be less than or equal to DTE Max.")

    if errors:
        msg = "<div class='settings-msg settings-msg-error'><strong>Validation errors:</strong><ul>"
        for e in errors:
            msg += f"<li>{e}</li>"
        msg += "</ul></div>"
        return HTMLResponse(content=msg)

    # Apply changes.
    if new_weights:
        settings.scoring_weights = new_weights
    for field_name, val in parsed.items():
        setattr(settings, field_name, val)

    # String fields.
    ai_backend = form.get("ai_backend")
    if ai_backend in ("ollama", "claude"):
        settings.ai_backend = ai_backend

    ollama_model = form.get("ollama_model")
    if ollama_model is not None and ollama_model.strip():
        settings.ollama_model = ollama_model.strip()

    # API keys - only update if non-empty (blank = keep current).
    claude_key = form.get("claude_api_key")
    if claude_key is not None and claude_key.strip():
        settings.claude_api_key = claude_key.strip()

    fred_key = form.get("fred_api_key")
    if fred_key is not None and fred_key.strip():
        settings.fred_api_key = fred_key.strip()

    # Persist to file.
    settings.save()

    # Update app state.
    request.app.state.settings = settings

    return HTMLResponse(
        content="<div class='settings-msg settings-msg-success'>Settings saved successfully.</div>"
    )


@router.post("/settings/reset", response_class=HTMLResponse)
async def reset_settings(request: Request):
    """Reset all settings to defaults, save to file, and redirect to settings."""
    defaults = Settings()
    defaults.save()
    request.app.state.settings = defaults

    # Return redirect via HTMX (HX-Redirect header).
    return HTMLResponse(
        content="",
        headers={"HX-Redirect": "/settings"},
    )


@router.get("/export", response_class=HTMLResponse)
async def export_report(request: Request):
    """Generate and download a self-contained HTML report of latest scan results."""
    conn = _get_db_conn(request)
    try:
        latest_scan = get_latest_scan(conn)
        scores: list[TickerScore] = []
        details: list[dict] = []

        if latest_scan:
            row = conn.execute(
                "SELECT id FROM scan_runs WHERE run_id = ?",
                (latest_scan.run_id,),
            ).fetchone()
            if row:
                scan_db_id = row["id"]
                scores = get_scores_for_scan(conn, scan_db_id)

                # Build detail sections for top 10.
                for s in scores[:10]:
                    detail: dict = {"score": s, "debate": None, "options_rec": None}

                    # Fetch AI debate.
                    debate_row = conn.execute(
                        "SELECT * FROM ai_theses WHERE scan_run_id = ? AND ticker = ?",
                        (scan_db_id, s.symbol),
                    ).fetchone()
                    if debate_row:
                        detail["debate"] = {
                            "bull_thesis": debate_row["bull_thesis"],
                            "bear_thesis": debate_row["bear_thesis"],
                            "risk_synthesis": debate_row["risk_synthesis"],
                            "conviction": debate_row["conviction"],
                            "recommendation": debate_row["recommendation"],
                            "direction": debate_row["direction"],
                        }

                    # Fetch options recommendation.
                    score_row = conn.execute(
                        "SELECT options_recommendation_json FROM ticker_scores "
                        "WHERE scan_run_id = ? AND ticker = ?",
                        (scan_db_id, s.symbol),
                    ).fetchone()
                    if score_row and score_row["options_recommendation_json"]:
                        detail["options_rec"] = json.loads(
                            score_row["options_recommendation_json"]
                        )

                    details.append(detail)
    finally:
        conn.close()

    report_date = datetime.now(UTC).strftime("%Y-%m-%d")
    filename = f"report_{report_date}.html"

    html = templates.get_template("_report.html").render(
        scan=latest_scan,
        scores=scores,
        details=details,
        report_date=report_date,
    )

    return Response(
        content=html,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
