"""FastAPI application factory for Option Alpha web dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Directories relative to this file.
_WEB_DIR = Path(__file__).parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


def create_app(config: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional Settings instance. Defaults to loading from env/file.

    Returns:
        Configured FastAPI application with routes, templates, and static files.
    """
    settings = config or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize database and start background scheduler on startup."""
        from option_alpha.data.universe_refresh import refresh_universe, should_refresh
        from option_alpha.persistence.database import initialize_db

        logger.info("Initializing database at %s", settings.db_path)
        conn = initialize_db(settings.db_path)
        conn.close()

        # --- Background scheduler for weekly universe refresh ---
        async def _scheduled_refresh():
            """Run universe refresh if needed."""
            if not should_refresh(settings):
                logger.info("Scheduled universe refresh skipped — not due yet")
                return
            logger.info("Scheduled universe refresh starting")
            result = await refresh_universe(settings, regenerate=False)
            if result.get("success"):
                logger.info(
                    "Scheduled universe refresh completed: %d tickers",
                    result.get("ticker_count", 0),
                )
            else:
                logger.warning(
                    "Scheduled universe refresh failed: %s",
                    result.get("error", "unknown"),
                )

        scheduler = AsyncIOScheduler()
        trigger = CronTrigger(
            day_of_week=settings.universe_refresh_schedule,
            hour=2,
            minute=0,
            timezone="UTC",
        )
        scheduler.add_job(_scheduled_refresh, trigger, id="universe_refresh")
        scheduler.start()

        next_run = scheduler.get_job("universe_refresh").next_run_time
        logger.info(
            "Universe refresh scheduler active — next run: %s", next_run
        )

        yield

        scheduler.shutdown(wait=False)
        logger.info("Universe refresh scheduler shut down")

    app = FastAPI(
        title="Option Alpha Dashboard",
        description="AI-powered options scanner with multi-agent debate",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store settings in app state for route handlers.
    app.state.settings = settings

    # Configure Jinja2 templates.
    tmpl = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Mount static files.
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Wire up templates to routes module.
    from option_alpha.web import routes, websocket

    routes.templates = tmpl

    # Include routers.
    app.include_router(routes.router)
    app.include_router(websocket.router)

    return app
