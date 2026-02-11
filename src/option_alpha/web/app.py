"""FastAPI application factory for Option Alpha web dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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
        """Initialize database on startup."""
        from option_alpha.persistence.database import initialize_db

        logger.info("Initializing database at %s", settings.db_path)
        conn = initialize_db(settings.db_path)
        conn.close()
        yield

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
