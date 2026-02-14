"""Universe management routes -- ticker/tag CRUD, bulk operations, search."""

from __future__ import annotations

import logging
import math
from typing import Optional

from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from option_alpha.data import universe_service
from option_alpha.persistence.database import initialize_db

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set by app factory.
templates: Optional[Jinja2Templates] = None


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class AddTickersRequest(BaseModel):
    symbols: list[str]
    tags: list[str] = []
    source: str = "manual"


class BulkActionRequest(BaseModel):
    symbols: list[str]
    action: str  # "activate" | "deactivate" | "tag" | "untag" | "remove"
    tag_slug: str | None = None


class CreateTagRequest(BaseModel):
    name: str


class PatchTickerRequest(BaseModel):
    is_active: bool | None = None
    tags: list[str] | None = None


class PatchTagRequest(BaseModel):
    is_active: bool | None = None
    name: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn(request: Request):
    """Get a database connection using app settings."""
    return initialize_db(request.app.state.settings.db_path)


# ---------------------------------------------------------------------------
# Universe dashboard page
# ---------------------------------------------------------------------------


@router.get("/universe", response_class=HTMLResponse)
async def universe_page(request: Request):
    """Render the universe management dashboard page."""
    # Template will be created in Issue #89; return placeholder for now.
    return HTMLResponse(
        content="<html><body><h1>Universe Management</h1>"
        "<p>Full UI coming in Issue #89.</p></body></html>"
    )


# ---------------------------------------------------------------------------
# Ticker endpoints
# ---------------------------------------------------------------------------


@router.get("/api/universe/tickers")
async def list_tickers(
    request: Request,
    tag: str | None = Query(default=None),
    active: int | None = Query(default=None),
    q: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
    sort: str = Query(default="symbol"),
    order: str = Query(default="asc"),
):
    """List tickers with optional filters, pagination, and sorting."""
    conn = _get_conn(request)
    try:
        # Build dynamic WHERE clause.
        conditions: list[str] = []
        params: list[str | int] = []

        if active is not None:
            conditions.append("ut.is_active = ?")
            params.append(active)

        if q:
            conditions.append("(ut.symbol LIKE ? OR ut.name LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])

        if tag:
            conditions.append(
                "ut.symbol IN ("
                "  SELECT tt.symbol FROM ticker_tags tt"
                "  JOIN universe_tags tg ON tt.tag_id = tg.id"
                "  WHERE tg.slug = ?"
                ")"
            )
            params.append(tag)

        where = " AND ".join(conditions) if conditions else "1=1"

        # Validate sort column.
        valid_sorts = {
            "symbol", "name", "source", "is_active", "created_at", "last_scanned_at",
        }
        if sort not in valid_sorts:
            sort = "symbol"
        sort_dir = "DESC" if order.lower() == "desc" else "ASC"

        # Count total.
        count_sql = f"SELECT COUNT(*) FROM universe_tickers ut WHERE {where}"
        total = conn.execute(count_sql, params).fetchone()[0]
        pages = max(1, math.ceil(total / per_page))

        # Fetch page.
        offset = (page - 1) * per_page
        query_sql = (
            f"SELECT ut.symbol, ut.name, ut.source, ut.is_active,"
            f"       ut.created_at, ut.last_scanned_at "
            f"FROM universe_tickers ut "
            f"WHERE {where} "
            f"ORDER BY ut.{sort} {sort_dir} "
            f"LIMIT ? OFFSET ?"
        )
        rows = conn.execute(query_sql, [*params, per_page, offset]).fetchall()

        # Batch-fetch tags for the returned tickers.
        tickers: list[dict] = []
        for row in rows:
            symbol = row["symbol"]
            tag_rows = conn.execute(
                "SELECT tg.slug FROM ticker_tags tt "
                "JOIN universe_tags tg ON tt.tag_id = tg.id "
                "WHERE tt.symbol = ? ORDER BY tg.slug",
                (symbol,),
            ).fetchall()
            tickers.append(
                {
                    "symbol": symbol,
                    "name": row["name"],
                    "source": row["source"],
                    "is_active": row["is_active"],
                    "created_at": row["created_at"],
                    "last_scanned_at": row["last_scanned_at"],
                    "tags": [tr["slug"] for tr in tag_rows],
                }
            )
    finally:
        conn.close()

    return JSONResponse(
        content={"tickers": tickers, "total": total, "page": page, "pages": pages}
    )


@router.post("/api/universe/tickers")
async def add_tickers(request: Request, body: AddTickersRequest):
    """Add tickers to the universe."""
    conn = _get_conn(request)
    try:
        added = universe_service.add_tickers(
            conn, body.symbols, tags=body.tags or None, source=body.source
        )
    finally:
        conn.close()
    return JSONResponse(content={"added": added, "total": len(body.symbols)})


@router.patch("/api/universe/tickers/{symbol}")
async def patch_ticker(request: Request, symbol: str, body: PatchTickerRequest):
    """Toggle active status or update tags for a ticker."""
    conn = _get_conn(request)
    try:
        # Check ticker exists.
        row = conn.execute(
            "SELECT symbol, is_active FROM universe_tickers WHERE symbol = ?",
            (symbol.upper(),),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Ticker not found: {symbol}")

        new_active = row["is_active"]

        if body.is_active is not None:
            try:
                universe_service.toggle_ticker(conn, symbol, body.is_active)
                new_active = 1 if body.is_active else 0
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc))

        if body.tags is not None:
            # Replace all tag associations: remove existing, add new.
            existing_tags = conn.execute(
                "SELECT tg.slug FROM ticker_tags tt "
                "JOIN universe_tags tg ON tt.tag_id = tg.id "
                "WHERE tt.symbol = ?",
                (symbol.upper(),),
            ).fetchall()
            existing_slugs = {r["slug"] for r in existing_tags}
            new_slugs = set(body.tags)

            # Remove tags no longer desired.
            to_remove = existing_slugs - new_slugs
            for slug in to_remove:
                universe_service.untag_tickers(conn, [symbol], slug)

            # Add new tags.
            to_add = new_slugs - existing_slugs
            for slug in to_add:
                universe_service.tag_tickers(conn, [symbol], slug)

    finally:
        conn.close()

    return JSONResponse(content={"symbol": symbol.upper(), "is_active": new_active})


@router.delete("/api/universe/tickers/{symbol}")
async def delete_ticker(request: Request, symbol: str):
    """Remove a ticker from the universe."""
    conn = _get_conn(request)
    try:
        removed = universe_service.remove_tickers(conn, [symbol])
        if removed == 0:
            raise HTTPException(status_code=404, detail=f"Ticker not found: {symbol}")
    finally:
        conn.close()
    return JSONResponse(content={"removed": symbol.upper()})


# ---------------------------------------------------------------------------
# Tag endpoints
# ---------------------------------------------------------------------------


@router.get("/api/universe/tags")
async def list_tags(request: Request):
    """List all tags with ticker counts."""
    conn = _get_conn(request)
    try:
        tags = universe_service.get_all_tags(conn)
    finally:
        conn.close()
    return JSONResponse(content=tags)


@router.post("/api/universe/tags")
async def create_tag(request: Request, body: CreateTagRequest):
    """Create a new tag."""
    conn = _get_conn(request)
    try:
        try:
            tag = universe_service.create_tag(conn, body.name)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    finally:
        conn.close()
    return JSONResponse(content=tag)


@router.patch("/api/universe/tags/{slug}")
async def patch_tag(request: Request, slug: str, body: PatchTagRequest):
    """Toggle active status or rename a tag."""
    conn = _get_conn(request)
    try:
        # Check tag exists.
        row = conn.execute(
            "SELECT * FROM universe_tags WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Tag not found: {slug}")

        if body.is_active is not None:
            try:
                universe_service.toggle_tag(conn, slug, body.is_active)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc))

        if body.name is not None:
            conn.execute(
                "UPDATE universe_tags SET name = ? WHERE slug = ?",
                (body.name, slug),
            )
            conn.commit()

        # Re-fetch tag for response.
        updated = conn.execute(
            "SELECT id, name, slug, is_preset, is_active FROM universe_tags WHERE slug = ?",
            (slug,),
        ).fetchone()
        result = {
            "id": updated["id"],
            "name": updated["name"],
            "slug": updated["slug"],
            "is_preset": updated["is_preset"],
            "is_active": updated["is_active"],
        }
    finally:
        conn.close()
    return JSONResponse(content=result)


@router.delete("/api/universe/tags/{slug}")
async def delete_tag(request: Request, slug: str):
    """Delete a tag."""
    conn = _get_conn(request)
    try:
        try:
            universe_service.delete_tag(conn, slug)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Tag not found: {slug}")
    finally:
        conn.close()
    return JSONResponse(content={"deleted": slug})


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------


@router.post("/api/universe/tickers/bulk")
async def bulk_action(request: Request, body: BulkActionRequest):
    """Execute a bulk action on multiple tickers."""
    conn = _get_conn(request)
    try:
        affected = 0

        if body.action == "activate":
            for sym in body.symbols:
                try:
                    universe_service.toggle_ticker(conn, sym, True)
                    affected += 1
                except ValueError:
                    pass

        elif body.action == "deactivate":
            for sym in body.symbols:
                try:
                    universe_service.toggle_ticker(conn, sym, False)
                    affected += 1
                except ValueError:
                    pass

        elif body.action == "tag":
            if not body.tag_slug:
                raise HTTPException(
                    status_code=400, detail="tag_slug required for 'tag' action"
                )
            affected = universe_service.tag_tickers(conn, body.symbols, body.tag_slug)

        elif body.action == "untag":
            if not body.tag_slug:
                raise HTTPException(
                    status_code=400, detail="tag_slug required for 'untag' action"
                )
            affected = universe_service.untag_tickers(
                conn, body.symbols, body.tag_slug
            )

        elif body.action == "remove":
            affected = universe_service.remove_tickers(conn, body.symbols)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action: {body.action}. "
                f"Valid: activate, deactivate, tag, untag, remove",
            )
    finally:
        conn.close()

    return JSONResponse(content={"action": body.action, "affected": affected})


# ---------------------------------------------------------------------------
# Search / typeahead
# ---------------------------------------------------------------------------


@router.get("/api/universe/search")
async def search_tickers(
    request: Request,
    q: str = Query(default=""),
):
    """Typeahead search for tickers by symbol or name."""
    if not q.strip():
        return JSONResponse(content=[])

    conn = _get_conn(request)
    try:
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT symbol, name, is_active FROM universe_tickers "
            "WHERE symbol LIKE ? OR name LIKE ? "
            "ORDER BY symbol LIMIT 20",
            (like, like),
        ).fetchall()
        results = [
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "is_active": row["is_active"],
            }
            for row in rows
        ]
    finally:
        conn.close()
    return JSONResponse(content=results)
