"""
app/api/endpoints.py – FastAPI route definitions.

Endpoints (aligned with task requirements):
  POST /forecast/tcs          – run the ForecastAgent
  GET  /status/{request_id}   – retrieve a stored result by UUID
  GET  /health/capabilities   – runtime diagnostics
"""

import asyncio
import importlib.util
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from app.agents.forecast_agent import ForecastAgent
from app.config import get_settings
from app.db.mysql_client import get_mysql_client
from app.rag.vector_store import is_populated
from app.schemas.forecast import ForecastRequest, ForecastResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-initialised global instances (same pattern as reference implementation)
_agent: Optional[ForecastAgent] = None
_db = None


def _ensure_services():
    """Initialise ForecastAgent and MySQLClient once on first request."""
    global _agent, _db
    if _agent is None:
        try:
            _agent = ForecastAgent()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ForecastAgent init failed: {exc}")
    if _db is None:
        _db = get_mysql_client()


# ── POST /forecast/tcs ─────────────────────────────────────────────────────────

@router.post("/forecast/tcs", tags=["Forecast"])
async def forecast_tcs(request: Request, req: ForecastRequest):
    """
    Run the TCS ForecastAgent.

    - Logs the incoming request to MySQL.
    - Downloads + indexes TCS financial PDFs into ChromaDB (first run only).
    - Runs financial_data_extractor and qualitative_analysis_tool, then synthesises the result.
    - Logs the result to MySQL.
    - Returns structured JSON forecast.
    """
    _ensure_services()

    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    payload = req.dict()

    # Log incoming request
    try:
        await _db.log_request(request_id, payload)
    except Exception as exc:
        logger.warning("Failed to log request %s: %s", request_id, exc)

    # Run agent in thread (non-blocking)
    try:
        logger.info("Running ForecastAgent for %s (%s)", req.ticker, request_id)

        result = await asyncio.to_thread(
            _agent.run,
            ticker=req.ticker,
            request_id=request_id,
            quarters=req.quarters,
            sources=req.sources,
            include_market=req.include_market,
        )

        # Log result
        try:
            await _db.log_result(
                request_uuid=request_id,
                result=result.get("result_json"),
                tools_used=result.get("tools_used", []),
                status=result.get("status", "success"),
                error_message=result.get("error"),
            )
        except Exception as exc:
            logger.warning("Failed to log result for %s: %s", request_id, exc)

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Agent error"))

        return ForecastResponse(
            status="success",
            request_id=request_id,
            result_json=result.get("result_json"),
            tools_used=result.get("tools_used", []),
        )

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"ForecastAgent timed out for {req.ticker}")
    except Exception as exc:
        logger.exception("ForecastAgent error for %s", request_id)
        try:
            await _db.log_result(
                request_uuid=request_id,
                result=None,
                status="error",
                error_message=str(exc),
            )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /status/{request_id} ───────────────────────────────────────────────────

@router.get("/status/{request_id}", tags=["Forecast"])
async def get_status(request_id: str):
    """Retrieve the stored forecast result for a given request UUID."""
    _ensure_services()
    try:
        result = await _db.get_result(request_id)
        if result:
            return result
        raise HTTPException(status_code=404, detail="Request ID not found.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch result for %s", request_id)
        raise HTTPException(status_code=500, detail="DB fetch failed.")


# ── GET /health/capabilities ───────────────────────────────────────────────────

@router.get("/health/capabilities", tags=["Ops"])
async def health_capabilities():
    """Runtime capability diagnostics – mirrors reference implementation."""

    def _has(pkg: str) -> bool:
        try:
            return importlib.util.find_spec(pkg) is not None
        except Exception:
            return False

    db_ok = False
    if _db:
        db_ok = await _db.check_health()

    cfg = get_settings()

    return {
        "status": "ok",
        "llm": {
            "provider": "google",
            "model": cfg.google_model,
            "api_key_set": bool(cfg.google_api_key),
        },
        "vector_store": {
            "backend": "chromadb",
            "populated": is_populated(),
        },
        "db": {
            "backend": "mysql+aiomysql (async SQLAlchemy)",
            "host": cfg.mysql_host,
            "connected": db_ok,
        },
        "pdf_tools": {
            "pdfplumber": _has("pdfplumber"),
            "pytesseract": _has("pytesseract"),
            "pdf2image": _has("pdf2image"),
        },
    }
