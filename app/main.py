"""
app/main.py – FastAPI application entry point.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.db.mysql_client import get_mysql_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="TCS Financial Forecasting Agent",
        description=(
            "LangChain-powered agent that analyses TCS quarterly financial reports "
            "and earnings-call transcripts (sourced from screener.in) to produce "
            "structured JSON forecasts."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        logger.info("Starting TCS Financial Forecasting Agent...")
        try:
            db = get_mysql_client()
            await db.init()
            logger.info("Database schema ready.")
        except Exception as exc:
            logger.warning("DB init failed (non-fatal): %s", exc)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
