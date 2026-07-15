"""
app/db/mysql_client.py - Async MySQL client using SQLAlchemy + aiomysql.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float,
    JSON, String, Text, select, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


def _build_url() -> str:
    cfg = get_settings()
    return "mysql+aiomysql://%s:%s@%s:%d/%s" % (
        cfg.mysql_user, cfg.mysql_password, cfg.mysql_host, cfg.mysql_port, cfg.mysql_db,
    )


class Base(DeclarativeBase):
    pass


class ForecastLog(Base):
    __tablename__ = "forecast_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_uuid = Column(String(64), unique=True, index=True, nullable=False)
    ticker = Column(String(16), default="TCS")
    quarters = Column(BigInteger, default=3)
    sources = Column(JSON)
    include_market = Column(Boolean, default=False)
    status = Column(String(32), default="pending")
    result_json = Column(JSON)
    tools_used = Column(JSON)
    processing_time_s = Column(Float)
    error_message = Column(Text)
    llm_provider = Column(String(32), default="openai")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


_engine = None
_async_session = None


def _get_engine():
    global _engine, _async_session
    if _engine is None:
        url = _build_url()
        _engine = create_async_engine(url, pool_pre_ping=True, echo=False)
        _async_session = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine


class MySQLClient:
    """Async MySQL client using SQLAlchemy + aiomysql."""

    async def init(self):
        engine = _get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("MySQLClient: schema ready")

    async def log_request(self, request_uuid: str, payload: dict) -> None:
        async with _async_session() as session:
            row = ForecastLog(
                request_uuid=request_uuid,
                ticker=payload.get("ticker", "TCS"),
                quarters=payload.get("quarters", 3),
                sources=payload.get("sources", []),
                include_market=payload.get("include_market", False),
                status="pending",
            )
            session.add(row)
            await session.commit()

    async def log_result(
        self,
        request_uuid: str,
        result: dict,
        tools_used: Optional[list] = None,
        processing_time_s: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """Update log row by request_uuid (NOT primary key id)."""
        async with _async_session() as session:
            stmt = select(ForecastLog).where(ForecastLog.request_uuid == request_uuid)
            result_row = (await session.execute(stmt)).scalar_one_or_none()
            if result_row is None:
                result_row = ForecastLog(request_uuid=request_uuid)
                session.add(result_row)
            result_row.status = status
            result_row.result_json = result
            result_row.tools_used = tools_used or []
            result_row.processing_time_s = processing_time_s
            result_row.error_message = error_message
            await session.commit()

    async def get_result(self, request_uuid: str) -> Optional[Dict[str, Any]]:
        """Retrieve result by request_uuid (NOT primary key id)."""
        async with _async_session() as session:
            stmt = select(ForecastLog).where(ForecastLog.request_uuid == request_uuid)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return {
                "request_uuid": row.request_uuid,
                "status": row.status,
                "result_json": row.result_json,
                "tools_used": row.tools_used,
                "processing_time_s": row.processing_time_s,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    async def check_health(self) -> bool:
        try:
            engine = _get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("DB health check failed: %s", exc)
            return False


_client: Optional[MySQLClient] = None


def get_mysql_client() -> MySQLClient:
    global _client
    if _client is None:
        _client = MySQLClient()
    return _client
