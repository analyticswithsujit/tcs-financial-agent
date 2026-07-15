"""
app/schemas/forecast.py – Pydantic request/response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    ticker: str = Field("TCS", description="Stock ticker (currently only TCS supported)")
    quarters: int = Field(3, ge=1, le=8, description="Number of past quarters to analyse")
    sources: List[str] = Field(
        default=["screener", "company-ir"],
        description="Document sources to include: 'screener', 'company-ir'",
    )
    include_market: bool = Field(False, description="Include live market data in forecast")


# ── Nested response models ─────────────────────────────────────────────────────

class FinancialMetrics(BaseModel):
    revenue_crore: Optional[str] = None
    revenue_growth_yoy: Optional[str] = None
    net_profit_crore: Optional[str] = None
    operating_margin: Optional[str] = None
    ebitda_margin: Optional[str] = None
    eps: Optional[str] = None


class QualitativeOutlook(BaseModel):
    management_tone: Optional[str] = None          # positive | cautious | negative
    key_themes: List[str] = Field(default_factory=list)
    forward_looking_statements: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    current_price: Optional[float] = None
    currency: Optional[str] = None
    pe_ratio: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    market_cap_cr: Optional[float] = None


class ForecastOutput(BaseModel):
    quarter_forecast: Optional[str] = None
    financial_metrics: FinancialMetrics = Field(default_factory=FinancialMetrics)
    qualitative_analysis: QualitativeOutlook = Field(default_factory=QualitativeOutlook)
    market_snapshot: Optional[MarketSnapshot] = None
    overall_outlook: Optional[str] = None
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    source_documents: List[str] = Field(default_factory=list)
    # Allow extra fields from raw agent output
    model_config = {"extra": "allow"}


# ── Top-level response ─────────────────────────────────────────────────────────

class ForecastResponse(BaseModel):
    status: str
    request_id: str
    result_json: Optional[Dict[str, Any]] = None
    tools_used: List[str] = Field(default_factory=list)
    processing_time_s: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    db_connected: bool
    vector_store_ready: bool
    pdf_tools: Dict[str, bool]
    llm_provider: str = "openai"
