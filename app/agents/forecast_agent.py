"""
app/agents/forecast_agent.py - ForecastAgent for TCS financial forecasting.

Key design decisions vs friend's project:
  - Google Gemini instead of OpenAI GPT-4o
  - Sequential tool calls (financial_data_extractor -> qualitative_analysis_tool
    -> optional market_data_tool -> synthesis) instead of an LLM-driven
    AgentExecutor tool-calling loop. Gemini's function-calling protocol now
    requires a `thought_signature` on every functionCall part to be echoed
    back on the next turn; LangChain's AgentExecutor does not round-trip that
    field, so multi-turn tool calling fails with a 400 from the API. Since
    the tool sequence here is fixed (not dynamically chosen by the model),
    calling the tools directly sidesteps the issue entirely and also cuts
    the number of LLM round-trips per request.
  - ChromaDB persistent store instead of FAISS in-memory
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.rag.chunker import load_and_chunk
from app.rag.document_loader import load_documents
from app.rag.vector_store import ingest_documents, is_populated
from app.tools.financial_extractor_tool import financial_data_extractor
from app.tools.market_data_tool import market_data_tool
from app.tools.qualitative_analysis_tool import qualitative_analysis_tool

logger = logging.getLogger(__name__)

# Note: {{ and }} produce literal { } in the formatted string sent to the LLM.
SYNTHESIS_PROMPT = (
    "You are an expert financial analyst specialising in Indian IT services companies, "
    "particularly Tata Consultancy Services (TCS).\n\n"
    "Below is data already extracted for the last {quarters} quarter(s) by dedicated "
    "extraction tools. Synthesise it into a single structured JSON forecast for the "
    "upcoming quarter.\n\n"
    "Financial metrics (from financial_data_extractor):\n{financial_data}\n\n"
    "Qualitative analysis (from qualitative_analysis_tool):\n{qualitative_data}\n\n"
    "Market snapshot (from market_data_tool, only present if requested):\n{market_data}\n\n"
    "CRITICAL - your response must be ONLY valid JSON (no markdown, no prose):\n\n"
    '{{\n'
    '  "quarter_forecast": "<upcoming quarter label, e.g. Q1 FY27>",\n'
    '  "financial_metrics": {{\n'
    '    "revenue_crore": "<string or null>",\n'
    '    "revenue_growth_yoy": "<string or null>",\n'
    '    "net_profit_crore": "<string or null>",\n'
    '    "operating_margin": "<string or null>",\n'
    '    "ebitda_margin": "<string or null>",\n'
    '    "eps": "<string or null>"\n'
    '  }},\n'
    '  "qualitative_analysis": {{\n'
    '    "management_tone": "positive|cautious|negative",\n'
    '    "key_themes": ["<theme>"],\n'
    '    "forward_looking_statements": ["<quote>"],\n'
    '    "risks": ["<risk>"],\n'
    '    "opportunities": ["<opportunity>"]\n'
    '  }},\n'
    '  "market_snapshot": null,\n'
    '  "overall_outlook": "<2-3 sentence narrative forecast>",\n'
    '  "confidence_score": 0.0,\n'
    '  "source_documents": ["<filename>"]\n'
    '}}\n\n'
    "Rules:\n"
    "- Never fabricate numbers. Only use the data provided above.\n"
    "- Set confidence_score 0.0-1.0 based on data completeness.\n"
    "- market_snapshot = null unless market data was provided above."
)


def _ingest_if_needed(quarters: int, sources: List[str]) -> None:
    if is_populated():
        logger.info("Vector store already populated - skipping ingestion")
        return
    logger.info("Vector store empty - loading and indexing documents...")
    raw_docs = load_documents(quarters=quarters, sources=sources)
    chunks = load_and_chunk(raw_docs)
    added = ingest_documents(chunks)
    logger.info("Ingested %d chunks into ChromaDB", added)


class ForecastAgent:
    """
    ForecastAgent that runs the financial/qualitative/market tools as a fixed
    sequential pipeline, then makes a single LLM call to synthesise the
    structured JSON forecast from their combined output.
    """

    def __init__(self):
        cfg = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=cfg.google_model,
            google_api_key=cfg.google_api_key,
            temperature=0,
            thinking_budget=0,
        )
        logger.info("ForecastAgent initialised (LLM: %s)", cfg.google_model)

    def _parse_output(self, raw: str) -> dict:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(cleaned)
        except Exception as exc:
            logger.warning("JSON parse failed (%s) - returning raw", exc)
            return {"raw_output": raw, "parse_error": str(exc)}

    def _run_pipeline(
        self,
        ticker: str,
        request_id: str,
        quarters: int,
        sources: List[str],
        include_market: bool,
    ) -> Dict[str, Any]:
        _ingest_if_needed(quarters, sources)

        tools_used = []

        financial_data = financial_data_extractor.invoke(
            f"Extract financial metrics for {ticker} for the last {quarters} quarter(s)"
        )
        tools_used.append("financial_data_extractor")

        qualitative_data = qualitative_analysis_tool.invoke(
            f"Summarise management tone, guidance, deal wins, and risks for {ticker} "
            f"over the last {quarters} quarter(s)"
        )
        tools_used.append("qualitative_analysis_tool")

        market_data = "Not requested"
        if include_market:
            market_data = market_data_tool.invoke(ticker)
            tools_used.append("market_data_tool")

        prompt = SYNTHESIS_PROMPT.format(
            quarters=quarters,
            financial_data=financial_data,
            qualitative_data=qualitative_data,
            market_data=market_data,
        )

        response = self.llm.invoke(prompt)
        raw_output = response.content

        forecast = self._parse_output(raw_output)
        forecast["_metadata"] = {
            "request_id": request_id,
            "ticker": ticker,
            "tools_used": tools_used,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "llm_provider": "google",
            "llm_model": self.llm.model,
        }

        return {
            "status": "success",
            "ticker": ticker,
            "request_id": request_id,
            "result_json": forecast,
            "tools_used": tools_used,
        }

    def run(
        self,
        ticker: str = "TCS",
        request_id: str = "",
        quarters: int = 3,
        sources: List[str] = None,
        include_market: bool = False,
    ) -> Dict[str, Any]:
        if sources is None:
            sources = ["screener", "company-ir"]
        try:
            return self._run_pipeline(ticker, request_id, quarters, sources, include_market)
        except Exception as exc:
            logger.exception("ForecastAgent.run() failed")
            return {"status": "error", "ticker": ticker, "request_id": request_id,
                    "error": str(exc)}
