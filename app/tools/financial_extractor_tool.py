"""
app/tools/financial_extractor_tool.py - LangChain tool for extracting TCS financial metrics.

BUG FIX: EXTRACTION_PROMPT schema example uses {{ / }} (doubled braces) so that
Python str.format(context=...) does not treat them as format placeholders,
which would raise a KeyError at runtime.

Fallback: uses invoke_with_fallback() so a 429 on gemini-flash-latest
automatically retries with gemini-1.5-flash, then gemini-1.5-pro.
"""
import json
import logging
import time

from langchain.tools import tool

from app.rag.vector_store import similarity_search
from app.utils.llm_factory import invoke_with_fallback

logger = logging.getLogger(__name__)


def _retrieve_financial_chunks(query: str, k: int = 8) -> list:
    return similarity_search(query, k=k, filter_financial=True)


def _format_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append("[%d] %s (%s)\n%s" % (i, c["doc_name"], c["quarter"], c["text"]))
    return "\n\n---\n\n".join(parts)


EXTRACTION_PROMPT = (
    "You are a financial data extraction specialist for Indian IT companies.\n"
    "Below are excerpts from TCS quarterly financial reports and fact sheets.\n\n"
    "Extract the following metrics (for each quarter mentioned):\n"
    "- Total Revenue (in INR crore)\n"
    "- Net Profit (in INR crore)\n"
    "- Operating Margin (%)\n"
    "- EBITDA Margin (%)\n"
    "- EPS (Basic, in INR)\n"
    "- Revenue Growth YoY (%)\n"
    "- Revenue Growth QoQ (%)\n\n"
    "Return ONLY a valid JSON object. No markdown, no prose.\n\n"
    'Schema:\n{{\n  "quarters": [\n    {{\n      "period": "Q4 FY26",\n'
    '      "revenue_crore": "63,973",\n      "net_profit_crore": "12,224",\n'
    '      "operating_margin": "24.5%",\n      "ebitda_margin": "26.1%",\n'
    '      "eps": "33.42",\n      "revenue_growth_yoy": "5.3%",\n'
    '      "revenue_growth_qoq": "1.2%"\n    }}\n  ],\n'
    '  "data_quality": "high|medium|low",\n'
    '  "source_documents": ["TCS_Q4FY26_QuarterlyResult.pdf"]\n}}\n\n'
    "Use null for any metric not found. Do NOT fabricate numbers.\n\n"
    "Context from TCS financial documents:\n---\n{context}\n---"
)


@tool
def financial_data_extractor(query: str) -> str:
    """
    Extracts TCS financial metrics (revenue, profit, margins, EPS) from quarterly
    financial reports stored in ChromaDB. Input should be a natural-language
    description of what financial data you need.
    """
    chunks = _retrieve_financial_chunks(query, k=8)
    if not chunks:
        return json.dumps({"error": "No financial documents found in vector store."})

    context = _format_context(chunks)
    prompt = EXTRACTION_PROMPT.format(context=context)

    last_exc = None
    for attempt in range(1, 4):
        try:
            # invoke_with_fallback handles model-level fallback automatically;
            # this outer loop handles transient JSON parse / network errors.
            raw, model_used = invoke_with_fallback(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                max_output_tokens=1500,
            )
            result = json.loads(raw)
            logger.info(
                "financial_data_extractor: extracted %d quarter(s) via %s",
                len(result.get("quarters", [])), model_used,
            )
            return json.dumps(result)
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2 ** attempt)

    return json.dumps({"error": f"Failed to extract financial metrics after 3 attempts: {last_exc}"})
