"""
app/tools/qualitative_analysis_tool.py - RAG-based qualitative analysis of TCS earnings calls.

Fallback: uses invoke_with_fallback() so a 429 on the primary model
automatically retries with gemini-1.5-flash, then gemini-1.5-pro.
"""
import json
import logging

from langchain.tools import tool

from app.rag.vector_store import similarity_search
from app.utils.llm_factory import invoke_with_fallback

logger = logging.getLogger(__name__)

THEME_QUERIES = [
    "revenue growth demand outlook next quarter guidance",
    "employee headcount attrition hiring talent",
    "deal wins TCV large deal signings pipeline",
    "margin pressure costs EBITDA profitability",
    "macro uncertainty risks geopolitical client spending",
]

# {{ and }} are escaped braces for Python str.format()
ANALYSIS_PROMPT = (
    "You are a senior equity analyst specialising in Indian IT services.\n"
    "The excerpts below are from TCS earnings-call concall transcripts.\n\n"
    "Analyse the management commentary and produce a structured qualitative summary.\n\n"
    "Return ONLY valid JSON (no markdown, no prose):\n\n"
    '{{\n'
    '  "management_tone": "positive|cautious|negative",\n'
    '  "key_themes": ["<theme 1>", "<theme 2>"],\n'
    '  "forward_looking_statements": ["<direct quote or paraphrase from management>"],\n'
    '  "deal_wins": ["<deal or TCV mentioned>"],\n'
    '  "risks": ["<risk factor>"],\n'
    '  "opportunities": ["<opportunity mentioned>"],\n'
    '  "attrition_commentary": "<string or null>",\n'
    '  "headcount_commentary": "<string or null>",\n'
    '  "guidance_summary": "<string or null>",\n'
    '  "sentiment_score": 0.0\n'
    '}}\n\n'
    "Rules:\n"
    "- management_tone: positive if revenue/deal commentary is optimistic; cautious if uncertain; negative if headwinds dominate.\n"
    "- sentiment_score: float 0.0 (very negative) to 1.0 (very positive).\n"
    "- Include only statements backed by the excerpts. Do NOT fabricate quotes.\n"
    "- forward_looking_statements must be near-verbatim where possible.\n\n"
    "Concall excerpts:\n---\n{context}\n---"
)


def _multi_retrieve(k_per_query: int = 4) -> list:
    seen = set()
    all_chunks = []
    for query in THEME_QUERIES:
        chunks = similarity_search(query, k=k_per_query, filter_financial=False)
        for c in chunks:
            key = c["text"][:120]
            if key not in seen:
                seen.add(key)
                all_chunks.append(c)
    return all_chunks[:20]


def _format_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append("[%d] %s (%s)\n%s" % (i, c["doc_name"], c["quarter"], c["text"]))
    return "\n\n---\n\n".join(parts)


def _score_sentiment(text: str) -> float:
    positive = ["growth", "strong", "win", "opportunity", "demand", "expand", "momentum", "optimistic"]
    cautious = ["uncertain", "headwind", "pressure", "challenging", "risk", "slow", "decline", "macro"]
    t = text.lower()
    p = sum(t.count(w) for w in positive)
    c = sum(t.count(w) for w in cautious)
    total = p + c
    return round(p / total, 2) if total > 0 else 0.5


@tool
def qualitative_analysis_tool(query: str) -> str:
    """
    Performs RAG-based qualitative analysis of TCS earnings-call transcripts.
    Retrieves management commentary on demand, attrition, guidance, deal wins,
    and risks, then synthesises a structured sentiment summary. Input can be
    any question about TCS management tone or outlook.
    """
    chunks = _multi_retrieve(k_per_query=4)
    if not chunks:
        return json.dumps({"error": "No concall documents found in vector store."})

    context = _format_context(chunks)
    heuristic_score = _score_sentiment(context)
    prompt = ANALYSIS_PROMPT.format(context=context)

    try:
        raw, model_used = invoke_with_fallback(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            max_output_tokens=1200,
        )
        result = json.loads(raw)
        llm_score = float(result.get("sentiment_score", heuristic_score))
        result["sentiment_score"] = round((llm_score + heuristic_score) / 2, 2)
        logger.info(
            "qualitative_analysis_tool: tone=%s, sentiment=%.2f (via %s)",
            result.get("management_tone", "?"),
            result["sentiment_score"],
            model_used,
        )
        return json.dumps(result)
    except Exception as exc:
        logger.error("qualitative_analysis_tool failed: %s", exc)
        return json.dumps({"error": str(exc)})
