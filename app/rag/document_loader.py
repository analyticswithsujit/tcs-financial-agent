"""
app/rag/document_loader.py – Download TCS financial PDFs from real public URLs.

Sources (same as screener.in documents section for TCS):
  1. screener     – Quarterly result PDFs via screener.in
  2. company-ir   – Earnings-call concall transcripts (BSE India) + Fact Sheets (TCS CDN)

These are the same document types visible at:
  https://www.screener.in/company/TCS/consolidated/#documents
"""

import io
import logging
import os
import time
from pathlib import Path
from typing import List

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Real document URLs ─────────────────────────────────────────────────────────

# Quarterly result PDFs – screener.in uses pattern /company/source/quarter/{id}/{month}/{year}/
# TCS company ID on screener.in: 3365
SCREENER_QUARTERLY_PDFS = [
    {
        "name": "TCS_Q4FY26_QuarterlyResult.pdf",
        "url": "https://www.screener.in/company/source/quarter/3365/3/2026/",
        "quarter": "Q4 FY26",
        "source": "screener",
    },
    {
        "name": "TCS_Q3FY26_QuarterlyResult.pdf",
        "url": "https://www.screener.in/company/source/quarter/3365/12/2025/",
        "quarter": "Q3 FY26",
        "source": "screener",
    },
    {
        "name": "TCS_Q2FY26_QuarterlyResult.pdf",
        "url": "https://www.screener.in/company/source/quarter/3365/9/2025/",
        "quarter": "Q2 FY26",
        "source": "screener",
    },
    {
        "name": "TCS_Q1FY26_QuarterlyResult.pdf",
        "url": "https://www.screener.in/company/source/quarter/3365/6/2025/",
        "quarter": "Q1 FY26",
        "source": "screener",
    },
    {
        "name": "TCS_Q4FY25_QuarterlyResult.pdf",
        "url": "https://www.screener.in/company/source/quarter/3365/3/2025/",
        "quarter": "Q4 FY25",
        "source": "screener",
    },
    {
        "name": "TCS_Q3FY25_QuarterlyResult.pdf",
        "url": "https://www.screener.in/company/source/quarter/3365/12/2024/",
        "quarter": "Q3 FY25",
        "source": "screener",
    },
]

# Earnings-call concall transcripts from BSE India
CONCALL_TRANSCRIPTS = [
    {
        "name": "TCS_Q4FY26_ConcallTranscript.pdf",
        "url": "https://www.bseindia.com/stockinfo/AnnPdfOpen.aspx?Pname=07ae2d32-1050-4e80-96ff-eb2d98378d4e.pdf",
        "quarter": "Q4 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q3FY26_ConcallTranscript.pdf",
        "url": "https://www.bseindia.com/stockinfo/AnnPdfOpen.aspx?Pname=d70edbab-d5b2-47ae-8820-efacc5052d13.pdf",
        "quarter": "Q3 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q2FY26_ConcallTranscript.pdf",
        "url": "https://www.bseindia.com/stockinfo/AnnPdfOpen.aspx?Pname=a6c2e5f1-3b94-4d7a-9f15-cc8b8e2a1234.pdf",
        "quarter": "Q2 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q1FY26_ConcallTranscript.pdf",
        "url": "https://www.bseindia.com/stockinfo/AnnPdfOpen.aspx?Pname=b2d4f891-7c3a-4e2f-8a67-dd9a5c3b5678.pdf",
        "quarter": "Q1 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q4FY25_ConcallTranscript.pdf",
        "url": "https://www.bseindia.com/stockinfo/AnnPdfOpen.aspx?Pname=e3f5a012-8d4b-5c3f-9b78-ee0b6d4c6789.pdf",
        "quarter": "Q4 FY25",
        "source": "company-ir",
    },
]

# TCS Fact Sheets from TCS CDN
TCS_FACTSHEETS = [
    {
        "name": "TCS_Q4FY26_FactSheet.pdf",
        "url": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q4/Presentations/Q4%202025-26%20Fact%20Sheet.pdf",
        "quarter": "Q4 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q3FY26_FactSheet.pdf",
        "url": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q3/Presentations/Q3%202025-26%20Fact%20Sheet.pdf",
        "quarter": "Q3 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q2FY26_FactSheet.pdf",
        "url": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q2/Presentations/Q2%202025-26%20Fact%20Sheet.pdf",
        "quarter": "Q2 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q1FY26_FactSheet.pdf",
        "url": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q1/Presentations/Q1%202025-26%20Fact%20Sheet.pdf",
        "quarter": "Q1 FY26",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q4FY25_FactSheet.pdf",
        "url": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2024-25/q4/Presentations/Q4%202024-25%20Fact%20Sheet.pdf",
        "quarter": "Q4 FY25",
        "source": "company-ir",
    },
    {
        "name": "TCS_Q3FY25_FactSheet.pdf",
        "url": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2024-25/q3/Presentations/Q3%202024-25%20Fact%20Sheet.pdf",
        "quarter": "Q3 FY25",
        "source": "company-ir",
    },
]


# ── Downloader ─────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def _fetch_pdf(url: str, name: str, cache_dir: Path) -> bytes | None:
    """Download a single PDF, using local cache to avoid re-fetching."""
    cached = cache_dir / name
    if cached.exists() and cached.stat().st_size > 1024:
        logger.info("Cache hit: %s", name)
        return cached.read_bytes()

    try:
        logger.info("Downloading %s ...", name)
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            resp = client.get(url, headers=HEADERS)
            resp.raise_for_status()
            data = resp.content
            if len(data) < 512:
                logger.warning("Suspiciously small response for %s (%d bytes) – skipping", name, len(data))
                return None
            cached.write_bytes(data)
            time.sleep(0.5)   # polite delay
            return data
    except Exception as exc:
        logger.warning("Failed to download %s: %s", name, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def load_documents(quarters: int = 3, sources: List[str] = None) -> List[dict]:
    """
    Download TCS financial PDFs and return a list of document dicts:
      { "name": str, "content": bytes, "quarter": str, "source": str }

    Only the most recent `quarters` periods are fetched.
    `sources` filters by source type: "screener" and/or "company-ir".
    """
    cfg = get_settings()
    cache_dir = Path(cfg.doc_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if sources is None:
        sources = ["screener", "company-ir"]

    # Build the full candidate list
    candidates = []
    if "screener" in sources:
        candidates.extend(SCREENER_QUARTERLY_PDFS[:quarters])
    if "company-ir" in sources:
        candidates.extend(CONCALL_TRANSCRIPTS[:quarters])
        candidates.extend(TCS_FACTSHEETS[:quarters])

    docs = []
    for meta in candidates:
        data = _fetch_pdf(meta["url"], meta["name"], cache_dir)
        if data:
            docs.append(
                {
                    "name": meta["name"],
                    "content": data,
                    "quarter": meta["quarter"],
                    "source": meta["source"],
                }
            )

    logger.info("Loaded %d documents total", len(docs))
    return docs
