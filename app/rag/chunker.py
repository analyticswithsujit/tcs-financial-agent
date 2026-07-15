"""
app/rag/chunker.py – Extract text from PDF bytes and split into overlapping chunks.

Extraction strategy (pdfplumber only – no camelot dependency):
  1. pdfplumber  – structured page text + inline tables
  2. pytesseract – OCR fallback for scanned/image-based pages (if pdfplumber yields nothing)

Each chunk is annotated with metadata so ChromaDB can filter by source type.
"""

import hashlib
import io
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800          # characters
OVERLAP = 150             # overlap between adjacent chunks
MIN_CHUNK_LEN = 80        # discard very short fragments


# ── Financial-section detector ─────────────────────────────────────────────────

_FINANCIAL_KEYWORDS = re.compile(
    r"\b(revenue|profit|margin|EBITDA|EPS|earnings|crore|lakh|FY\d{2}|Q[1-4]\s?FY"
    r"|\d[\d,]+\.?\d*\s?(cr|lakh|mn|bn)?|growth|decline|quarter|annual)\b",
    re.IGNORECASE,
)


def _is_financial(text: str) -> bool:
    return bool(_FINANCIAL_KEYWORDS.search(text))


# ── PDF text extraction ────────────────────────────────────────────────────────

def _extract_text_pdfplumber(pdf_bytes: bytes) -> str:
    """Primary: use pdfplumber for structured text + table extraction."""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                # Tables first (converts to pipe-delimited rows for readability)
                table_text = ""
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        if row:
                            table_text += " | ".join(str(c or "").strip() for c in row) + "\n"

                # Plain page text
                raw = page.extract_text() or ""
                pages.append(f"{raw}\n{table_text}".strip())

        return "\n\n".join(p for p in pages if p)
    except Exception as exc:
        logger.warning("pdfplumber extraction failed: %s", exc)
        return ""


def _extract_text_ocr(pdf_bytes: bytes) -> str:
    """Fallback: OCR for scanned PDFs using pytesseract + pdf2image."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(pdf_bytes, dpi=200)
        texts = [pytesseract.image_to_string(img, lang="eng") for img in images]
        return "\n\n".join(texts)
    except Exception as exc:
        logger.warning("OCR fallback failed: %s", exc)
        return ""


def _extract_text(pdf_bytes: bytes) -> str:
    text = _extract_text_pdfplumber(pdf_bytes)
    if len(text.strip()) < 200:
        logger.info("pdfplumber yielded little text – trying OCR fallback")
        text = _extract_text_ocr(pdf_bytes)
    return text


# ── Chunker ───────────────────────────────────────────────────────────────────

def _split_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split on paragraph boundaries when possible, then by character limit."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Para itself may be too large – hard split it
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_chunk(documents: List[dict]) -> List[dict]:
    """
    Given a list of document dicts (from document_loader.load_documents),
    extract text, split into chunks, and return a list of chunk dicts:

    {
        "text":        str,
        "doc_name":    str,
        "quarter":     str,
        "source":      str,
        "is_financial": bool,
        "chunk_id":    str,   # deterministic hash for deduplication
    }
    """
    all_chunks = []

    for doc in documents:
        name = doc["name"]
        pdf_bytes = doc.get("content", b"")
        if not pdf_bytes:
            logger.warning("No content for %s – skipping", name)
            continue

        logger.info("Extracting text from %s ...", name)
        text = _extract_text(pdf_bytes)

        if not text.strip():
            logger.warning("Zero text extracted from %s", name)
            continue

        raw_chunks = _split_chunks(text)
        logger.info("  → %d chunks from %s", len(raw_chunks), name)

        for chunk in raw_chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            all_chunks.append(
                {
                    "text": chunk,
                    "doc_name": name,
                    "quarter": doc.get("quarter", ""),
                    "source": doc.get("source", ""),
                    "is_financial": _is_financial(chunk),
                    "chunk_id": chunk_id,
                }
            )

    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks
