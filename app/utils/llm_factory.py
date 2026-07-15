"""
app/utils/llm_factory.py – LLM factory with automatic model fallback.

Fallback chain: gemini-flash-latest → gemini-1.5-flash → gemini-1.5-pro

Rules:
- Falls back only on retryable errors (HTTP 429, 503, quota exhausted,
  model unavailable). Auth errors (400 API_KEY_INVALID, 403) propagate
  immediately — retrying with another model won't help.
- thinking_budget=0 is Gemini 2.5 Flash-specific; it is silently dropped
  for non-2.5 models to avoid a 400 "invalid parameter" error.
- The embedding model (GoogleGenerativeAIEmbeddings) is NOT touched here.
"""
import logging
from typing import Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Ordered fallback chain ────────────────────────────────────────────────────
FALLBACK_CHAIN = [
    "gemini-flash-latest",   # Gemini 2.5 Flash (primary)
    "gemini-1.5-flash",      # Fast, generous free-tier quota
    "gemini-1.5-pro",        # Most capable free-tier fallback
]

# Models that accept thinking_budget (Gemini 2.5 series only)
_THINKING_BUDGET_MODELS = {
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview",
}

# Substrings in the exception message that indicate a retryable error
_RETRYABLE_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit",
    "503",
    "service unavailable",
    "overloaded",
    "model not found",
    "not found",
    "resource_exhausted",
    "resourceexhausted",
)

# Substrings that mean "don't bother retrying with another model"
_FATAL_MARKERS = (
    "api_key_invalid",
    "api key invalid",
    "401",
    "403",
    "permission denied",
    "content_filter",
    "content filter",
    "safety",
)


def _classify_error(exc: Exception) -> str:
    """Return 'retryable', 'fatal', or 'unknown'."""
    msg = str(exc).lower()
    if any(m in msg for m in _FATAL_MARKERS):
        return "fatal"
    if any(m in msg for m in _RETRYABLE_MARKERS):
        return "retryable"
    return "unknown"  # unknown → also retry (fail safe)


def make_llm(model: str, **extra_kwargs) -> ChatGoogleGenerativeAI:
    """
    Create a ChatGoogleGenerativeAI for the given model.
    thinking_budget=0 is added automatically for 2.5-series models only.
    Any extra_kwargs (e.g. max_output_tokens) are forwarded.
    """
    cfg = get_settings()
    kwargs = dict(
        model=model,
        google_api_key=cfg.google_api_key,
        temperature=0,
        **extra_kwargs,
    )
    if model in _THINKING_BUDGET_MODELS:
        kwargs["thinking_budget"] = 0
    return ChatGoogleGenerativeAI(**kwargs)


def invoke_with_fallback(
    prompt: str,
    generation_config: Optional[dict] = None,
    **llm_kwargs,
) -> Tuple[str, str]:
    """
    Invoke the LLM with automatic model fallback.

    Args:
        prompt:            The string prompt to send.
        generation_config: Optional dict forwarded as generation_config
                           (e.g. {"response_mime_type": "application/json"}).
        **llm_kwargs:      Extra kwargs for ChatGoogleGenerativeAI
                           (e.g. max_output_tokens=1500).

    Returns:
        (content: str, model_used: str)

    Raises:
        The last exception if every model in the chain fails.
    """
    cfg = get_settings()

    # Build chain: primary first, then unique fallbacks in order
    primary = cfg.google_model
    chain = [primary] + [m for m in FALLBACK_CHAIN if m != primary]

    last_exc: Optional[Exception] = None

    for idx, model in enumerate(chain):
        try:
            llm = make_llm(model, **llm_kwargs)
            if idx > 0:
                logger.info(
                    "LLM fallback: trying model '%s' (attempt %d/%d)",
                    model, idx + 1, len(chain),
                )
            if generation_config:
                response = llm.invoke(prompt, generation_config=generation_config)
            else:
                response = llm.invoke(prompt)

            if idx > 0:
                logger.info("Fallback model '%s' succeeded.", model)
            return response.content, model

        except Exception as exc:
            last_exc = exc
            kind = _classify_error(exc)

            if kind == "fatal":
                # Auth / billing / safety errors — no point trying other models
                logger.error(
                    "Fatal error from model '%s' (will not retry): %s",
                    model, str(exc)[:200],
                )
                raise

            # Retryable or unknown — try next model
            logger.warning(
                "Model '%s' failed [%s]: %s — trying next model",
                model, kind, str(exc)[:150],
            )

    # All models exhausted
    raise last_exc
