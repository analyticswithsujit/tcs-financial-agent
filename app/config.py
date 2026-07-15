"""
app/config.py – Centralised settings loaded once via lru_cache.

All configuration is driven by environment variables (see .env.example).
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Google Gemini ─────────────────────────────────────────────────────────
    google_api_key: str = ""
    google_model: str = "gemini-flash-latest"

    # ── MySQL ────────────────────────────────────────────────────────────────
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "tcs_forecast"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "tcs_documents"

    # ── Document cache ────────────────────────────────────────────────────────
    doc_cache_dir: str = "./doc_cache"

    # ── Alpha Vantage (optional – market data) ────────────────────────────────
    alpha_vantage_key: str = ""

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
