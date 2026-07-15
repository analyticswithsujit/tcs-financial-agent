"""
app/tools/market_data_tool.py – Live TCS market data via yfinance.

Bonus tool: fetches current stock price, P/E ratio, 52-week range, and
market cap for TCS.NS (NSE listing). Graceful fallback if the API is
unavailable or the market is closed.
"""

import json
import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

# INR to INR crore conversion constant
_CRORE = 1e7


@tool
def market_data_tool(ticker: str = "TCS.NS") -> str:
    """
    Fetches live market data for TCS from Yahoo Finance (yfinance).
    Returns current price, P/E ratio, 52-week high/low, and market cap in crore.
    Input should be the ticker symbol, e.g. 'TCS.NS' for NSE or 'TCS.BO' for BSE.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        pe = info.get("trailingPE") or info.get("forwardPE")
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        mkt_cap = info.get("marketCap")

        result = {
            "ticker": ticker,
            "current_price": round(price, 2) if price else None,
            "currency": info.get("currency", "INR"),
            "pe_ratio": round(pe, 2) if pe else None,
            "week_52_high": round(high_52, 2) if high_52 else None,
            "week_52_low": round(low_52, 2) if low_52 else None,
            "market_cap_cr": round(mkt_cap / _CRORE, 0) if mkt_cap else None,
            "exchange": info.get("exchange", "NSE"),
        }

        logger.info("market_data_tool: fetched data for %s (price=%s)", ticker, result["current_price"])
        return json.dumps(result)

    except ImportError:
        return json.dumps({"error": "yfinance not installed. Run: pip install yfinance"})
    except Exception as exc:
        logger.warning("market_data_tool failed for %s: %s", ticker, exc)
        return json.dumps({"error": str(exc), "ticker": ticker})
