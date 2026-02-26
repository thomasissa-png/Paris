#!/usr/bin/env python3
"""
Polymarket Live Opportunity Scanner v6
=======================================
Full trading intelligence: ROI, annualized returns, mispricing direction,
multi-outcome support, category filtering, smart scoring, Kelly sizing,
CLOB spread verification, prediction logging, webhook alerts.

Edge informationnel — 7 analyzers detect markets where external data
disagrees with market prices:
  - Weather:    OpenWeatherMap forecast vs. temperature/rain markets
  - Finance:    Yahoo Finance price + volatility vs. stock/index/forex/commodity markets
  - Earnings:   Yahoo Finance earnings data vs. "beat/miss" markets
  - Sports:     Bookmaker consensus odds vs. sports outcome markets
  - Fed/Macro:  FRED economic data vs. rate/CPI/unemployment/GDP markets
  - Elections:  RealClearPolitics polls vs. election/approval markets
  - Crypto:     CoinGecko prices vs. crypto price target markets

Run:  pip install -r requirements.txt && python polymarket_scanner.py
Open: http://localhost:5000

Optional env vars:
  OPENWEATHERMAP_API_KEY  — free key from openweathermap.org (1000 calls/day)
  THE_ODDS_API_KEY        — free key from the-odds-api.com (500 requests/month)
  FRED_API_KEY            — free key from fred.stlouisfed.org (unlimited)
  DISCORD_WEBHOOK_URL     — Discord channel webhook for alerts
  TELEGRAM_BOT_TOKEN      — Telegram bot token for alerts
  TELEGRAM_CHAT_ID        — Telegram chat ID for alerts
  PREDICTIONS_LOG         — path to prediction log (default: predictions.jsonl)
  PORTFOLIO_FILE          — path to portfolio file (default: portfolio.json)
"""

import json
import logging
import math
import os
import re
import threading
import time as _time
from datetime import datetime, timezone, timedelta

import requests
from flask import Flask, jsonify, Response

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GAMMA_API = "https://gamma-api.polymarket.com/markets"
API_PARAMS = {
    "active": "true",
    "closed": "false",
    "order": "volume",
    "ascending": "false",
}
BATCH_SIZE = 100
MAX_PAGES = 5           # up to 500 markets
MAX_DAYS = 60
MIN_VOLUME = 1000
MIN_LIQUIDITY = 5000
CACHE_TTL = 25          # seconds
MAX_ANN_ROI = 1000.0    # cap annualized ROI to avoid absurd display
EST_FEE_PCT = 2.0       # estimated round-trip trading fees (%)
MIN_EDGE = 0.05         # minimum edge (5%) to qualify as "edge" tier
PORT = 5000

# --- External analysis (optional API keys) ---
OWM_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OWM_CACHE_TTL = 600     # 10 min cache for weather forecasts
ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
FINANCE_CACHE_TTL = 300  # 5 min cache for financial data
ODDS_CACHE_TTL = 300     # 5 min cache for sports odds

# --- CLOB API (real spread verification) ---
CLOB_API_BASE = "https://clob.polymarket.com"
CLOB_CACHE_TTL = 30      # 30s cache for order books

# --- Notifications ---
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# --- Prediction logging & portfolio ---
PREDICTIONS_LOG_FILE = os.environ.get("PREDICTIONS_LOG", "predictions.jsonl")
PORTFOLIO_FILE = os.environ.get("PORTFOLIO_FILE", "portfolio.json")

# --- CoinGecko rate limiting ---
COINGECKO_MIN_INTERVAL = 2.0  # min seconds between requests

# --- Historical scans ---
MAX_SCAN_HISTORY = 10

# Crypto filter — ONLY unambiguous tokens
# Removed: sol, eth, ada, link, dot, bnb, matic, ltc (too many false positives)
CRYPTO_RE = re.compile(
    r"\b("
    r"bitcoin|btc|ethereum|solana|crypto(?:currenc(?:y|ies))?|"
    r"dogecoin|doge|ripple|xrp|cardano|avalanche|avax|litecoin|"
    r"chainlink|polkadot|shiba\s*inu|pepe\s*coin|memecoin|"
    r"token\s+price|coin\s+price|crypto\s+price|"
    r"defi|nft|blockchain|stablecoin|altcoin|"
    r"binance|uniswap|aave|satoshi|gwei"
    r")\b",
    re.IGNORECASE,
)
# "coinbase" removed from CRYPTO_RE — conflicts with COIN stock ticker.
# Coinbase stock markets ("Coinbase stock above $300?") are legitimate finance.
# Actual crypto questions about "Coinbase exchange" will still match via "crypto" keyword.
_COINBASE_STOCK_RE = re.compile(
    r"\bcoinbase\b(?!\s+(?:stock|share|price|ipo|earning))", re.IGNORECASE
)


def is_crypto(question: str) -> bool:
    if CRYPTO_RE.search(question):
        return True
    # "coinbase" alone (not "coinbase stock/share/price") → crypto
    if _COINBASE_STOCK_RE.search(question):
        return True
    return False

# ---------------------------------------------------------------------------
# Weather Analyzer — external edge detection
# ---------------------------------------------------------------------------

# Map city names / aliases → (lat, lon) for common Polymarket weather markets
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.71, -74.01), "nyc": (40.71, -74.01), "manhattan": (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "chicago": (41.88, -87.63), "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07), "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49), "san diego": (32.72, -117.16),
    "dallas": (32.78, -96.80), "austin": (30.27, -97.74),
    "miami": (25.76, -80.19), "atlanta": (33.75, -84.39),
    "boston": (42.36, -71.06), "seattle": (47.61, -122.33),
    "denver": (39.74, -104.99), "nashville": (36.16, -86.78),
    "washington": (38.91, -77.04), "dc": (38.91, -77.04),
    "san francisco": (37.77, -122.42), "sf": (37.77, -122.42),
    "las vegas": (36.17, -115.14), "portland": (45.51, -122.68),
    "detroit": (42.33, -83.05), "minneapolis": (44.98, -93.27),
    "tampa": (27.95, -82.46), "charlotte": (35.23, -80.84),
    "columbus": (39.96, -82.99), "indianapolis": (39.77, -86.16),
    "jacksonville": (30.33, -81.66), "memphis": (35.15, -90.05),
    "oklahoma city": (35.47, -97.52), "raleigh": (35.78, -78.64),
    "baltimore": (39.29, -76.61), "milwaukee": (43.04, -87.91),
    "kansas city": (39.10, -94.58), "sacramento": (38.58, -121.49),
    "cleveland": (41.50, -81.69), "pittsburgh": (40.44, -80.00),
    "cincinnati": (39.10, -84.51), "orlando": (28.54, -81.38),
    "st. louis": (38.63, -90.20), "st louis": (38.63, -90.20),
    "salt lake city": (40.76, -111.89), "san jose": (37.34, -121.89),
    "new orleans": (29.95, -90.07),
    "london": (51.51, -0.13), "paris": (48.86, 2.35), "tokyo": (35.68, 139.69),
    "toronto": (43.65, -79.38), "sydney": (-33.87, 151.21),
    "berlin": (52.52, 13.41), "madrid": (40.42, -3.70), "rome": (41.90, 12.50),
    "mexico city": (19.43, -99.13), "mumbai": (19.08, 72.88),
}

# --- Keyword-based weather question detection (flexible order) ---
# Step 1: detect question type via keywords
_TEMP_KEYWORDS_RE = re.compile(
    r"\b(?:temperature|high|low|temp|hot|cold|warm|heat|freeze|frost)\b",
    re.IGNORECASE,
)
_RAIN_KEYWORDS_RE = re.compile(
    r"\b(?:rain|precipitation|snow|storm|shower|drizzle)\b",
    re.IGNORECASE,
)
# Step 2: extract threshold + unit (city found separately via _find_city)
# Anchored patterns: prefer number with explicit unit, fallback to number after keywords
_TEMP_THRESHOLD_UNIT_RE = re.compile(
    r"(?P<threshold>\d+(?:\.\d+)?)\s*(?P<unit>[°]\s*[FCfc]|fahrenheit|celsius|[FCfc]\b)",
    re.IGNORECASE,
)
_TEMP_THRESHOLD_KEYWORD_RE = re.compile(
    r"(?:above|over|exceed|reach|hit|at least|below|under|drop to)\s+"
    r"(?P<threshold>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Step 3: optional specific date in question ("March 5", "on February 28")
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_IN_QUESTION_RE = re.compile(
    r"\b(?P<month>" + "|".join(_MONTH_MAP.keys()) + r")\s+(?P<day>\d{1,2})\b",
    re.IGNORECASE,
)


def _parse_target_date(question: str, now_dt: datetime) -> datetime | None:
    """Extract a specific target date from question text (e.g. 'March 5').

    Returns a timezone-aware datetime at end-of-day UTC, or None.
    """
    m = _DATE_IN_QUESTION_RE.search(question)
    if not m:
        return None
    month = _MONTH_MAP.get(m.group("month").lower())
    day = int(m.group("day"))
    if not month or day < 1 or day > 31:
        return None
    year = now_dt.year
    try:
        target = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
    except ValueError:
        return None
    # If target is in the past, try next year
    if target < now_dt:
        try:
            target = datetime(year + 1, month, day, 23, 59, 59, tzinfo=timezone.utc)
        except ValueError:
            return None
    return target


_owm_cache: dict[str, tuple[float, object]] = {}
_owm_lock = threading.Lock()


def _fetch_owm_forecast(lat: float, lon: float) -> list[dict] | None:
    """Fetch 5-day/3h forecast from OpenWeatherMap. Returns list of entries or None."""
    if not OWM_API_KEY:
        return None
    cache_key = f"{lat:.2f},{lon:.2f}"
    now_ts = _time.time()
    with _owm_lock:
        # Purge expired entries to prevent unbounded cache growth
        expired = [k for k, (ts, _) in _owm_cache.items() if now_ts - ts >= OWM_CACHE_TTL]
        for k in expired:
            del _owm_cache[k]
        if cache_key in _owm_cache:
            ts, data = _owm_cache[cache_key]
            if now_ts - ts < OWM_CACHE_TTL:
                return data

    try:
        resp = requests.get(OWM_FORECAST_URL, params={
            "lat": lat, "lon": lon, "appid": OWM_API_KEY,
            "units": "imperial",  # Fahrenheit — Polymarket is US-centric
        }, timeout=5)
        resp.raise_for_status()
        entries = resp.json().get("list", [])
        with _owm_lock:
            _owm_cache[cache_key] = (_time.time(), entries)
        return entries
    except Exception:
        return None


def _find_city(text: str) -> tuple[str, float, float] | None:
    """Extract city name from text and return (name, lat, lon) or None."""
    text_lower = text.lower()
    # Direct match in known cities (longest match first)
    best = None
    for city_name, (lat, lon) in CITY_COORDS.items():
        if city_name in text_lower:
            if best is None or len(city_name) > len(best[0]):
                best = (city_name, lat, lon)
    return best


def _to_fahrenheit(val: float, unit: str | None) -> float:
    """Convert to Fahrenheit if unit looks like Celsius."""
    if unit and unit.strip().lower() in ("c", "celsius", "°c"):
        return val * 9 / 5 + 32
    return val  # default: already Fahrenheit


def _horizon_confidence(entries: list[dict], now_dt: datetime | None = None) -> str:
    """Determine confidence level based on forecast horizon (item 12).

    J+0-1 (0-24h)  → high
    J+2-3 (24-72h) → medium
    J+4-5 (72h+)   → low
    Uses the latest entry's timestamp to determine horizon.
    """
    if not entries:
        return "low"
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    latest_dt = max(
        datetime.fromtimestamp(e.get("dt", 0), tz=timezone.utc)
        for e in entries
    )
    hours_out = (latest_dt - now_dt).total_seconds() / 3600
    if hours_out <= 24:
        return "high"
    elif hours_out <= 72:
        return "medium"
    return "low"


def analyze_weather(question: str, end_dt: datetime) -> dict | None:
    """Analyze a weather market question against OWM forecast data.

    Uses keyword detection (not rigid regex) for flexible question parsing.
    Temperature: sigmoid calibration for probability estimation.
    Rain: composite formula P(at least one) = 1 - prod(1 - pop_i).
    Confidence: degrades by forecast horizon (J+1 vs J+5).

    Return keys:
        estimated_prob (float 0-1): our probability estimate
        source (str): "OpenWeatherMap"
        analysis (str): human-readable explanation
        confidence (str): "high" / "medium" / "low"
        data_point (str): the key data value used
    """
    if not OWM_API_KEY:
        return None

    q_lower = question.lower()
    is_temp = bool(_TEMP_KEYWORDS_RE.search(question))
    is_rain = bool(_RAIN_KEYWORDS_RE.search(question))

    if not is_temp and not is_rain:
        return None

    # --- Find city (flexible — works regardless of question order) ---
    city_info = _find_city(question)
    if not city_info:
        return None
    city_name, lat, lon = city_info

    entries = _fetch_owm_forecast(lat, lon)
    if not entries:
        return None

    # Filter entries: use specific date if found, otherwise up to market end date
    now_dt = datetime.now(timezone.utc)
    target_dt = _parse_target_date(question, now_dt)
    filter_end = target_dt if target_dt and target_dt <= end_dt else end_dt

    relevant = []
    for e in entries:
        dt_unix = e.get("dt", 0)
        entry_dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
        if target_dt:
            # For specific date: only entries on that day
            if entry_dt.date() == target_dt.date():
                relevant.append(e)
        else:
            if entry_dt <= filter_end:
                relevant.append(e)
    if not relevant:
        return None

    # Confidence degrades by forecast horizon (item 12)
    confidence = _horizon_confidence(relevant, now_dt)

    # --- Temperature analysis (sigmoid calibration, item 3) ---
    if is_temp:
        # Extract threshold: prefer number with unit (80F), fallback to number after keyword
        m_thresh = _TEMP_THRESHOLD_UNIT_RE.search(question)
        if m_thresh:
            threshold = float(m_thresh.group("threshold"))
            unit = m_thresh.group("unit")
        else:
            m_thresh = _TEMP_THRESHOLD_KEYWORD_RE.search(question)
            if not m_thresh:
                return None
            threshold = float(m_thresh.group("threshold"))
            unit = None
        threshold_f = _to_fahrenheit(threshold, unit)

        is_below = any(w in q_lower for w in
                       ["below", "under", "stay under", "drop to", "fall to",
                        "less than", "lower than", "won't exceed", "not reach",
                        "not exceed", "won't reach"])
        is_above = not is_below and any(w in q_lower for w in
                       ["above", "over", "exceed", "reach", "hit", "at least",
                        "higher than", "more than", "surpass", "top"])

        highs = [e.get("main", {}).get("temp_max", 0) for e in relevant]
        max_high = max(highs) if highs else 0

        # Sigmoid calibration: prob = 1/(1 + exp(-margin/k))
        # margin = max_forecast - threshold (for "above" questions)
        # k=3 gives smooth transition around threshold
        if is_below:
            margin = threshold_f - max_high
            prob = 1.0 / (1.0 + math.exp(-margin / 3.0))
            direction = "below"
        else:
            # Default to "above" for ambiguous phrasing
            margin = max_high - threshold_f
            prob = 1.0 / (1.0 + math.exp(-margin / 3.0))
            direction = "above"

        return {
            "estimated_prob": round(prob, 3),
            "source": "OpenWeatherMap",
            "analysis": f"Forecast: max {max_high:.0f}°F for {city_name.title()} "
                        f"({len(relevant)} pts, {confidence} conf). "
                        f"Threshold: {direction} {threshold_f:.0f}°F.",
            "confidence": confidence,
            "data_point": f"{max_high:.0f}°F max forecast",
        }

    # --- Rain/precipitation analysis (composite formula, item 5) ---
    if is_rain:
        is_snow = "snow" in q_lower

        if is_snow:
            # For snow questions: use only entries with snow in weather description
            # AND weight by pop for those entries
            snow_pops = []
            for e in relevant:
                weather_list = e.get("weather", [])
                has_snow = any("snow" in w.get("main", "").lower()
                               for w in weather_list)
                pop = e.get("pop", 0)
                if has_snow:
                    snow_pops.append(pop)
                else:
                    # Scale down non-snow entries (some precip could turn to snow)
                    temp = e.get("main", {}).get("temp", 40)
                    if temp <= 34:  # near freezing — precip likely snow
                        snow_pops.append(pop * 0.8)
                    elif temp <= 38:
                        snow_pops.append(pop * 0.3)
                    # else: too warm, skip
            pop_values = snow_pops if snow_pops else [0.0]
            precip_type = "snow"
        else:
            pop_values = [e.get("pop", 0) for e in relevant]
            precip_type = "rain"

        max_pop = max(pop_values) if pop_values else 0
        avg_pop = sum(pop_values) / len(pop_values) if pop_values else 0

        # Composite: P(at least one period with precip) = 1 - prod(1 - pop_i)
        no_precip = 1.0
        for pop in pop_values:
            no_precip *= (1.0 - pop)
        prob = 1.0 - no_precip

        return {
            "estimated_prob": round(prob, 3),
            "source": "OpenWeatherMap",
            "analysis": f"Forecast: {precip_type} composite prob {prob*100:.0f}% "
                        f"(max PoP {max_pop*100:.0f}%, avg {avg_pop*100:.0f}%) "
                        f"for {city_name.title()} ({len(relevant)} pts, {confidence} conf).",
            "confidence": confidence,
            "data_point": f"{prob*100:.0f}% composite PoP",
        }

    return None


# ---------------------------------------------------------------------------
# Finance Analyzer — stock/index/commodity edge detection
# ---------------------------------------------------------------------------
# Uses Yahoo Finance for current price + historical volatility.
# Estimates probability via log-normal diffusion model (Black-Scholes style).
# No API key required.

_finance_cache: dict[str, tuple[float, object]] = {}
_finance_lock = threading.Lock()

# Map company names / indices / commodities → Yahoo Finance ticker
FINANCE_TICKERS: dict[str, str] = {
    # US stocks (top by Polymarket market frequency)
    "tesla": "TSLA", "tsla": "TSLA",
    "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN",
    "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "meta": "META", "facebook": "META",
    "netflix": "NFLX", "nflx": "NFLX",
    "gamestop": "GME", "gme": "GME",
    "amd": "AMD",
    "palantir": "PLTR", "pltr": "PLTR",
    "coinbase": "COIN",
    "trump media": "DJT", "djt": "DJT",
    # Indices
    "s&p 500": "^GSPC", "s&p": "^GSPC", "spy": "SPY",
    "nasdaq": "^IXIC", "qqq": "QQQ",
    "dow jones": "^DJI", "dow": "^DJI",
    "russell 2000": "^RUT",
    # Commodities
    "gold": "GC=F",
    "oil": "CL=F", "crude oil": "CL=F", "wti": "CL=F",
    "silver": "SI=F",
    "natural gas": "NG=F",
    "copper": "HG=F",
    "platinum": "PL=F",
    # Forex (major pairs via Yahoo Finance)
    "euro": "EURUSD=X", "eur/usd": "EURUSD=X", "eurusd": "EURUSD=X",
    "gbp": "GBPUSD=X", "gbp/usd": "GBPUSD=X", "british pound": "GBPUSD=X",
    "yen": "USDJPY=X", "usd/jpy": "USDJPY=X", "usdjpy": "USDJPY=X",
    "yuan": "USDCNY=X", "usd/cny": "USDCNY=X", "renminbi": "USDCNY=X",
    "dollar index": "DX-Y.NYB", "dxy": "DX-Y.NYB",
    # Additional stocks (Polymarket frequent)
    "uber": "UBER",
    "disney": "DIS", "dis": "DIS",
    "intel": "INTC", "intc": "INTC",
    "boeing": "BA",
    "jpmorgan": "JPM", "jpm": "JPM",
    "berkshire": "BRK-B",
    "spotify": "SPOT",
    "snap": "SNAP", "snapchat": "SNAP",
    "robinhood": "HOOD",
    "rivian": "RIVN",
    "lucid": "LCID",
    "affirm": "AFRM",
    "sofi": "SOFI",
    "crowdstrike": "CRWD",
    "snowflake": "SNOW",
    # Crypto-adjacent stocks (not crypto themselves)
    "microstrategy": "MSTR", "mstr": "MSTR",
    "marathon digital": "MARA",
    "riot platforms": "RIOT",
}

_FINANCE_KEYWORDS_RE = re.compile(
    r"\b(?:stock|share|price|close|trading|"
    r"s&p|nasdaq|dow|gold|oil|silver|copper|platinum|natural gas|"
    r"euro|eur/usd|gbp|yen|yuan|dollar index|dxy|forex|currency|"
    r"tesla|apple|amazon|google|nvidia|microsoft|meta|netflix|"
    r"gamestop|palantir|amd|uber|disney|intel|boeing|jpmorgan|"
    r"berkshire|spotify|snap|robinhood|rivian|lucid|affirm|sofi|"
    r"crowdstrike|snowflake|microstrategy|marathon digital|riot platforms)\b",
    re.IGNORECASE,
)

# Match $1,234.56 or bare numbers after price-related words
_PRICE_THRESHOLD_RE = re.compile(
    r"(?:\$\s*([\d,]+(?:\.\d+)?)|"
    r"(?:above|over|exceed|reach|hit|below|under|at least)\s+\$?([\d,]+(?:\.\d+)?))",
    re.IGNORECASE,
)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _find_ticker(question: str) -> str | None:
    """Extract Yahoo Finance ticker symbol from question text.

    Returns ticker string or None. Longest match wins.
    """
    q_lower = question.lower()
    best = None
    best_len = 0
    for name, ticker in FINANCE_TICKERS.items():
        if name in q_lower and len(name) > best_len:
            best = ticker
            best_len = len(name)
    return best


def _fetch_yahoo_chart(ticker: str) -> dict | None:
    """Fetch current price + historical volatility from Yahoo Finance.

    Returns {price, daily_vol, ticker} or None.
    Cache: 5 min TTL per ticker, expired entries purged.
    """
    now_ts = _time.time()
    with _finance_lock:
        expired = [k for k, (ts, _) in _finance_cache.items()
                   if now_ts - ts >= FINANCE_CACHE_TTL]
        for k in expired:
            del _finance_cache[k]
        if ticker in _finance_cache:
            ts, data = _finance_cache[ticker]
            if now_ts - ts < FINANCE_CACHE_TTL:
                return data

    try:
        chart = None
        for base_url in [YAHOO_CHART_URL,
                         "https://query2.finance.yahoo.com/v8/finance/chart"]:
            try:
                resp = requests.get(
                    f"{base_url}/{ticker}",
                    params={"range": "3mo", "interval": "1d"},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=8,
                )
                resp.raise_for_status()
                chart = resp.json().get("chart", {}).get("result", [])
                if chart:
                    break
            except Exception:
                continue
        if not chart:
            return None

        meta = chart[0].get("meta", {})
        current_price = meta.get("regularMarketPrice", 0)
        closes = (chart[0].get("indicators", {})
                  .get("quote", [{}])[0].get("close", []))
        closes = [c for c in closes if c is not None and c > 0]

        if len(closes) < 10 or current_price <= 0:
            return None

        # Daily volatility from log returns — EWMA (λ=0.94, RiskMetrics style)
        # Weights recent observations more heavily for reactive vol estimate
        log_returns = [math.log(closes[i] / closes[i - 1])
                       for i in range(1, len(closes))
                       if closes[i - 1] > 0]
        if not log_returns:
            return None
        ewma_lambda = 0.94
        var_ewma = log_returns[0] ** 2
        for r in log_returns[1:]:
            var_ewma = ewma_lambda * var_ewma + (1 - ewma_lambda) * r ** 2
        daily_vol = math.sqrt(var_ewma)

        result_data = {"price": current_price, "daily_vol": daily_vol,
                       "ticker": ticker}
        with _finance_lock:
            _finance_cache[ticker] = (_time.time(), result_data)
        return result_data
    except requests.exceptions.HTTPError as exc:
        _logger.warning("Yahoo Finance HTTP error for %s: %s", ticker, exc)
        return None
    except requests.exceptions.Timeout:
        _logger.warning("Yahoo Finance timeout for %s", ticker)
        return None
    except Exception as exc:
        _logger.warning("Yahoo Finance error for %s: %s", ticker, exc)
        return None


def analyze_finance(question: str, end_dt: datetime) -> dict | None:
    """Analyze financial market question using current price + volatility model.

    Uses log-normal diffusion: P(S_T > K) = Φ(-z) where
    z = ln(K/S) / (σ√T),  S=current price, K=threshold, T=days, σ=daily vol.
    No API key required — uses Yahoo Finance public chart endpoint.
    """
    if not _FINANCE_KEYWORDS_RE.search(question):
        return None

    ticker = _find_ticker(question)
    if not ticker:
        return None

    # Extract price threshold
    m = _PRICE_THRESHOLD_RE.search(question)
    if not m:
        return None
    raw_val = m.group(1) or m.group(2)
    threshold = float(raw_val.replace(",", ""))
    if threshold <= 0:
        return None

    quote = _fetch_yahoo_chart(ticker)
    if not quote:
        return None

    current = quote["price"]
    daily_vol = quote["daily_vol"]

    now_dt = datetime.now(timezone.utc)
    days = max((end_dt - now_dt).total_seconds() / 86400, 0.1)

    # Log-normal probability with drift (Black-Scholes style)
    # drift = (r - σ²/2) × T, using risk-free rate ~4.5% annualized
    RISK_FREE_ANNUAL = 0.045
    r_daily = RISK_FREE_ANNUAL / 252
    sigma_period = daily_vol * math.sqrt(days)
    drift = (r_daily - 0.5 * daily_vol ** 2) * days
    if sigma_period < 0.001:
        prob_above = 1.0 if current >= threshold else 0.0
    else:
        z = (math.log(threshold / current) - drift) / sigma_period
        prob_above = 1.0 - _normal_cdf(z)

    q_lower = question.lower()
    is_below = any(w in q_lower for w in
                   ["below", "under", "fall", "drop", "decline", "crash",
                    "sink", "less than", "lower than"])
    prob = (1.0 - prob_above) if is_below else prob_above

    # Confidence: shorter horizon + closer to target = more reliable
    if days <= 7:
        confidence = "high"
    elif days <= 30:
        confidence = "medium"
    else:
        confidence = "low"

    pct_away = abs(threshold - current) / current * 100
    direction = "below" if is_below else "above"

    return {
        "estimated_prob": round(prob, 3),
        "source": "Yahoo Finance",
        "analysis": (f"{ticker} @ ${current:,.2f}, target {direction} "
                     f"${threshold:,.2f} ({pct_away:.1f}% away, {days:.0f}d, "
                     f"vol {daily_vol*100:.1f}%/d). P={prob*100:.0f}%."),
        "confidence": confidence,
        "data_point": f"${current:,.2f} ({ticker})",
    }


# ---------------------------------------------------------------------------
# Sports Analyzer — bookmaker consensus edge detection
# ---------------------------------------------------------------------------
# Compares Polymarket sports prices to bookmaker consensus odds via The Odds API.
# Devigging removes bookmaker margin to get "true" probabilities.
# Optional: requires THE_ODDS_API_KEY env var (free tier: 500 req/month).

_odds_cache: dict[str, tuple[float, object]] = {}
_odds_lock = threading.Lock()

# Sport keywords → The Odds API sport keys
SPORT_KEYS: dict[str, str] = {
    # Basketball
    "nba": "basketball_nba",
    "ncaa basketball": "basketball_ncaab", "march madness": "basketball_ncaab",
    "wnba": "basketball_wnba",
    # Football
    "nfl": "americanfootball_nfl",
    "super bowl": "americanfootball_nfl_super_bowl",
    "ncaa football": "americanfootball_ncaaf",
    "college football": "americanfootball_ncaaf",
    # Baseball
    "mlb": "baseball_mlb", "world series": "baseball_mlb",
    # Hockey
    "nhl": "icehockey_nhl", "stanley cup": "icehockey_nhl",
    # Soccer
    "premier league": "soccer_epl", "epl": "soccer_epl",
    "champions league": "soccer_uefa_champions_league",
    "la liga": "soccer_spain_la_liga",
    "bundesliga": "soccer_germany_bundesliga",
    "serie a": "soccer_italy_serie_a",
    "ligue 1": "soccer_france_ligue_one",
    "mls": "soccer_usa_mls",
    # Combat
    "ufc": "mma_mixed_martial_arts", "mma": "mma_mixed_martial_arts",
    "boxing": "boxing_boxing",
    # Motorsport
    "formula 1": "motorsport_formula_one", "f1 ": "motorsport_formula_one",
}

_SPORTS_DETECT_RE = re.compile(
    r"\b(?:win|beat|defeat|champion(?:ship)?|final|playoff|match|game|fight|bout|"
    r"nba|nfl|mlb|nhl|ufc|mma|boxing|premier league|champions league|"
    r"super bowl|world series|stanley cup|march madness|"
    r"lakers|celtics|warriors|nets|heat|bucks|sixers|knicks|"
    r"chiefs|eagles|ravens|49ers|cowboys|bills|dolphins|"
    r"yankees|dodgers|astros|braves|mets|phillies|"
    r"oilers|rangers|panthers|bruins|avalanche|"
    r"real madrid|barcelona|manchester|liverpool|arsenal|chelsea|"
    r"psg|bayern|juventus|inter milan)\b",
    re.IGNORECASE,
)


def _find_sport(question: str) -> str | None:
    """Find The Odds API sport key from question keywords. Longest match wins."""
    q_lower = question.lower()
    best = None
    best_len = 0
    for keyword, sport_key in SPORT_KEYS.items():
        if keyword in q_lower and len(keyword) > best_len:
            best = sport_key
            best_len = len(keyword)
    return best


def _fetch_odds(sport_key: str) -> list[dict] | None:
    """Fetch upcoming odds from The Odds API. Cache 5 min, purge expired."""
    if not ODDS_API_KEY:
        return None
    now_ts = _time.time()
    with _odds_lock:
        expired = [k for k, (ts, _) in _odds_cache.items()
                   if now_ts - ts >= ODDS_CACHE_TTL]
        for k in expired:
            del _odds_cache[k]
        if sport_key in _odds_cache:
            ts, data = _odds_cache[sport_key]
            if now_ts - ts < ODDS_CACHE_TTL:
                return data

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
        with _odds_lock:
            _odds_cache[sport_key] = (_time.time(), events)
        return events
    except Exception:
        return None


def _match_event(question: str, events: list[dict]) -> dict | None:
    """Find the event best matching the question by team name overlap.

    Returns event dict or None (requires >= 4 chars matched).
    """
    q_lower = question.lower()
    best = None
    best_score = 0
    for event in events:
        home = event.get("home_team", "").lower()
        away = event.get("away_team", "").lower()
        score = 0
        for team in [home, away]:
            if team in q_lower:
                score += len(team)
            else:
                for word in team.split():
                    if len(word) >= 4 and word in q_lower:
                        score += len(word)
        if score > best_score:
            best = event
            best_score = score
    return best if best_score >= 4 else None


def _find_target_team(question: str, event: dict) -> str:
    """Determine which team the question is about."""
    q_lower = question.lower()
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    for team in [home, away]:
        if team.lower() in q_lower:
            return team
        for word in team.lower().split():
            if len(word) >= 4 and word in q_lower:
                return team
    return home  # fallback to home


def _consensus_probability(event: dict, team_name: str) -> tuple[float, int]:
    """Calculate devigged consensus probability for a team.

    Averages implied probability across all bookmakers after removing vig.
    Handles 3-way markets (with draw) properly — draw probability is excluded
    from team probability, not split between teams.
    Returns (probability, num_bookmakers).
    """
    probs: list[float] = []
    team_lower = team_name.lower()
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            # Find target team's decimal odds
            target_odds = None
            other_odds: list[float] = []
            for out in outcomes:
                name = out.get("name", "").lower()
                price = out.get("price", 0)
                if price <= 1.0:
                    continue
                if team_lower in name or name in team_lower:
                    target_odds = price
                else:
                    other_odds.append(price)
            if target_odds and other_odds:
                # Devig: remove bookmaker margin
                raw = 1.0 / target_odds
                total = raw + sum(1.0 / o for o in other_odds)
                if total > 0:
                    probs.append(raw / total)
    if not probs:
        return 0.0, 0
    return sum(probs) / len(probs), len(probs)


def _has_draw_market(event: dict) -> bool:
    """Check if event has a draw outcome (3-way market like soccer)."""
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) >= 3:
                for out in outcomes:
                    if out.get("name", "").lower() == "draw":
                        return True
    return False


def analyze_sports(question: str, end_dt: datetime) -> dict | None:
    """Analyze sports market using bookmaker consensus odds.

    Fetches odds from The Odds API, matches event by team names,
    calculates devigged consensus probability across bookmakers.
    Confidence combines time horizon AND bookmaker count.
    Requires THE_ODDS_API_KEY env var.
    """
    if not ODDS_API_KEY:
        return None
    if not _SPORTS_DETECT_RE.search(question):
        return None

    sport_key = _find_sport(question)
    if not sport_key:
        return None

    events = _fetch_odds(sport_key)
    if not events:
        return None

    matched = _match_event(question, events)
    if not matched:
        return None

    target = _find_target_team(question, matched)
    prob, num_books = _consensus_probability(matched, target)
    if num_books < 2:
        return None  # need at least 2 bookmakers for consensus

    home = matched.get("home_team", "?")
    away = matched.get("away_team", "?")

    # Confidence: combine bookmaker count AND time horizon
    now_dt = datetime.now(timezone.utc)
    days_to_event = max((end_dt - now_dt).total_seconds() / 86400, 0)
    # Bookmaker score: 2=0, 3-4=1, 5+=2
    book_score = 0 if num_books < 3 else (1 if num_books < 5 else 2)
    # Time score: <3d=2, 3-14d=1, 14d+=0
    time_score = 2 if days_to_event < 3 else (1 if days_to_event < 14 else 0)
    combined = book_score + time_score  # 0-4
    if combined >= 3:
        confidence = "high"
    elif combined >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    draw_note = ""
    if _has_draw_market(matched):
        draw_note = " (3-way: draw possible)"

    return {
        "estimated_prob": round(prob, 3),
        "source": "The Odds API",
        "analysis": (f"Consensus {num_books} bookmakers: {target} "
                     f"@ {prob*100:.0f}%. "
                     f"Match: {home} vs {away}.{draw_note}"),
        "confidence": confidence,
        "data_point": f"{prob*100:.0f}% ({num_books} books)",
    }


# ---------------------------------------------------------------------------
# Earnings Analyzer — Yahoo Finance earnings surprise detection
# ---------------------------------------------------------------------------
# Detects "Will X beat earnings?" markets and compares to analyst consensus.
# Uses Yahoo Finance public endpoint — no API key needed.

_earnings_cache: dict[str, tuple[float, object]] = {}
_earnings_lock = threading.Lock()
EARNINGS_CACHE_TTL = 600  # 10 min

_EARNINGS_KEYWORDS_RE = re.compile(
    r"\b(?:earnings|revenue|eps|profit|quarterly|beat|miss|"
    r"q[1-4]\s*(?:20\d{2})?|earnings\s*call|guidance|"
    r"report\s*(?:revenue|earnings|profit)|fiscal)\b",
    re.IGNORECASE,
)

_EARNINGS_BEAT_RE = re.compile(
    r"\b(?:beat|exceed|surpass|top|above)\b", re.IGNORECASE
)
_EARNINGS_MISS_RE = re.compile(
    r"\b(?:miss|below|under|fall\s+short|disappoint)\b", re.IGNORECASE
)


def _fetch_earnings_data(ticker: str) -> dict | None:
    """Fetch earnings estimates from Yahoo Finance.

    Returns {ticker, est_eps, actual_eps, surprise_pct, beat_rate} or None.
    """
    now_ts = _time.time()
    cache_key = f"earn_{ticker}"
    with _earnings_lock:
        expired = [k for k, (ts, _) in _earnings_cache.items()
                   if now_ts - ts >= EARNINGS_CACHE_TTL]
        for k in expired:
            del _earnings_cache[k]
        if cache_key in _earnings_cache:
            ts, data = _earnings_cache[cache_key]
            if now_ts - ts < EARNINGS_CACHE_TTL:
                return data

    try:
        result = None
        for base in ["https://query1.finance.yahoo.com",
                      "https://query2.finance.yahoo.com"]:
            try:
                resp = requests.get(
                    f"{base}/v10/finance/quoteSummary/{ticker}",
                    params={"modules": "earningsTrend,earnings"},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=8,
                )
                resp.raise_for_status()
                result = resp.json().get("quoteSummary", {}).get("result", [])
                if result:
                    break
            except Exception:
                continue
        if not result:
            return None

        modules = result[0]

        # Extract beat rate from historical earnings
        earnings_data = modules.get("earnings", {})
        earnings_chart = earnings_data.get("earningsChart", {})
        quarterly = earnings_chart.get("quarterly", [])

        beats = 0
        total = 0
        for q in quarterly:
            actual_raw = q.get("actual", {})
            est_raw = q.get("estimate", {})
            actual_val = actual_raw.get("raw") if isinstance(actual_raw, dict) else None
            est_val = est_raw.get("raw") if isinstance(est_raw, dict) else None
            if actual_val is not None and est_val is not None:
                total += 1
                if actual_val > est_val:
                    beats += 1
        beat_rate = beats / total if total >= 2 else 0.65  # default 65%

        # Extract current estimate + analyst revisions from earningsTrend
        trend = modules.get("earningsTrend", {}).get("trend", [])
        est_eps = None
        revision_trend = 0  # positive = upgrades, negative = downgrades
        for t in trend:
            period = t.get("period", "")
            if period in ("0q", "+1q"):
                eps_est = t.get("earningsEstimate", {})
                avg_raw = eps_est.get("avg", {})
                if isinstance(avg_raw, dict) and avg_raw.get("raw") is not None:
                    est_eps = avg_raw["raw"]
                # Analyst revisions: up vs down in last 7/30 days
                rev = t.get("epsTrend", {})
                rev_7d = rev.get("7daysAgo", {})
                rev_30d = rev.get("30daysAgo", {})
                cur_raw = rev.get("current", {})
                cur_est = cur_raw.get("raw") if isinstance(cur_raw, dict) else None
                r7 = rev_7d.get("raw") if isinstance(rev_7d, dict) else None
                r30 = rev_30d.get("raw") if isinstance(rev_30d, dict) else None
                if cur_est is not None:
                    if r7 is not None and r7 != 0:
                        revision_trend = (cur_est - r7) / abs(r7)
                    elif r30 is not None and r30 != 0:
                        revision_trend = (cur_est - r30) / abs(r30)
                break

        # Revenue growth from recent quarters
        rev_growth = None
        financials = earnings_data.get("financialsChart", {})
        yearly = financials.get("yearly", [])
        if len(yearly) >= 2:
            prev_rev = yearly[-2].get("revenue", {}).get("raw", 0)
            cur_rev = yearly[-1].get("revenue", {}).get("raw", 0)
            if prev_rev > 0:
                rev_growth = (cur_rev - prev_rev) / prev_rev

        data = {
            "ticker": ticker,
            "beat_rate": round(beat_rate, 3),
            "est_eps": est_eps,
            "num_quarters": total,
            "revision_trend": round(revision_trend, 4),
            "revenue_growth": round(rev_growth, 4) if rev_growth is not None else None,
        }
        with _earnings_lock:
            _earnings_cache[cache_key] = (_time.time(), data)
        return data
    except Exception as exc:
        _logger.warning("Yahoo earnings error for %s: %s", ticker, exc)
        return None


def analyze_earnings(question: str, end_dt: datetime) -> dict | None:
    """Analyze earnings market using historical beat rate + analyst consensus.

    Detects "Will X beat earnings?" style questions and returns probability
    based on historical beat rate (last 4 quarters).
    No API key required — uses Yahoo Finance.
    """
    if not _EARNINGS_KEYWORDS_RE.search(question):
        return None

    ticker = _find_ticker(question)
    if not ticker:
        return None

    data = _fetch_earnings_data(ticker)
    if not data:
        return None

    beat_rate = data["beat_rate"]
    est_eps = data["est_eps"]
    num_q = data["num_quarters"]
    revision_trend = data.get("revision_trend", 0)
    revenue_growth = data.get("revenue_growth")

    # Determine direction: beat or miss
    is_miss = bool(_EARNINGS_MISS_RE.search(question))
    is_beat = bool(_EARNINGS_BEAT_RE.search(question)) or not is_miss

    if is_miss:
        prob = 1.0 - beat_rate
        direction = "miss"
    else:
        prob = beat_rate
        direction = "beat"

    # Adjust probability based on analyst revision trend
    # Upward revisions → higher beat probability (analysts still behind)
    # Downward revisions → lower beat probability (estimates being cut)
    if revision_trend > 0.02 and direction == "beat":
        prob = min(prob + 0.05, 0.95)  # slight boost for upward revisions
    elif revision_trend < -0.02 and direction == "beat":
        prob = max(prob - 0.05, 0.05)  # slight penalty for downgrades
    elif revision_trend > 0.02 and direction == "miss":
        prob = max(prob - 0.05, 0.05)
    elif revision_trend < -0.02 and direction == "miss":
        prob = min(prob + 0.05, 0.95)

    # Revenue growth as supporting signal
    rev_note = ""
    if revenue_growth is not None:
        if revenue_growth > 0.10 and direction == "beat":
            prob = min(prob + 0.03, 0.95)
            rev_note = f", rev +{revenue_growth*100:.0f}%"
        elif revenue_growth < -0.05 and direction == "beat":
            prob = max(prob - 0.03, 0.05)
            rev_note = f", rev {revenue_growth*100:.0f}%"

    # Confidence based on data quality
    if num_q >= 4:
        confidence = "high"
    elif num_q >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Adjust confidence by time horizon
    now_dt = datetime.now(timezone.utc)
    days = max((end_dt - now_dt).total_seconds() / 86400, 0)
    if days > 30:
        confidence = "low"
    elif days > 14 and confidence == "high":
        confidence = "medium"

    eps_note = f", EPS est ${est_eps:.2f}" if est_eps is not None else ""
    rev_trend_note = ""
    if abs(revision_trend) > 0.01:
        rev_trend_note = f", revisions {'↑' if revision_trend > 0 else '↓'}{abs(revision_trend)*100:.1f}%"

    return {
        "estimated_prob": round(prob, 3),
        "source": "Yahoo Finance Earnings",
        "analysis": (f"{ticker} historical {direction} rate: {beat_rate*100:.0f}% "
                     f"({num_q}Q data{eps_note}{rev_trend_note}{rev_note}). "
                     f"P({direction})={prob*100:.0f}%."),
        "confidence": confidence,
        "data_point": f"{beat_rate*100:.0f}% {direction} rate ({num_q}Q)",
    }


# ---------------------------------------------------------------------------
# Fed/Macro Analyzer — FRED API economic indicator detection
# ---------------------------------------------------------------------------
# Detects markets about Fed rates, CPI/inflation, unemployment, GDP.
# Uses FRED (Federal Reserve Economic Data) — free API key optional.
# Falls back to hardcoded recent values if no key.

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CACHE_TTL = 1800  # 30 min (macro data moves slowly)
_fred_cache: dict[str, tuple[float, object]] = {}
_fred_lock = threading.Lock()

# FRED series IDs for key economic indicators
FRED_SERIES: dict[str, dict] = {
    "fed_rate": {
        "series_id": "DFEDTARU",  # Fed funds upper target
        "label": "Fed Funds Rate",
        "unit": "%",
    },
    "cpi": {
        "series_id": "CPIAUCSL",  # CPI All Urban Consumers
        "label": "CPI",
        "unit": "index",
    },
    "cpi_yoy": {
        "series_id": "CPALTT01USM657N",  # CPI YoY %
        "label": "CPI YoY",
        "unit": "%",
    },
    "unemployment": {
        "series_id": "UNRATE",  # Unemployment Rate
        "label": "Unemployment Rate",
        "unit": "%",
    },
    "gdp_growth": {
        "series_id": "A191RL1Q225SBEA",  # Real GDP Growth
        "label": "GDP Growth",
        "unit": "%",
    },
    "nonfarm_payrolls": {
        "series_id": "PAYEMS",  # Total Nonfarm Payrolls
        "label": "Nonfarm Payrolls",
        "unit": "thousands",
    },
    "pce": {
        "series_id": "PCEPI",  # PCE Price Index
        "label": "PCE",
        "unit": "index",
    },
}

_FED_RATE_RE = re.compile(
    r"\b(?:fed(?:eral)?\s*(?:funds?\s*)?rate|interest\s*rate|fomc|"
    r"rate\s*(?:cut|hike|raise|hold|pause|unchanged|decision)|"
    r"(?:cut|hike|raise|hold|pause)\s*rates?|"
    r"basis\s*points?|bps|"
    r"fed\s*(?:meeting|decision|cut|hike|raise|hold))\b",
    re.IGNORECASE,
)

_INFLATION_RE = re.compile(
    r"\b(?:inflation|cpi|consumer\s*price|pce|"
    r"core\s*(?:inflation|cpi|pce)|"
    r"price\s*index)\b",
    re.IGNORECASE,
)

_UNEMPLOYMENT_RE = re.compile(
    r"\b(?:unemployment|jobless|nonfarm|payroll|jobs?\s*report|"
    r"labor\s*market|employment\s*(?:rate|report|data))\b",
    re.IGNORECASE,
)

_GDP_RE = re.compile(
    r"\b(?:gdp|gross\s*domestic|economic\s*growth|recession|"
    r"gdp\s*(?:growth|contraction|report))\b",
    re.IGNORECASE,
)

_MACRO_THRESHOLD_RE = re.compile(
    r"(?:above|over|exceed|reach|below|under|at least|higher than|lower than)"
    r"\s+(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)

_MACRO_CUT_HIKE_RE = re.compile(
    r"\b(?:cut|lower|reduce|ease|dovish)\b", re.IGNORECASE
)
_MACRO_HIKE_RE = re.compile(
    r"\b(?:hike|raise|increase|tighten|hawkish)\b", re.IGNORECASE
)
_MACRO_HOLD_RE = re.compile(
    r"\b(?:hold|unchanged|pause|no\s*change|maintain|steady)\b", re.IGNORECASE
)


def _fetch_fred_series(series_id: str) -> float | None:
    """Fetch latest value from FRED API. Returns float or None."""
    if not FRED_API_KEY:
        return None
    now_ts = _time.time()
    cache_key = f"fred_{series_id}"
    with _fred_lock:
        expired = [k for k, (ts, _) in _fred_cache.items()
                   if now_ts - ts >= FRED_CACHE_TTL]
        for k in expired:
            del _fred_cache[k]
        if cache_key in _fred_cache:
            ts, data = _fred_cache[cache_key]
            if now_ts - ts < FRED_CACHE_TTL:
                return data

    try:
        resp = requests.get(FRED_API_URL, params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }, timeout=8)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        for o in obs:
            val_str = o.get("value", ".")
            if val_str != ".":
                val = float(val_str)
                with _fred_lock:
                    _fred_cache[cache_key] = (_time.time(), val)
                return val
        return None
    except Exception as exc:
        _logger.warning("FRED API error for %s: %s", series_id, exc)
        return None


def _fetch_fred_trend(series_id: str, limit: int = 12) -> list[float]:
    """Fetch last N values from FRED for trend analysis."""
    if not FRED_API_KEY:
        return []
    try:
        resp = requests.get(FRED_API_URL, params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }, timeout=8)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        vals = []
        for o in obs:
            v = o.get("value", ".")
            if v != ".":
                vals.append(float(v))
        return vals  # most recent first
    except Exception:
        return []


def _rate_trend_probabilities(current_rate: float | None,
                              trend: list[float]) -> dict:
    """Derive hold/cut/hike probabilities from rate trend.

    Instead of static base rates, uses actual rate trajectory.
    Returns {hold, cut, hike} probabilities summing to 1.0.
    """
    if not trend or len(trend) < 2:
        # Fallback to base rates
        return {"hold": 0.75, "cut": 0.15, "hike": 0.10}

    # Count recent changes: how many of last N meetings had moves?
    changes = []
    for i in range(len(trend) - 1):
        diff = trend[i] - trend[i + 1]  # trend[0] = most recent
        if abs(diff) > 0.001:
            changes.append(diff)

    total_obs = len(trend) - 1
    n_holds = total_obs - len(changes)
    hold_rate = n_holds / total_obs if total_obs > 0 else 0.75

    # Determine direction momentum
    n_cuts = sum(1 for c in changes if c < 0)
    n_hikes = sum(1 for c in changes if c > 0)

    if len(changes) == 0:
        # All holds → strong hold probability
        return {"hold": 0.85, "cut": 0.08, "hike": 0.07}

    # Recent bias: look at last 3 moves
    recent = changes[:3] if len(changes) >= 3 else changes
    recent_direction = sum(1 if c < 0 else -1 for c in recent)

    if recent_direction > 0:
        # Cutting cycle
        cut_p = min(0.15 + 0.10 * len(recent), 0.50)
        hold_p = max(hold_rate - 0.05, 0.30)
        hike_p = max(1.0 - cut_p - hold_p, 0.02)
    elif recent_direction < 0:
        # Hiking cycle
        hike_p = min(0.15 + 0.10 * len(recent), 0.50)
        hold_p = max(hold_rate - 0.05, 0.30)
        cut_p = max(1.0 - hike_p - hold_p, 0.02)
    else:
        hold_p = max(hold_rate, 0.50)
        cut_p = (1.0 - hold_p) / 2
        hike_p = (1.0 - hold_p) / 2

    # Normalize
    total = hold_p + cut_p + hike_p
    return {
        "hold": round(hold_p / total, 3),
        "cut": round(cut_p / total, 3),
        "hike": round(hike_p / total, 3),
    }


def analyze_fed_macro(question: str, end_dt: datetime) -> dict | None:
    """Analyze Fed/macro market using FRED economic data + rate trend.

    Detects: rate decisions, CPI/inflation, unemployment, GDP markets.
    Uses FRED API when key available, with trend-based probability model.
    """
    q_lower = question.lower()

    # --- Fed rate decision markets ---
    if _FED_RATE_RE.search(question):
        current_rate = _fetch_fred_series("DFEDTARU")
        rate_trend = _fetch_fred_trend("DFEDTARU", 12)

        is_cut = bool(_MACRO_CUT_HIKE_RE.search(question))
        is_hike = bool(_MACRO_HIKE_RE.search(question))
        is_hold = bool(_MACRO_HOLD_RE.search(question))

        # Extract threshold if present (e.g., "rate above 4.5%")
        m_thresh = _MACRO_THRESHOLD_RE.search(question)
        threshold = float(m_thresh.group(1)) if m_thresh else None

        # Use trend-based probabilities instead of static base rates
        trend_probs = _rate_trend_probabilities(current_rate, rate_trend)

        if current_rate is not None:
            rate_str = f"{current_rate:.2f}%"
            trend_note = f" (trend: H={trend_probs['hold']*100:.0f}% C={trend_probs['cut']*100:.0f}% K={trend_probs['hike']*100:.0f}%)"

            if threshold is not None:
                is_below = any(w in q_lower for w in
                               ["below", "under", "lower", "less", "fall"])
                if is_below:
                    prob = 0.7 if current_rate > threshold else 0.3
                else:
                    prob = 0.7 if current_rate >= threshold else 0.3
                direction = "below" if is_below else "above"
                analysis = (f"Fed rate @ {rate_str}, target {direction} "
                            f"{threshold}%.{trend_note}")
            elif is_hold:
                prob = trend_probs["hold"]
                analysis = f"Fed rate @ {rate_str}. Trend-based hold P={prob*100:.0f}%.{trend_note}"
            elif is_cut:
                prob = trend_probs["cut"]
                analysis = f"Fed rate @ {rate_str}. Trend-based cut P={prob*100:.0f}%.{trend_note}"
            elif is_hike:
                prob = trend_probs["hike"]
                analysis = f"Fed rate @ {rate_str}. Trend-based hike P={prob*100:.0f}%.{trend_note}"
            else:
                prob = 0.50
                analysis = f"Fed rate @ {rate_str}. Ambiguous direction.{trend_note}"
        else:
            # No FRED key — use static base rates as fallback
            if is_hold:
                prob = 0.75
            elif is_cut:
                prob = 0.30
            elif is_hike:
                prob = 0.10
            else:
                prob = 0.50
            rate_str = "N/A"
            analysis = "Fed rate direction analysis (no FRED key for live data)."

        now_dt = datetime.now(timezone.utc)
        days = max((end_dt - now_dt).total_seconds() / 86400, 0)
        confidence = "high" if days <= 7 else ("medium" if days <= 30 else "low")

        return {
            "estimated_prob": round(prob, 3),
            "source": "FRED / Fed Analysis",
            "analysis": analysis,
            "confidence": confidence,
            "data_point": f"Rate: {rate_str}",
        }

    # --- CPI / Inflation markets ---
    if _INFLATION_RE.search(question):
        cpi_yoy = _fetch_fred_series("CPALTT01USM657N")
        m_thresh = _MACRO_THRESHOLD_RE.search(question)
        threshold = float(m_thresh.group(1)) if m_thresh else None

        if cpi_yoy is not None and threshold is not None:
            is_below = any(w in q_lower for w in
                           ["below", "under", "lower", "less", "fall"])
            margin = abs(cpi_yoy - threshold)
            if is_below:
                prob = 1.0 / (1.0 + math.exp(-(threshold - cpi_yoy) / 0.5))
            else:
                prob = 1.0 / (1.0 + math.exp(-(cpi_yoy - threshold) / 0.5))
            direction = "below" if is_below else "above"
            analysis = (f"CPI YoY @ {cpi_yoy:.1f}%, target {direction} "
                        f"{threshold}% ({margin:.1f}pp away). "
                        f"Sigmoid P={prob*100:.0f}%.")
            data_point = f"CPI YoY: {cpi_yoy:.1f}%"
        elif cpi_yoy is not None:
            prob = 0.50
            analysis = f"CPI YoY @ {cpi_yoy:.1f}%. No threshold detected."
            data_point = f"CPI YoY: {cpi_yoy:.1f}%"
        else:
            return None  # No data, can't analyze

        now_dt = datetime.now(timezone.utc)
        days = max((end_dt - now_dt).total_seconds() / 86400, 0)
        confidence = "high" if days <= 7 else ("medium" if days <= 30 else "low")

        return {
            "estimated_prob": round(prob, 3),
            "source": "FRED",
            "analysis": analysis,
            "confidence": confidence,
            "data_point": data_point,
        }

    # --- Unemployment markets ---
    if _UNEMPLOYMENT_RE.search(question):
        unemp = _fetch_fred_series("UNRATE")
        m_thresh = _MACRO_THRESHOLD_RE.search(question)
        threshold = float(m_thresh.group(1)) if m_thresh else None

        if unemp is not None and threshold is not None:
            is_below = any(w in q_lower for w in
                           ["below", "under", "lower", "less", "fall"])
            if is_below:
                prob = 1.0 / (1.0 + math.exp(-(threshold - unemp) / 0.3))
            else:
                prob = 1.0 / (1.0 + math.exp(-(unemp - threshold) / 0.3))
            direction = "below" if is_below else "above"
            analysis = (f"Unemployment @ {unemp:.1f}%, target {direction} "
                        f"{threshold}%. Sigmoid P={prob*100:.0f}%.")
            data_point = f"Unemployment: {unemp:.1f}%"
        elif unemp is not None:
            prob = 0.50
            analysis = f"Unemployment @ {unemp:.1f}%. No threshold detected."
            data_point = f"Unemployment: {unemp:.1f}%"
        else:
            return None

        now_dt = datetime.now(timezone.utc)
        days = max((end_dt - now_dt).total_seconds() / 86400, 0)
        confidence = "high" if days <= 7 else ("medium" if days <= 30 else "low")

        return {
            "estimated_prob": round(prob, 3),
            "source": "FRED",
            "analysis": analysis,
            "confidence": confidence,
            "data_point": data_point,
        }

    # --- GDP / Recession markets ---
    if _GDP_RE.search(question):
        gdp = _fetch_fred_series("A191RL1Q225SBEA")

        # Recession = 2 consecutive quarters of negative GDP
        is_recession_q = "recession" in q_lower

        if gdp is not None:
            if is_recession_q:
                # Recession probability based on current GDP growth
                if gdp < 0:
                    prob = 0.55  # already contracting
                elif gdp < 1.0:
                    prob = 0.30  # slow growth
                else:
                    prob = 0.15  # healthy growth
                analysis = (f"GDP growth @ {gdp:.1f}%. "
                            f"Recession P={prob*100:.0f}%.")
            else:
                m_thresh = _MACRO_THRESHOLD_RE.search(question)
                threshold = float(m_thresh.group(1)) if m_thresh else None
                if threshold is not None:
                    is_below = any(w in q_lower for w in
                                   ["below", "under", "lower", "less"])
                    if is_below:
                        prob = 1.0 / (1.0 + math.exp(-(threshold - gdp) / 0.5))
                    else:
                        prob = 1.0 / (1.0 + math.exp(-(gdp - threshold) / 0.5))
                    direction = "below" if is_below else "above"
                    analysis = (f"GDP @ {gdp:.1f}%, target {direction} "
                                f"{threshold}%. P={prob*100:.0f}%.")
                else:
                    prob = 0.50
                    analysis = f"GDP growth @ {gdp:.1f}%. No threshold detected."
            data_point = f"GDP: {gdp:.1f}%"
        else:
            if is_recession_q:
                prob = 0.20  # base rate
                analysis = "GDP data unavailable. Using base recession rate."
                data_point = "GDP: N/A"
            else:
                return None

        now_dt = datetime.now(timezone.utc)
        days = max((end_dt - now_dt).total_seconds() / 86400, 0)
        confidence = "medium" if days <= 30 else "low"

        return {
            "estimated_prob": round(prob, 3),
            "source": "FRED",
            "analysis": analysis,
            "confidence": confidence,
            "data_point": data_point,
        }

    return None


# ---------------------------------------------------------------------------
# Crypto Price Analyzer — CoinGecko API
# ---------------------------------------------------------------------------
# Detects crypto price target markets and compares to current market price.
# Uses CoinGecko free API — no key needed.
# Only activates for markets that pass crypto detection but have price targets.

COINGECKO_API = "https://api.coingecko.com/api/v3"
CRYPTO_CACHE_TTL = 300  # 5 min
_crypto_cache: dict[str, tuple[float, object]] = {}
_crypto_lock = threading.Lock()

CRYPTO_TICKERS: dict[str, str] = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "cardano": "cardano", "ada": "cardano",
    "ripple": "ripple", "xrp": "ripple",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "polkadot": "polkadot", "dot": "polkadot",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
    "chainlink": "chainlink", "link": "chainlink",
    "litecoin": "litecoin", "ltc": "litecoin",
    "polygon": "matic-network", "matic": "matic-network",
    "shiba inu": "shiba-inu", "shib": "shiba-inu",
    "tron": "tron", "trx": "tron",
    "uniswap": "uniswap", "uni": "uniswap",
    "near protocol": "near", "near": "near",
    "sui": "sui",
    "aptos": "aptos", "apt": "aptos",
    "arbitrum": "arbitrum", "arb": "arbitrum",
    "optimism": "optimism", "op": "optimism",
    "pepe": "pepe",
}

_CRYPTO_PRICE_RE = re.compile(
    r"\b(?:price|reach|hit|close|above|below|exceed|surpass|"
    r"worth|trade|trading)\b",
    re.IGNORECASE,
)


def _find_crypto_ticker(question: str) -> str | None:
    """Find CoinGecko coin ID from question. Longest match wins."""
    q_lower = question.lower()
    best = None
    best_len = 0
    for name, coin_id in CRYPTO_TICKERS.items():
        if name in q_lower and len(name) > best_len:
            best = coin_id
            best_len = len(name)
    return best


_cg_last_request_ts = 0.0
_cg_rate_lock = threading.Lock()


def _cg_rate_limit():
    """Enforce minimum interval between CoinGecko requests."""
    global _cg_last_request_ts
    with _cg_rate_lock:
        now = _time.time()
        wait = COINGECKO_MIN_INTERVAL - (now - _cg_last_request_ts)
        if wait > 0:
            _time.sleep(wait)
        _cg_last_request_ts = _time.time()


def _fetch_crypto_price(coin_id: str) -> dict | None:
    """Fetch current price + 30d change from CoinGecko. Cache 5 min."""
    now_ts = _time.time()
    cache_key = f"crypto_{coin_id}"
    with _crypto_lock:
        expired = [k for k, (ts, _) in _crypto_cache.items()
                   if now_ts - ts >= CRYPTO_CACHE_TTL]
        for k in expired:
            del _crypto_cache[k]
        if cache_key in _crypto_cache:
            ts, data = _crypto_cache[cache_key]
            if now_ts - ts < CRYPTO_CACHE_TTL:
                return data

    try:
        _cg_rate_limit()
        resp = requests.get(
            f"{COINGECKO_API}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        market_data = data.get("market_data", {})
        current = market_data.get("current_price", {}).get("usd")
        ath = market_data.get("ath", {}).get("usd")
        change_30d = market_data.get("price_change_percentage_30d", 0)
        vol_30d = abs(change_30d) / 100 if change_30d else 0.30  # rough monthly vol

        if not current or current <= 0:
            return None

        result = {
            "price": current,
            "ath": ath,
            "change_30d": change_30d,
            "monthly_vol": vol_30d,
            "coin_id": coin_id,
        }
        with _crypto_lock:
            _crypto_cache[cache_key] = (_time.time(), result)
        return result
    except Exception as exc:
        _logger.warning("CoinGecko error for %s: %s", coin_id, exc)
        return None


def analyze_crypto(question: str, end_dt: datetime) -> dict | None:
    """Analyze crypto price market using CoinGecko data + log-normal model.

    Similar to finance analyzer but for crypto assets.
    Uses higher volatility assumptions (crypto-appropriate).
    No API key required.
    """
    if not _CRYPTO_PRICE_RE.search(question):
        return None

    coin_id = _find_crypto_ticker(question)
    if not coin_id:
        return None

    # Extract price threshold
    m = _PRICE_THRESHOLD_RE.search(question)
    if not m:
        return None
    raw_val = m.group(1) or m.group(2)
    threshold = float(raw_val.replace(",", ""))
    if threshold <= 0:
        return None

    data = _fetch_crypto_price(coin_id)
    if not data:
        return None

    current = data["price"]
    monthly_vol = data["monthly_vol"]
    # Convert monthly vol to daily (assuming ~30 trading days)
    daily_vol = monthly_vol / math.sqrt(30) if monthly_vol > 0 else 0.05

    now_dt = datetime.now(timezone.utc)
    days = max((end_dt - now_dt).total_seconds() / 86400, 0.1)

    # Log-normal model (same as finance but with higher vol)
    sigma_period = daily_vol * math.sqrt(days)
    # No drift for crypto (no risk-free rate equivalent)
    if sigma_period < 0.001:
        prob_above = 1.0 if current >= threshold else 0.0
    else:
        z = math.log(threshold / current) / sigma_period
        prob_above = 1.0 - _normal_cdf(z)

    q_lower = question.lower()
    is_below = any(w in q_lower for w in
                   ["below", "under", "fall", "drop", "decline", "crash",
                    "sink", "less than", "lower than"])
    prob = (1.0 - prob_above) if is_below else prob_above

    if days <= 7:
        confidence = "medium"  # crypto always lower confidence
    elif days <= 30:
        confidence = "low"
    else:
        confidence = "low"

    pct_away = abs(threshold - current) / current * 100
    direction = "below" if is_below else "above"
    coin_name = coin_id.replace("-", " ").title()

    return {
        "estimated_prob": round(prob, 3),
        "source": "CoinGecko",
        "analysis": (f"{coin_name} @ ${current:,.2f}, target {direction} "
                     f"${threshold:,.2f} ({pct_away:.1f}% away, {days:.0f}d, "
                     f"vol ~{daily_vol*100:.1f}%/d). P={prob*100:.0f}%."),
        "confidence": confidence,
        "data_point": f"${current:,.2f} ({coin_name})",
    }


# ---------------------------------------------------------------------------
# Elections/Polls Analyzer — polling aggregator edge detection
# ---------------------------------------------------------------------------
# Detects election/political markets and compares to RealClearPolitics/538 data.
# Uses RCP RSS feed for polling averages — no API key needed.

POLLS_CACHE_TTL = 1800  # 30 min (polls update slowly)
_polls_cache: dict[str, tuple[float, object]] = {}
_polls_lock = threading.Lock()

_ELECTION_RE = re.compile(
    r"\b(?:elections?|elected|nominees?|primari(?:y|es)|caucus|ballot|"
    r"presidential|gubernatorial|midterms?|runoff|"
    r"democrats?(?:ic)?|republicans?|gop|"
    r"senate|house\s*(?:of\s*representatives?|race|seat|control)|"
    r"congress(?:ional)?|governor(?:'?s)?)\b",
    re.IGNORECASE,
)

_APPROVAL_RE = re.compile(
    r"\b(?:approval|favorab(?:le|ility)|job\s*approval|"
    r"approval\s*rating|disapproval)\b",
    re.IGNORECASE,
)

# Mapping of common political figures / races for RCP scraping
_POLITICAL_FIGURES: dict[str, dict] = {
    "trump": {"name": "Donald Trump", "party": "R"},
    "biden": {"name": "Joe Biden", "party": "D"},
    "harris": {"name": "Kamala Harris", "party": "D"},
    "desantis": {"name": "Ron DeSantis", "party": "R"},
    "newsom": {"name": "Gavin Newsom", "party": "D"},
    "haley": {"name": "Nikki Haley", "party": "R"},
    "vance": {"name": "JD Vance", "party": "R"},
    "buttigieg": {"name": "Pete Buttigieg", "party": "D"},
    "shapiro": {"name": "Josh Shapiro", "party": "D"},
    "whitmer": {"name": "Gretchen Whitmer", "party": "D"},
}

# Baseline probabilities for common election patterns
_INCUMBENT_ADVANTAGE = 0.55  # incumbents win ~55% of the time
_PARTY_CONTROL: dict[str, float] = {
    "senate_dem": 0.45,  # baseline Dem Senate control
    "senate_rep": 0.55,
    "house_dem": 0.50,
    "house_rep": 0.50,
}


def _find_political_figure(question: str) -> dict | None:
    """Find political figure in question. Returns info dict or None."""
    q_lower = question.lower()
    for key, info in _POLITICAL_FIGURES.items():
        if key in q_lower:
            return info
    return None


def _fetch_approval_data(figure_name: str) -> dict | None:
    """Fetch approval rating from RCP-style polling aggregator.

    Uses a public JSON endpoint. Cache 30 min.
    """
    now_ts = _time.time()
    cache_key = f"approval_{figure_name}"
    with _polls_lock:
        expired = [k for k, (ts, _) in _polls_cache.items()
                   if now_ts - ts >= POLLS_CACHE_TTL]
        for k in expired:
            del _polls_cache[k]
        if cache_key in _polls_cache:
            ts, data = _polls_cache[cache_key]
            if now_ts - ts < POLLS_CACHE_TTL:
                return data

    # Try RealClearPolitics JSON endpoint
    try:
        # RCP public polling average page
        slug = figure_name.lower().replace(" ", "_")
        resp = requests.get(
            f"https://www.realclearpolling.com/api/polls/approval/{slug}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Extract latest average
            if isinstance(data, dict):
                approve = data.get("approve") or data.get("average")
                disapprove = data.get("disapprove")
                if approve:
                    result = {
                        "approve": float(approve),
                        "disapprove": float(disapprove) if disapprove else None,
                        "source": "RealClearPolitics",
                    }
                    with _polls_lock:
                        _polls_cache[cache_key] = (_time.time(), result)
                    return result
    except Exception as exc:
        _logger.warning("RCP approval fetch error for %s: %s", figure_name, exc)

    return None


def analyze_elections(question: str, end_dt: datetime) -> dict | None:
    """Analyze election/political market using polling data + base rates.

    Detects: election outcomes, approval ratings, party control markets.
    Uses RealClearPolitics when available, falls back to base rates.
    No API key required.
    """
    q_lower = question.lower()

    # --- Approval rating markets ---
    if _APPROVAL_RE.search(question):
        figure = _find_political_figure(question)
        if not figure:
            return None

        approval_data = _fetch_approval_data(figure["name"])
        m_thresh = _MACRO_THRESHOLD_RE.search(question)
        threshold = float(m_thresh.group(1)) if m_thresh else None

        if approval_data and threshold:
            approve = approval_data["approve"]
            is_below = any(w in q_lower for w in
                           ["below", "under", "lower", "less", "fall"])
            margin = approve - threshold
            if is_below:
                prob = 1.0 / (1.0 + math.exp(-(threshold - approve) / 2.0))
            else:
                prob = 1.0 / (1.0 + math.exp(-(approve - threshold) / 2.0))
            direction = "below" if is_below else "above"
            analysis = (f"{figure['name']} approval @ {approve:.1f}%, "
                        f"target {direction} {threshold}%. "
                        f"P={prob*100:.0f}%.")
            data_point = f"Approval: {approve:.1f}%"
            confidence = "medium"
        elif approval_data:
            approve = approval_data["approve"]
            prob = 0.50
            analysis = f"{figure['name']} approval @ {approve:.1f}%."
            data_point = f"Approval: {approve:.1f}%"
            confidence = "low"
        else:
            return None

        return {
            "estimated_prob": round(prob, 3),
            "source": "RealClearPolitics",
            "analysis": analysis,
            "confidence": confidence,
            "data_point": data_point,
        }

    # --- Election outcome markets ---
    if _ELECTION_RE.search(question):
        figure = _find_political_figure(question)

        # Party control markets
        is_senate = "senate" in q_lower
        is_house = "house" in q_lower
        is_dem = any(w in q_lower for w in ["democrat", "dem ", "democratic"])
        is_rep = any(w in q_lower for w in ["republican", "gop", "rep "])

        if (is_senate or is_house) and (is_dem or is_rep):
            chamber = "senate" if is_senate else "house"
            party = "dem" if is_dem else "rep"
            key = f"{chamber}_{party}"
            prob = _PARTY_CONTROL.get(key, 0.50)
            analysis = (f"{chamber.title()} {party.upper()} control: "
                        f"base rate {prob*100:.0f}%.")
            data_point = f"{chamber.title()} control base rate"

            now_dt = datetime.now(timezone.utc)
            days = max((end_dt - now_dt).total_seconds() / 86400, 0)
            confidence = "low"  # pure base rate without polling data

            return {
                "estimated_prob": round(prob, 3),
                "source": "Polling Analysis",
                "analysis": analysis,
                "confidence": confidence,
                "data_point": data_point,
            }

        # Specific candidate markets — use generic base rates
        if figure:
            is_incumbent = any(w in q_lower for w in
                               ["re-elect", "reelect", "re-election"])
            prob = _INCUMBENT_ADVANTAGE if is_incumbent else 0.50
            analysis = (f"{figure['name']} ({figure['party']}): "
                        f"base rate analysis. Incumbent advantage "
                        f"{'applies' if is_incumbent else 'N/A'}.")
            data_point = f"{figure['name']} ({figure['party']})"

            now_dt = datetime.now(timezone.utc)
            days = max((end_dt - now_dt).total_seconds() / 86400, 0)
            confidence = "low"

            return {
                "estimated_prob": round(prob, 3),
                "source": "Polling Analysis",
                "analysis": analysis,
                "confidence": confidence,
                "data_point": data_point,
            }

    return None


# ---------------------------------------------------------------------------
# Analyzer health tracking
# ---------------------------------------------------------------------------

_analyzer_health: dict[str, dict] = {}
_health_lock = threading.Lock()

_ANALYZER_NAMES = [
    "weather", "finance", "earnings", "sports",
    "fed_macro", "elections", "crypto",
]


def _record_health(name: str, success: bool, detail: str = ""):
    with _health_lock:
        _analyzer_health[name] = {
            "status": "ok" if success else "error",
            "last_check": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        }


# ---------------------------------------------------------------------------
# CLOB API — real spread verification
# ---------------------------------------------------------------------------

_clob_cache: dict[str, tuple[float, dict]] = {}
_clob_lock = threading.Lock()


def _fetch_order_book(token_id: str) -> dict | None:
    """Fetch order book from Polymarket CLOB."""
    now_ts = _time.time()
    with _clob_lock:
        if token_id in _clob_cache:
            ts, data = _clob_cache[token_id]
            if now_ts - ts < CLOB_CACHE_TTL:
                return data

    try:
        resp = requests.get(
            f"{CLOB_API_BASE}/book",
            params={"token_id": token_id},
            timeout=5,
        )
        resp.raise_for_status()
        book = resp.json()

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        spread = best_ask - best_bid

        result = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 4),
            "spread_pct": round(spread / best_ask * 100, 2) if best_ask > 0 else 0,
            "bid_depth": round(bid_depth, 2),
            "ask_depth": round(ask_depth, 2),
        }
        with _clob_lock:
            _clob_cache[token_id] = (_time.time(), result)
        return result
    except Exception:
        return None


def verify_arb_execution(market: dict, prices: list[float]) -> dict:
    """Check CLOB spreads for arb execution feasibility.

    Returns {executable, total_spread_cost_pct, effective_arb_pct, books}.
    """
    clob_ids_raw = market.get("clobTokenIds", "[]")
    try:
        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
    except (json.JSONDecodeError, TypeError):
        return {"executable": None, "total_spread_cost_pct": 0,
                "effective_arb_pct": 0, "books": []}

    if not clob_ids or not isinstance(clob_ids, list):
        return {"executable": None, "total_spread_cost_pct": 0,
                "effective_arb_pct": 0, "books": []}

    books = []
    total_spread_cost = 0.0
    for i, tid in enumerate(clob_ids):
        book = _fetch_order_book(tid)
        if book:
            books.append(book)
            # For arb: we buy at ask price, so spread is the cost
            total_spread_cost += book["spread"]
        else:
            books.append(None)

    arb_deviation = abs(sum(prices) - 1.0)
    spread_cost_pct = total_spread_cost * 100  # as percentage of $1
    effective_arb = (arb_deviation * 100) - spread_cost_pct

    return {
        "executable": effective_arb > 0 if books else None,
        "total_spread_cost_pct": round(spread_cost_pct, 2),
        "effective_arb_pct": round(effective_arb, 2),
        "books": books,
    }


# ---------------------------------------------------------------------------
# Analyzer orchestrator — multi-analyzer with Bayesian fusion
# ---------------------------------------------------------------------------

def run_analyzers(question: str, end_dt: datetime) -> dict | None:
    """Run all analyzers, return best result by confidence-weighted edge.

    If multiple analyzers match, picks the one with highest confidence.
    Order: weather → finance → earnings → sports → fed/macro → elections → crypto.
    """
    _analyzers = [
        ("weather", analyze_weather),
        ("finance", analyze_finance),
        ("earnings", analyze_earnings),
        ("sports", analyze_sports),
        ("fed_macro", analyze_fed_macro),
        ("elections", analyze_elections),
        ("crypto", analyze_crypto),
    ]

    results = []
    for name, fn in _analyzers:
        try:
            r = fn(question, end_dt)
            if r:
                _record_health(name, True, r.get("source", ""))
                results.append((name, r))
            # Don't record "no match" as error — only actual failures
        except Exception as exc:
            _record_health(name, False, str(exc))

    if not results:
        return None

    if len(results) == 1:
        return results[0][1]

    # Multiple analyzers matched — pick best by confidence then edge magnitude
    conf_rank = {"high": 3, "medium": 2, "low": 1}
    results.sort(key=lambda x: (
        conf_rank.get(x[1].get("confidence", "low"), 0),
        abs(x[1].get("estimated_prob", 0.5) - 0.5),  # extremity of prediction
    ), reverse=True)

    # Return best, but note all sources in analysis
    best = results[0][1]
    if len(results) > 1:
        other_sources = [r[1].get("source", r[0]) for r in results[1:]]
        best["also_analyzed_by"] = other_sources
    return best


# ---------------------------------------------------------------------------
# Kelly Criterion — optimal position sizing
# ---------------------------------------------------------------------------

def kelly_fraction(win_prob: float, buy_price: float) -> float:
    """Calculate Kelly criterion fraction for optimal bet sizing.

    Kelly f* = (bp - q) / b
    where b = net odds (payout/stake - 1), p = win prob, q = 1-p.
    Returns fraction of bankroll to bet (0 to 1), capped at 25%.
    """
    if buy_price <= 0 or buy_price >= 1 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    b = (1.0 / buy_price) - 1.0  # net odds
    q = 1.0 - win_prob
    f = (b * win_prob - q) / b
    # Cap at 25% (full Kelly is too aggressive)
    return max(0.0, min(f, 0.25))


# ---------------------------------------------------------------------------
# Prediction logging — for backtesting
# ---------------------------------------------------------------------------

_predictions_lock = threading.Lock()


def _log_prediction(opp: dict):
    """Append prediction to JSONL file for future backtesting."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": opp.get("id", ""),
        "question": opp.get("question", ""),
        "tier": opp.get("tier", ""),
        "score": opp.get("score", 0),
        "market_state": {
            "yes_price": opp.get("yes", 0),
            "no_price": opp.get("no", 0),
            "price_sum": opp.get("price_sum", 0),
            "liquidity": opp.get("liquidity", 0),
            "volume": opp.get("volume", 0),
        },
        "trade": {
            "label": opp.get("trade_label", ""),
            "side": opp.get("trade_side_class", ""),
            "ev_per_100": opp.get("ev_per_100", 0),
            "net_per_100": opp.get("net_per_100", 0),
            "kelly_pct": opp.get("kelly_pct", 0),
        },
    }
    ea = opp.get("edge_analysis")
    if ea:
        entry["edge"] = {
            "source": ea.get("source", ""),
            "estimated_prob": ea.get("estimated_prob", 0),
            "edge": ea.get("edge", 0),
            "confidence": ea.get("confidence", ""),
        }

    try:
        with _predictions_lock:
            with open(PREDICTIONS_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Webhook notifications — Discord / Telegram
# ---------------------------------------------------------------------------

_last_notified_ids: set[str] = set()


def _send_discord_webhook(message: str):
    """Send notification to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL,
                      json={"content": message},
                      timeout=5)
    except Exception:
        pass


def _send_telegram_message(message: str):
    """Send notification via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                  "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception:
        pass


def _notify_new_opportunities(tiers: dict):
    """Send notifications for new edge/super opportunities."""
    global _last_notified_ids
    new_ids = set()
    messages = []

    for tier_name in ("edge", "super"):
        for opp in tiers.get(tier_name, []):
            oid = opp.get("id", "")
            new_ids.add(oid)
            if oid and oid not in _last_notified_ids:
                emoji = "🧠" if tier_name == "edge" else "✅"
                ev = opp.get("ev_per_100", 0)
                net = opp.get("net_per_100", 0)
                msg = (f"{emoji} *{tier_name.upper()}* | "
                       f"{opp.get('question', '')[:80]}\n"
                       f"EV: +${ev:.2f}/100 | Net: ${net:.2f} | "
                       f"Kelly: {opp.get('kelly_pct', 0):.1f}%\n"
                       f"{opp.get('url', '')}")
                messages.append(msg)

    for msg in messages[:5]:  # max 5 notifications per scan
        _send_discord_webhook(msg)
        _send_telegram_message(msg)

    _last_notified_ids = new_ids


# ---------------------------------------------------------------------------
# Portfolio tracker — local JSON
# ---------------------------------------------------------------------------

_portfolio_lock = threading.Lock()


def _load_portfolio() -> list[dict]:
    """Load portfolio from JSON file."""
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_portfolio(portfolio: list[dict]):
    """Save portfolio to JSON file."""
    with _portfolio_lock:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)


# ---------------------------------------------------------------------------
# Historical scan tracking
# ---------------------------------------------------------------------------

_scan_history: list[dict] = []
_history_lock = threading.Lock()


def _record_scan_history(scan_data: dict):
    """Keep last N scans for historical comparison."""
    summary = {
        "timestamp": scan_data.get("refreshed_at", ""),
        "total_opps": scan_data.get("total_opps", 0),
        "tier_counts": {
            t: len(scan_data.get("tiers", {}).get(t, []))
            for t in ("edge", "super", "interesting", "watch")
        },
        "top_opportunities": [],
    }
    # Store top 5 opportunities across all tiers
    for tier_name in ("edge", "super", "interesting"):
        for opp in scan_data.get("tiers", {}).get(tier_name, [])[:3]:
            summary["top_opportunities"].append({
                "question": opp.get("question", "")[:80],
                "tier": opp.get("tier", ""),
                "score": opp.get("score", 0),
                "ev_per_100": opp.get("ev_per_100", 0),
            })

    with _history_lock:
        _scan_history.append(summary)
        if len(_scan_history) > MAX_SCAN_HISTORY:
            _scan_history.pop(0)


app = Flask(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_cache: dict = {"data": None, "ts": 0.0}

# ---------------------------------------------------------------------------
# API — paginated fetch
# ---------------------------------------------------------------------------

_logger = logging.getLogger("polymarket_scanner")


def fetch_all_markets() -> list[dict]:
    """Fetch up to MAX_PAGES * BATCH_SIZE markets with pagination.

    Retries each page up to 2 times on failure with 1s backoff.
    Logs warnings on partial fetches.
    """
    all_markets: list[dict] = []
    for page in range(MAX_PAGES):
        params = {**API_PARAMS, "limit": BATCH_SIZE, "offset": page * BATCH_SIZE}
        batch = None
        for attempt in range(3):
            try:
                resp = requests.get(GAMMA_API, params=params, timeout=10)
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as exc:
                if attempt < 2:
                    _time.sleep(1)
                    _logger.warning("Gamma API page %d attempt %d failed: %s",
                                    page, attempt + 1, exc)
                else:
                    _logger.error("Gamma API page %d failed after 3 attempts: %s",
                                  page, exc)
        if not batch:
            if all_markets:
                _logger.warning("Gamma API: got %d markets from %d/%d pages",
                                len(all_markets), page, MAX_PAGES)
            break
        all_markets.extend(batch)
        if len(batch) < BATCH_SIZE:
            break
    return all_markets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def parse_date(ds: str | None) -> datetime | None:
    if not ds:
        return None
    ds = ds.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ds)
    except ValueError:
        return None


# is_crypto() defined above near CRYPTO_RE


def truncate(s: str, n: int = 15) -> str:
    return s[:n - 2] + ".." if len(s) > n else s


def extract_category(m: dict) -> str:
    """Best-effort category from API tags or question heuristics."""
    tags = m.get("tags")
    if tags:
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        if isinstance(tags, list) and tags:
            first = tags[0]
            lbl = first.get("label", first) if isinstance(first, dict) else str(first)
            if lbl:
                return lbl.strip().title()

    q = (m.get("question", "") or "").lower()
    rules = [
        ("Politics", ["trump", "biden", "election", "congress", "senate",
                       "governor", "president", "democrat", "republican",
                       "vote", "poll", "white house", "legislation", "impeach",
                       "supreme court", "scotus", "parliament", "prime minister",
                       "midterm", "primary", "caucus", "nominee"]),
        ("Sports", ["nfl", "nba", "mlb", "nhl", "soccer", "football",
                     "basketball", "baseball", "tennis", "ufc", "boxing",
                     "super bowl", "world cup", "championship", "playoffs",
                     "grand slam", "formula 1", "f1 ", "olympics",
                     "cricket", "rugby", "esports", "golf", "mma"]),
        ("Finance", ["stock", "share", "earnings", "revenue", "eps",
                      "s&p", "nasdaq", "dow jones", "ipo",
                      "acquisition", "merger", "dividend",
                      "close above", "close below", "trading above",
                      "market cap"]),
        ("Economics", ["fed ", "interest rate", "gdp", "inflation",
                       "recession", "cpi", "pce",
                       "jobs report", "unemployment", "treasury", "tariff",
                       "nonfarm", "payroll", "fomc", "rate cut", "rate hike",
                       "government shutdown"]),
        ("Crypto", ["bitcoin", "btc", "ethereum", "solana",
                     "crypto", "dogecoin", "ripple", "xrp",
                     "cardano", "blockchain", "defi", "nft",
                     "token price", "coin price", "altcoin"]),
        ("Commodities", ["gold price", "oil price", "silver price",
                          "crude oil", "natural gas", "copper price",
                          "platinum", "commodity"]),
        ("Forex", ["euro ", "eur/usd", "gbp", "yen", "yuan",
                    "dollar index", "dxy", "forex", "currency",
                    "exchange rate"]),
        ("Tech", [" ai ", "openai", "google", "apple", "microsoft", "tesla",
                  "spacex", "chatgpt", "artificial intelligence", "llm ",
                  "semiconductor", "chip", "data center"]),
        ("Entertainment", ["oscar", "grammy", "emmy", "movie", "album",
                          "celebrity", "tiktok", "youtube", "netflix",
                          "disney", "box office", "billboard",
                          "golden globe", "rotten tomatoes", "tweet"]),
        ("Geopolitics", ["war ", "ukraine", "russia", "china", "nato",
                        "sanction", "missile", "military", "iran", "israel",
                        "ceasefire", "invasion", "north korea",
                        "taiwan", "venezuela", "cuba"]),
        ("Weather", ["weather", "hurricane", "earthquake", "temperature",
                     "climate", "wildfire", "flood", "tornado",
                     "snow", "precipitation"]),
        ("Science", ["fda", "vaccine", "covid", "trial", "study",
                     "nasa", "mars ", "space station", "pandemic",
                     "virus", "clinical trial"]),
    ]
    for cat, keywords in rules:
        if any(kw in q for kw in keywords):
            return cat
    return "Other"

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def classify(max_price: float, deviation: float,
             price_sum: float,
             edge_analysis: dict | None = None) -> tuple[str | None, str, str]:
    """Classify opportunity: (tier, reason_label, reason_type) or (None,'','').

    Philosophy: only REAL edge, no luck.
    - edge (external data diverges from market) → top priority if edge > MIN_EDGE
    - arbitrage (sum < 1.0, >2%) = guaranteed profit → super/interesting
    - near_certain (>90%) = EV ≈ $0, involves luck → watch only
    - overround (sum > 1.0) = excluded (margin against you)
    - arbs <2% excluded (unprofitable after ~2% fees)
    """
    # --- External edge (data-driven) — REAL informational advantage ---
    # Checked FIRST: even very certain markets can have edge (item 6)
    if edge_analysis and abs(edge_analysis.get("edge", 0)) >= MIN_EDGE:
        source = edge_analysis.get("source", "Analyse")
        return ("edge", f"Edge {source}", "edge")

    if max_price > 0.995:
        return (None, "", "")  # Too certain — negligible profit

    # --- Guaranteed arbitrage (sum < 1.0, min 2% to cover fees) ---
    if price_sum < 1.0 and deviation > 0.04:
        return ("super", "Arb garanti", "arbitrage")
    if price_sum < 1.0 and deviation > 0.02:
        return ("interesting", "Petit arb", "arbitrage")

    # --- Near-certain (EV ≈ $0, involves luck) → watch only ---
    if max_price > 0.90:
        return ("watch", "Haute proba", "near_certain")

    return (None, "", "")


_CONFIDENCE_MULT = {"high": 1.0, "medium": 0.7, "low": 0.4}


def compute_score(deviation: float, price_sum: float, max_price: float,
                  days_left: float, liquidity: float,
                  ext_edge: float = 0.0,
                  confidence: str = "high") -> float:
    """Score by attractiveness. Edge > arbs >> near-certain.

    confidence (str): "high"/"medium"/"low" — modulates ext_edge weight (item 16).
    """
    if ext_edge >= MIN_EDGE:
        conf_mult = _CONFIDENCE_MULT.get(confidence, 0.4)
        edge = ext_edge * conf_mult * 40  # external edge weighted by confidence
    elif price_sum < 1.0 and deviation > 0.02:
        edge = deviation * 50          # arb: dominant weight
    else:
        edge = max(max_price - 0.70, 0) * 0.1  # near-certain: minimal weight
    time_f = 1.0 / max(days_left, 0.08)
    liq_f = min(math.log10(max(liquidity, 1)) / 6.0, 1.0)
    return round(edge * time_f * liq_f * 10000, 1)

# ---------------------------------------------------------------------------
# Process one market → opportunity dict or None
# ---------------------------------------------------------------------------

def process_market(m: dict, now: datetime) -> dict | None:
    question = m.get("question", "") or m.get("title", "")
    slug = m.get("slug", "")
    condition_id = m.get("conditionId", "")
    end_date_str = m.get("endDate")
    volume = parse_float(m.get("volume"))
    volume_24h = parse_float(
        m.get("volume24hr") or m.get("volume24Hr") or m.get("volume_24h") or 0
    )
    liquidity = parse_float(m.get("liquidity"))
    outcome_prices_raw = m.get("outcomePrices", "[]")
    outcomes_raw = m.get("outcomes", '["Yes", "No"]')
    category = extract_category(m)

    # --- Date gate ---
    end_dt = parse_date(end_date_str)
    if end_dt is None:
        return None
    delta_s = (end_dt - now).total_seconds()
    if delta_s < 0:
        return None
    days_left = delta_s / 86400.0
    if days_left > MAX_DAYS:
        return None

    # --- Volume / liquidity gate ---
    if volume < MIN_VOLUME or liquidity < MIN_LIQUIDITY:
        return None

    # --- Crypto gate (allows crypto price markets through for analyzer) ---
    if is_crypto(question):
        # Let crypto price markets through if they have a price target
        has_price_target = bool(_PRICE_THRESHOLD_RE.search(question))
        has_crypto_ticker = _find_crypto_ticker(question) is not None
        if not (has_price_target and has_crypto_ticker):
            return None

    # --- Volume 24h gate (skip dead markets when data available) ---
    has_24h = (m.get("volume24hr") is not None or
               m.get("volume24Hr") is not None or
               m.get("volume_24h") is not None)
    if has_24h and volume_24h <= 0:
        return None

    # --- Parse outcomes ---
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except (json.JSONDecodeError, TypeError):
        outcomes = []
    if not isinstance(outcomes, list):
        outcomes = []

    # --- Parse prices ---
    try:
        prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not prices or len(prices) < 2:
        return None

    float_prices = [parse_float(p) for p in prices]
    num_outcomes = len(float_prices)
    is_binary = num_outcomes == 2
    price_sum = sum(float_prices)
    deviation = abs(price_sum - 1.0)

    # --- Best outcome ---
    max_idx = max(range(num_outcomes), key=lambda i: float_prices[i])
    max_price = float_prices[max_idx]
    max_outcome_name = outcomes[max_idx] if max_idx < len(outcomes) else f"Outcome {max_idx + 1}"

    if is_binary:
        yes_price = float_prices[0]
        no_price = float_prices[1]
        yes_label = outcomes[0] if outcomes else "Yes"
        no_label = outcomes[1] if len(outcomes) > 1 else "No"
    else:
        yes_price = max_price
        no_price = round(max(1.0 - max_price, 0), 4)
        yes_label = truncate(max_outcome_name, 14)
        no_label = f"Others ({num_outcomes - 1})"

    # --- External analysis (weather, sports, etc.) ---
    edge_analysis = None
    analysis_result = run_analyzers(question, end_dt)
    if analysis_result:
        est_prob = analysis_result["estimated_prob"]
        # Edge = est_prob vs Yes price (items 1-2: use Yes price, not max_price)
        # For binary: Yes = float_prices[0], for multi: Yes = top outcome
        yes_market_price = float_prices[0] if is_binary else max_price
        raw_edge = est_prob - yes_market_price
        analysis_result["edge"] = round(raw_edge, 4)
        analysis_result["market_price"] = yes_market_price
        edge_analysis = analysis_result

    # --- Classify ---
    tier, reason_label, reason_type = classify(max_price, deviation, price_sum, edge_analysis)
    if tier is None:
        return None

    days_left_int = max(int(days_left), 0)
    if days_left < 1:
        days_display = round(days_left * 24, 1)
        days_unit = "h"
    else:
        days_display = days_left_int
        days_unit = "d"

    # --- Trade recommendation ---
    if reason_type == "arbitrage":
        if num_outcomes == 2:
            trade_label = f"ARB: BUY ALL @ {int(price_sum * 100)}\u00a2"
        else:
            trade_label = f"ARB: BUY {num_outcomes}\u00d7 @ {int(price_sum * 100)}\u00a2"
        trade_side_class = "arb"
        buy_price = price_sum
        is_guaranteed = True
    elif reason_type == "edge":
        edge_val = edge_analysis["edge"] if edge_analysis else 0
        # Items 1-2: edge is est_prob - yes_price
        # Positive edge → Yes is underpriced → BUY YES
        # Negative edge → Yes is overpriced → BUY NO
        if edge_val >= 0:
            # Buy Yes (outcome 0 for binary, top outcome for multi)
            if is_binary:
                buy_name = outcomes[0] if outcomes else "Yes"
                buy_price = float_prices[0]
                trade_side_class = "yes"
            else:
                buy_name = max_outcome_name
                buy_price = max_price
                trade_side_class = "yes"
            trade_label = f"BUY {truncate(buy_name.upper(), 12)} @ {int(buy_price * 100)}\u00a2"
        else:
            # Buy No (outcome 1 for binary, cheapest for multi)
            if is_binary:
                buy_name = outcomes[1] if len(outcomes) > 1 else "No"
                buy_price = float_prices[1]
                trade_side_class = "no"
            else:
                other_idx = min((i for i in range(num_outcomes) if i != max_idx),
                                key=lambda i: float_prices[i])
                buy_name = outcomes[other_idx] if other_idx < len(outcomes) else f"Out.{other_idx+1}"
                buy_price = float_prices[other_idx]
                trade_side_class = "no"
            trade_label = f"BUY {truncate(buy_name.upper(), 12)} @ {int(buy_price * 100)}\u00a2"
        is_guaranteed = False
    else:
        trade_label = f"BUY {truncate(max_outcome_name.upper(), 12)} @ {int(max_price * 100)}\u00a2"
        trade_side_class = "yes" if (max_idx == 0 or not is_binary) else "no"
        buy_price = max_price
        is_guaranteed = False

    # --- Profit, ROI, EV, Fees ---
    profit_100 = round(100 * (1.0 / buy_price - 1.0), 2) if buy_price > 0 else 0
    roi_pct = (1.0 / buy_price - 1.0) * 100 if buy_price > 0 else 0
    ann_roi = min(round(roi_pct * (365.0 / max(days_left, 0.04)), 1), MAX_ANN_ROI)

    if reason_type == "arbitrage":
        ev_per_100 = round((1.0 - price_sum) / price_sum * 100, 2) if price_sum > 0 else 0
        risk_per_100 = 0.0
        # Fees scale with number of trades required
        fee_per_100 = round(EST_FEE_PCT * num_outcomes / 2.0, 2)
        net_per_100 = round(ev_per_100 - fee_per_100, 2)
        arb_warning = f"{num_outcomes} trades requis"
    elif reason_type == "edge" and edge_analysis:
        # Item 11: correct EV = (win_prob / buy_price - 1) × 100
        # win_prob depends on which side we buy:
        #   positive edge → buy Yes → win_prob = est_prob
        #   negative edge → buy No  → win_prob = 1 - est_prob
        est_prob = edge_analysis.get("estimated_prob", 0)
        edge_val = edge_analysis.get("edge", 0)
        win_prob = est_prob if edge_val >= 0 else (1.0 - est_prob)
        ev_per_100 = round((win_prob / buy_price - 1) * 100, 2) if buy_price > 0 else 0.0
        risk_per_100 = 100.0  # you can still lose
        fee_per_100 = round(EST_FEE_PCT, 2)
        net_per_100 = round(ev_per_100 - fee_per_100, 2)
        arb_warning = ""
    else:
        ev_per_100 = 0.0
        risk_per_100 = 100.0
        fee_per_100 = 0.0
        net_per_100 = 0.0
        arb_warning = ""

    # --- Vol/Liq ratio ---
    vol_liq = round(volume / liquidity, 1) if liquidity > 0 else 999
    thin = liquidity < 10000 or vol_liq > 20

    # --- Spread/slippage warning for arbs ---
    spread_warning = ""
    if reason_type == "arbitrage":
        liq_per_outcome = liquidity / num_outcomes if num_outcomes > 0 else liquidity
        if liq_per_outcome < 5000:
            spread_warning = (f"~${liq_per_outcome:,.0f}/outcome — "
                              f"liquidité insuffisante, spread probable")
        elif liquidity < 25000:
            spread_warning = "Faible liquidité — spread peut annuler le gain"
        elif num_outcomes > 2 and liq_per_outcome < 15000:
            spread_warning = (f"Multi-outcome ({num_outcomes}), "
                              f"~${liq_per_outcome:,.0f}/outcome — slippage probable")

    # --- CLOB spread verification for arbs ---
    clob_data = None
    if reason_type == "arbitrage":
        clob_data = verify_arb_execution(m, float_prices)
        if clob_data.get("executable") is False:
            spread_warning = (f"CLOB: spread cost {clob_data['total_spread_cost_pct']:.1f}% "
                              f"annule l'arb ({clob_data['effective_arb_pct']:.1f}% net)")
        elif clob_data.get("executable") is True:
            effective = clob_data["effective_arb_pct"]
            if effective < 1.0:
                spread_warning = (f"CLOB: arb net ~{effective:.1f}% après spread — "
                                  f"marge très faible")

    # --- Kelly criterion ---
    kelly_pct = 0.0
    if reason_type == "edge" and edge_analysis:
        est_prob = edge_analysis.get("estimated_prob", 0)
        edge_val = edge_analysis.get("edge", 0)
        win_prob = est_prob if edge_val >= 0 else (1.0 - est_prob)
        kelly_pct = round(kelly_fraction(win_prob, buy_price) * 100, 1)
    elif reason_type == "arbitrage" and price_sum > 0:
        # For arbs, Kelly is effectively 100% (guaranteed) but cap display
        kelly_pct = 25.0  # max display for guaranteed arbs

    # --- Composite score ---
    ext_edge = abs(edge_analysis["edge"]) if edge_analysis else 0.0
    edge_conf = edge_analysis.get("confidence", "high") if edge_analysis else "high"
    score = compute_score(deviation, price_sum, max_price, days_left, liquidity,
                          ext_edge, confidence=edge_conf)

    return {
        "tier": tier,
        "id": condition_id or slug or question[:40],
        "question": question,
        "yes": yes_price,
        "no": no_price,
        "yes_label": yes_label,
        "no_label": no_label,
        "max_price": max_price,
        "price_sum": round(price_sum, 4),
        "volume": volume,
        "volume_24h": volume_24h,
        "liquidity": liquidity,
        "vol_liq": vol_liq,
        "thin": thin,
        "days_left": days_left_int,
        "days_display": days_display,
        "days_unit": days_unit,
        "reason": reason_label,
        "reason_type": reason_type,
        "trade_label": trade_label,
        "trade_side_class": trade_side_class,
        "profit_100": profit_100,
        "ann_roi": ann_roi,
        "ann_roi_capped": ann_roi >= MAX_ANN_ROI,
        "guaranteed": is_guaranteed,
        "ev_per_100": ev_per_100,
        "risk_per_100": risk_per_100,
        "fee_per_100": fee_per_100,
        "net_per_100": net_per_100,
        "arb_warning": arb_warning,
        "spread_warning": spread_warning,
        "edge_analysis": edge_analysis,
        "clob_data": clob_data,
        "kelly_pct": kelly_pct,
        "score": score,
        "category": category,
        "num_outcomes": num_outcomes,
        "is_binary": is_binary,
        "url": f"https://polymarket.com/event/{slug}" if slug else "",
    }

# ---------------------------------------------------------------------------
# Scan — orchestrate
# ---------------------------------------------------------------------------

def scan() -> dict:
    now = datetime.now(timezone.utc)
    try:
        raw = fetch_all_markets()
    except Exception as exc:
        return {
            "error": str(exc),
            "fetched": 0,
            "qualified": 0,
            "refreshed_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "categories": [],
            "tiers": {"edge": [], "super": [], "interesting": [], "watch": []},
        }

    fetched = len(raw)
    tiers: dict[str, list] = {"edge": [], "super": [], "interesting": [], "watch": []}
    cats: set[str] = set()
    seen: set[str] = set()
    qualified = 0

    for m in raw:
        mid = m.get("conditionId") or m.get("slug") or ""
        if mid:
            if mid in seen:
                continue
            seen.add(mid)

        opp = process_market(m, now)
        if opp is None:
            continue
        qualified += 1
        cats.add(opp["category"])
        tiers[opp["tier"]].append(opp)

    # Sort each tier by composite score (best first)
    for t in tiers:
        tiers[t].sort(key=lambda x: -x["score"])

    total_opps = sum(len(v) for v in tiers.values())

    # --- Log predictions for backtesting ---
    for tier_name in ("edge", "super", "interesting"):
        for opp in tiers[tier_name]:
            _log_prediction(opp)

    # --- Build result ---
    result = {
        "error": None,
        "fetched": fetched,
        "qualified": qualified,
        "total_opps": total_opps,
        "refreshed_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "categories": sorted(cats),
        "tiers": tiers,
    }

    # --- Record scan history ---
    _record_scan_history(result)

    # --- Webhook notifications for new edge/super ---
    _notify_new_opportunities(tiers)

    # --- Analyzer health summary ---
    with _health_lock:
        result["analyzer_health"] = dict(_analyzer_health)

    return result


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polymarket Scanner</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ==========================================================================
   DESIGN TOKENS
   ========================================================================== */
:root {
  --pm-bg-page:       #12151f;
  --pm-bg-surface:    #181b28;
  --pm-bg-card:       #1c2030;
  --pm-bg-card-hover: #222639;
  --pm-bg-elevated:   #262a3d;
  --pm-border:        #262a3b;
  --pm-border-light:  #2e3348;
  --pm-text-primary:  #f0f2f5;
  --pm-text-secondary:#858d9d;
  --pm-text-tertiary: #5d6577;
  --pm-blue:          #2e5cff;
  --pm-blue-hover:    #4d75ff;
  --pm-blue-pressed:  #2249d6;
  --pm-blue-bg:       rgba(46,92,255,0.10);
  --pm-green:         #47c97a;
  --pm-green-soft:    rgba(71,201,122,0.12);
  --pm-green-mid:     rgba(71,201,122,0.22);
  --pm-red:           #ff6464;
  --pm-red-soft:      rgba(255,100,100,0.10);
  --pm-red-mid:       rgba(255,100,100,0.20);
  --pm-orange:        #ff9332;
  --pm-orange-soft:   rgba(255,147,50,0.12);
  --pm-purple:        #9b6dff;
  --pm-purple-soft:   rgba(155,109,255,0.12);
  --pm-cyan:          #3dc2ec;
  --pm-cyan-soft:     rgba(61,194,236,0.10);
  --tier-super:       #ff9332;
  --tier-super-bg:    rgba(255,147,50,0.08);
  --tier-int:         #9b6dff;
  --tier-int-bg:      rgba(155,109,255,0.06);
  --tier-edge:        #00e676;
  --tier-edge-bg:     rgba(0,230,118,0.08);
  --tier-watch:       #3dc2ec;
  --tier-watch-bg:    rgba(61,194,236,0.05);
  --r-xs: 4px; --r-sm: 6px; --r-md: 8px; --r-lg: 12px; --r-full: 100px;
  --font: "Open Sauce One","Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}

/* ==========================================================================
   RESET & BASE
   ========================================================================== */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
html{font-size:14px;-webkit-font-smoothing:antialiased}
body{font-family:var(--font);background:var(--pm-bg-page);color:var(--pm-text-primary);line-height:1.5;min-height:100vh}
a{color:inherit;text-decoration:none}
button{font-family:var(--font);cursor:pointer}

/* ==========================================================================
   LAYOUT
   ========================================================================== */
.shell{max-width:1280px;margin:0 auto;padding:0 24px}

/* ==========================================================================
   TOPBAR
   ========================================================================== */
.topbar{position:sticky;top:0;z-index:100;background:rgba(18,21,31,0.85);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid var(--pm-border)}
.topbar-in{max-width:1280px;margin:0 auto;padding:0 24px;height:56px;display:flex;align-items:center;justify-content:space-between;gap:12px}
.topbar-brand{display:flex;align-items:center;gap:10px;flex-shrink:0}
.topbar-logo{width:28px;height:28px;background:var(--pm-blue);border-radius:var(--r-sm);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:15px}
.topbar-title{font-size:15px;font-weight:700;letter-spacing:-0.2px}
.topbar-title span{color:var(--pm-text-tertiary);font-weight:400;margin-left:6px}
.topbar-meta{display:flex;align-items:center;gap:16px;font-size:12.5px;color:var(--pm-text-tertiary)}
.topbar-meta strong{color:var(--pm-text-secondary);font-weight:600}
.topbar-opps{background:var(--pm-green-soft);color:var(--pm-green);font-weight:700;padding:2px 10px;border-radius:var(--r-full);font-size:12px}
.topbar-actions{display:flex;align-items:center;gap:8px;flex-shrink:0}

/* ==========================================================================
   BUTTONS
   ========================================================================== */
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border:none;border-radius:var(--r-sm);font-size:13px;font-weight:600;transition:background .15s;white-space:nowrap}
.btn-primary{background:var(--pm-blue);color:#fff}
.btn-primary:hover{background:var(--pm-blue-hover)}
.btn-primary:active{background:var(--pm-blue-pressed)}
.btn-primary:disabled{background:var(--pm-bg-elevated);color:var(--pm-text-tertiary);cursor:not-allowed}
.btn-ghost{background:transparent;color:var(--pm-text-secondary);border:1px solid var(--pm-border)}
.btn-ghost:hover{background:var(--pm-bg-card-hover);color:var(--pm-text-primary);border-color:var(--pm-border-light)}
.btn-icon{padding:6px;width:32px;height:32px;justify-content:center;font-size:15px}

.live-dot{width:6px;height:6px;background:var(--pm-green);border-radius:50%;animation:pulse-d 2s ease-in-out infinite}
@keyframes pulse-d{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(71,201,122,.5)}50%{opacity:.7;box-shadow:0 0 0 4px rgba(71,201,122,0)}}
.btn .spinner-sm{width:14px;height:14px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}

/* ==========================================================================
   FILTER BAR
   ========================================================================== */
.filter-bar{padding:14px 0 10px;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid var(--pm-border);margin-bottom:4px}
.filter-bar::-webkit-scrollbar{display:none}
.filter-pill{padding:5px 12px;border-radius:var(--r-full);font-size:12px;font-weight:500;white-space:nowrap;border:1px solid var(--pm-border);background:transparent;color:var(--pm-text-secondary);transition:all .15s}
.filter-pill:hover{background:var(--pm-bg-card-hover);color:var(--pm-text-primary)}
.filter-pill.active{background:var(--pm-blue-bg);color:var(--pm-blue);border-color:rgba(46,92,255,.3)}

/* ==========================================================================
   STATS LINE
   ========================================================================== */
.stats-line{font-size:11.5px;color:var(--pm-text-tertiary);padding:8px 0 16px}
.stats-line strong{color:var(--pm-text-secondary);font-weight:600}

/* ==========================================================================
   ERROR
   ========================================================================== */
.error-banner{background:var(--pm-red-soft);border:1px solid rgba(255,100,100,.3);color:var(--pm-red);padding:12px 16px;border-radius:var(--r-md);margin:16px 0;font-size:13px;font-weight:500}

/* ==========================================================================
   TIER SECTIONS
   ========================================================================== */
.tier-section{margin-bottom:32px}
.tier-header{display:flex;align-items:center;gap:10px;padding:12px 0 10px;margin-bottom:12px;border-bottom:1px solid var(--pm-border)}
.tier-icon{font-size:18px;line-height:1}
.tier-label{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.6px}
.tier-count{font-size:11px;font-weight:600;padding:2px 8px;border-radius:var(--r-full);line-height:1.5}
.tier-edge .tier-label{color:var(--tier-edge)}.tier-edge .tier-count{background:var(--tier-edge-bg);color:var(--tier-edge)}
.tier-super .tier-label{color:var(--tier-super)}.tier-super .tier-count{background:var(--tier-super-bg);color:var(--tier-super)}
.tier-int .tier-label{color:var(--tier-int)}.tier-int .tier-count{background:var(--tier-int-bg);color:var(--tier-int)}
.tier-watch .tier-label{color:var(--tier-watch)}.tier-watch .tier-count{background:var(--tier-watch-bg);color:var(--tier-watch)}

/* ==========================================================================
   CARD GRID
   ========================================================================== */
.card-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
@media(max-width:860px){.card-grid{grid-template-columns:1fr}}
@media(min-width:1120px){.card-grid{grid-template-columns:repeat(3,1fr)}}

/* ==========================================================================
   MARKET CARD
   ========================================================================== */
.market-card{background:var(--pm-bg-card);border:1px solid var(--pm-border);border-radius:var(--r-lg);padding:14px 14px 12px;transition:border-color .15s,background .15s;display:flex;flex-direction:column;gap:10px;position:relative;overflow:hidden}
.market-card:hover{border-color:var(--pm-border-light);background:var(--pm-bg-card-hover)}
.market-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:3px 0 0 3px}
.tier-edge .market-card::before{background:var(--tier-edge)}
.tier-super .market-card::before{background:var(--tier-super)}
.tier-int .market-card::before{background:var(--tier-int)}
.tier-watch .market-card::before{background:var(--tier-watch)}

/* card top */
.card-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.card-question{font-size:13.5px;font-weight:600;color:var(--pm-text-primary);line-height:1.4;flex:1;min-width:0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card-question a{color:inherit;transition:color .12s}
.card-question a:hover{color:var(--pm-blue)}
.card-badges{display:flex;align-items:center;gap:4px;flex-shrink:0}
.badge-new{font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;background:var(--pm-green);color:var(--pm-bg-page);text-transform:uppercase;letter-spacing:.4px;animation:pop .3s ease}
@keyframes pop{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:scale(1)}}
.badge-multi{font-size:10px;font-weight:600;padding:1px 6px;border-radius:3px;background:var(--pm-purple-soft);color:var(--pm-purple)}
.card-countdown{font-size:12px;font-weight:700;padding:3px 7px;border-radius:var(--r-xs);white-space:nowrap}
.cd-urgent{background:var(--pm-orange-soft);color:var(--pm-orange)}
.cd-normal{background:var(--pm-bg-elevated);color:var(--pm-text-secondary)}

/* prices */
.card-prices{display:flex;gap:6px}
.price-box{flex:1;display:flex;align-items:center;justify-content:center;gap:6px;padding:9px 6px;border-radius:var(--r-md);font-weight:700}
.price-box-yes{background:var(--pm-green-soft);color:var(--pm-green)}
.price-box-no{background:var(--pm-red-soft);color:var(--pm-red)}
.price-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.3px;opacity:.7}
.price-pct{font-size:19px;letter-spacing:-.5px;font-variant-numeric:tabular-nums}

/* prob bar */
.prob-bar{height:4px;border-radius:2px;background:var(--pm-red-mid);overflow:hidden}
.prob-bar-fill{height:100%;border-radius:2px;background:var(--pm-green);transition:width .4s ease}

/* trade recommendation */
.card-trade{display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--pm-bg-surface);border-radius:var(--r-md);border:1px solid var(--pm-border);flex-wrap:wrap}
.trade-action{font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:var(--r-xs);white-space:nowrap}
.trade-action-yes{background:var(--pm-green-soft);color:var(--pm-green)}
.trade-action-no{background:var(--pm-red-soft);color:var(--pm-red)}
.trade-action-arb{background:var(--pm-blue-bg);color:var(--pm-blue)}
.trade-profit{font-size:13px;font-weight:700;color:var(--pm-green);font-variant-numeric:tabular-nums}
.trade-per{font-size:11px;color:var(--pm-text-tertiary);margin-left:-4px}
.trade-ann{font-size:11px;color:var(--pm-text-secondary);margin-left:auto;white-space:nowrap}
.trade-guaranteed{font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;background:var(--pm-green-soft);color:var(--pm-green);text-transform:uppercase;letter-spacing:.4px}
.trade-speculative{font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;background:var(--pm-orange-soft);color:var(--pm-orange);text-transform:uppercase;letter-spacing:.4px}
.trade-ev{font-size:10px;padding:2px 7px;border-radius:var(--r-xs);white-space:nowrap}
.trade-ev-pos{font-weight:600;background:var(--pm-green-soft);color:var(--pm-green)}
.trade-ev-zero{font-weight:500;background:var(--pm-bg-elevated);color:var(--pm-text-tertiary)}
.trade-warning{width:100%;font-size:10px;color:var(--pm-orange);font-weight:500;margin-top:2px}
.trade-kelly{font-size:9px;font-weight:600;padding:1px 5px;border-radius:3px;background:rgba(46,92,255,.15);color:var(--pm-blue);letter-spacing:.3px}
.health-bar{display:flex;gap:6px;align-items:center;padding:6px 12px;background:var(--pm-bg-elevated);border-radius:var(--r-xs);margin-bottom:8px;font-size:11px;flex-wrap:wrap}
.health-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.health-ok{background:var(--pm-green)}
.health-err{background:var(--pm-red)}
.health-na{background:var(--pm-text-tertiary)}
.health-item{display:flex;align-items:center;gap:3px;color:var(--pm-text-secondary)}
.history-mini{font-size:10px;color:var(--pm-text-tertiary);margin-top:4px}
.trade-edge{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:4px;background:var(--tier-edge-bg);color:var(--tier-edge)}
.edge-analysis{width:100%;padding:6px 8px;margin-top:4px;border-radius:6px;background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.15);font-size:11px;color:var(--pm-text-secondary);line-height:1.4}
.edge-analysis strong{color:var(--tier-edge);font-weight:600}
.edge-confidence{display:inline-block;font-size:9px;font-weight:600;padding:1px 5px;border-radius:3px;margin-left:4px}
.edge-confidence.high{background:rgba(0,230,118,0.15);color:var(--tier-edge)}
.edge-confidence.medium{background:rgba(255,147,50,0.15);color:var(--pm-orange)}
.edge-confidence.low{background:rgba(255,100,100,0.15);color:var(--pm-red)}

/* footer */
.card-footer{display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap}
.card-stats{display:flex;gap:10px;font-size:11.5px;color:var(--pm-text-tertiary)}
.card-stats span{display:flex;align-items:center;gap:2px}
.card-stats strong{color:var(--pm-text-secondary);font-weight:600}
.thin-flag{color:var(--pm-orange);font-weight:600;font-size:10px}
.card-tags{display:flex;gap:4px;flex-wrap:wrap}
.category-tag{font-size:10px;font-weight:600;padding:2px 7px;border-radius:var(--r-full);background:var(--pm-bg-elevated);color:var(--pm-text-secondary)}
.reason-tag{font-size:10px;font-weight:600;padding:2px 7px;border-radius:var(--r-full);white-space:nowrap}
.tier-edge .reason-tag{background:var(--tier-edge-bg);color:var(--tier-edge)}
.tier-super .reason-tag{background:var(--tier-super-bg);color:var(--tier-super)}
.tier-int .reason-tag{background:var(--tier-int-bg);color:var(--tier-int)}
.tier-watch .reason-tag{background:var(--tier-watch-bg);color:var(--tier-watch)}

/* ==========================================================================
   EMPTY & SKELETON
   ========================================================================== */
.empty-state{text-align:center;padding:28px 16px;color:var(--pm-text-tertiary);font-size:13px;border:1px dashed var(--pm-border);border-radius:var(--r-lg);grid-column:1/-1}
.skeleton-card{background:var(--pm-bg-card);border:1px solid var(--pm-border);border-radius:var(--r-lg);padding:16px;display:flex;flex-direction:column;gap:12px}
.sk-line{height:14px;border-radius:4px;background:linear-gradient(90deg,var(--pm-bg-elevated) 25%,var(--pm-bg-card-hover) 50%,var(--pm-bg-elevated) 75%);background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite}
.sk-s{width:55%}.sk-l{width:88%}
.sk-block{height:52px;border-radius:var(--r-md);background:linear-gradient(90deg,var(--pm-bg-elevated) 25%,var(--pm-bg-card-hover) 50%,var(--pm-bg-elevated) 75%);background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* ==========================================================================
   FOOTER
   ========================================================================== */
.app-footer{text-align:center;color:var(--pm-text-tertiary);font-size:12px;padding:28px 0 16px;border-top:1px solid var(--pm-border);margin-top:16px}
.app-footer a{color:var(--pm-text-secondary);transition:color .12s}
.app-footer a:hover{color:var(--pm-blue)}

/* ==========================================================================
   SCROLLBAR
   ========================================================================== */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--pm-border-light);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--pm-text-tertiary)}

/* ==========================================================================
   RESPONSIVE
   ========================================================================== */
@media(max-width:640px){
  .shell{padding:0 14px}
  .topbar-in{padding:0 14px;height:50px}
  .topbar-meta{display:none}
  .price-pct{font-size:16px}
  .card-trade{gap:6px}
  .trade-ann{margin-left:0}
}
</style>
</head>
<body>

<!-- === TOPBAR === -->
<div class="topbar">
  <div class="topbar-in">
    <div class="topbar-brand">
      <div class="topbar-logo">P</div>
      <div class="topbar-title">Polymarket <span>Scanner</span></div>
    </div>
    <div class="topbar-meta">
      <span class="topbar-opps" id="topbar-opps">0 opportunities</span>
      <span>Last: <strong id="refreshed">&mdash;</strong></span>
    </div>
    <div class="topbar-actions">
      <button class="btn btn-ghost btn-icon" id="soundBtn" title="Sound alerts">&#128263;</button>
      <div class="live-dot"></div>
      <button class="btn btn-primary" id="refreshBtn">
        <span class="spinner-sm" id="spinner"></span>
        <span id="refreshLabel">Refresh</span>
      </button>
    </div>
  </div>
</div>

<!-- === MAIN === -->
<div class="shell">

  <div id="error-area"></div>

  <div class="filter-bar" id="filter-bar">
    <button class="filter-pill active" data-cat="all">All</button>
  </div>

  <div class="stats-line" id="stats-line">
    Fetched <strong id="st-fetched">&mdash;</strong> markets &middot;
    <strong id="st-qualified">&mdash;</strong> qualified &middot;
    <strong id="st-opps">&mdash;</strong> opportunities
  </div>
  <div class="health-bar" id="health-bar"></div>

  <!-- skeleton -->
  <div id="skeleton">
    <div class="card-grid">
      <div class="skeleton-card"><div class="sk-line sk-l"></div><div class="sk-block"></div><div class="sk-line sk-s"></div></div>
      <div class="skeleton-card"><div class="sk-line sk-l"></div><div class="sk-block"></div><div class="sk-line sk-s"></div></div>
      <div class="skeleton-card"><div class="sk-line sk-l"></div><div class="sk-block"></div><div class="sk-line sk-s"></div></div>
    </div>
  </div>

  <!-- content -->
  <div id="content" style="display:none">
    <div class="tier-section tier-edge" id="sec-edge">
      <div class="tier-header">
        <span class="tier-icon">&#129504;</span>
        <span class="tier-label">Edge Informationnel</span>
        <span class="tier-count" id="cnt-edge">0</span>
      </div>
      <div class="card-grid" id="tier-edge"></div>
    </div>
    <div class="tier-section tier-super" id="sec-super">
      <div class="tier-header">
        <span class="tier-icon">&#9989;</span>
        <span class="tier-label">Gain Garanti</span>
        <span class="tier-count" id="cnt-super">0</span>
      </div>
      <div class="card-grid" id="tier-super"></div>
    </div>
    <div class="tier-section tier-int" id="sec-int">
      <div class="tier-header">
        <span class="tier-icon">&#128176;</span>
        <span class="tier-label">Petit Arb</span>
        <span class="tier-count" id="cnt-int">0</span>
      </div>
      <div class="card-grid" id="tier-int"></div>
    </div>
    <div class="tier-section tier-watch" id="sec-watch">
      <div class="tier-header">
        <span class="tier-icon">&#128064;</span>
        <span class="tier-label">Speculatif</span>
        <span class="tier-count" id="cnt-watch">0</span>
      </div>
      <div class="card-grid" id="tier-watch"></div>
    </div>
  </div>

  <footer class="app-footer">
    Auto-refresh 30s &middot; Data via
    <a href="https://polymarket.com" target="_blank" rel="noopener">Polymarket</a> Gamma API
    &middot; Cached 25s &middot; Prix indicatifs &mdash; v&eacute;rifier le carnet d&#39;ordres
  </footer>
</div>

<script>
/* =====================================================================
   JS — Scanner Engine
   ===================================================================== */
var previousIds = new Set();
var currentFilter = 'all';
var isLoading = false;
var soundOn = false;
var audioCtx = null;

/* --- Utils --- */
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}
function fmtMoney(n){
  if(n>=1e6)return '$'+(n/1e6).toFixed(1)+'M';
  if(n>=1e3)return '$'+(n/1e3).toFixed(1)+'K';
  return '$'+Math.round(n);
}
function fmtRoi(n){
  if(Math.abs(n)>=10000)return Math.round(n/1000)+'K%';
  if(Math.abs(n)>=1000)return (n/1000).toFixed(1)+'K%';
  return Math.round(n)+'%';
}

/* --- Sound --- */
function toggleSound(){
  soundOn=!soundOn;
  document.getElementById('soundBtn').innerHTML=soundOn?'&#128276;':'&#128263;';
  if(soundOn&&!audioCtx){try{audioCtx=new(window.AudioContext||window.webkitAudioContext)()}catch(e){}}
}
function playAlert(){
  if(!soundOn||!audioCtx)return;
  try{
    var o=audioCtx.createOscillator(),g=audioCtx.createGain();
    o.connect(g);g.connect(audioCtx.destination);
    o.frequency.value=880;o.type='sine';
    g.gain.setValueAtTime(0.07,audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001,audioCtx.currentTime+0.3);
    o.start();o.stop(audioCtx.currentTime+0.3);
  }catch(e){}
}

/* --- Filter --- */
function buildFilters(cats){
  var bar=document.getElementById('filter-bar');
  var h='<button class="filter-pill'+(currentFilter==='all'?' active':'')+'" data-cat="all">All</button>';
  cats.forEach(function(c){
    h+='<button class="filter-pill'+(currentFilter===c?' active':'')+'" data-cat="'+esc(c)+'">'+esc(c)+'</button>';
  });
  bar.innerHTML=h;
}
function filterCat(cat){
  currentFilter=cat;
  document.querySelectorAll('.filter-pill').forEach(function(p){
    p.classList.toggle('active',p.dataset.cat===cat);
  });
  document.querySelectorAll('.market-card').forEach(function(c){
    c.style.display=(cat==='all'||c.dataset.category===cat)?'':'none';
  });
  ['edge','super','int','watch'].forEach(function(t){
    var g=document.getElementById('tier-'+t);
    var cards=g.querySelectorAll('.market-card');
    var vis=0;cards.forEach(function(c){if(c.style.display!=='none')vis++});
    var fe=g.querySelector('.filter-empty');
    if(cards.length>0&&vis===0){
      if(!fe)g.insertAdjacentHTML('beforeend','<div class="empty-state filter-empty">No matches for this filter.</div>');
    }else if(fe){fe.remove()}
  });
}
document.getElementById('filter-bar').addEventListener('click',function(e){
  var p=e.target.closest('.filter-pill');if(p)filterCat(p.dataset.cat);
});

/* --- Render card --- */
function renderCard(item){
  var yP=(item.yes*100).toFixed(1), nP=(item.no*100).toFixed(1);
  var urg=item.days_left<=7;
  var link=item.url?'<a href="'+esc(item.url)+'" target="_blank" rel="noopener">'+esc(item.question)+'</a>':esc(item.question);

  /* badges */
  var bdg='';
  if(item.is_new)bdg+='<span class="badge-new">NEW</span>';
  if(!item.is_binary)bdg+='<span class="badge-multi">'+item.num_outcomes+' out.</span>';
  bdg+='<span class="card-countdown '+(urg?'cd-urgent':'cd-normal')+'">'+item.days_display+item.days_unit+'</span>';

  /* trade */
  var tCls='trade-action-'+item.trade_side_class;
  var tr='<span class="trade-action '+tCls+'">'+esc(item.trade_label)+'</span>';
  if(item.guaranteed){
    tr+='<span class="trade-profit">+$'+item.profit_100.toFixed(2)+'</span>'
      +'<span class="trade-per">/ $100</span>'
      +'<span class="trade-ann">'+(item.ann_roi_capped?'&ge;':'')+fmtRoi(item.ann_roi)+' ann.</span>'
      +'<span class="trade-guaranteed">GARANTI</span>'
      +'<span class="trade-ev trade-ev-pos">Net ~$'+item.net_per_100.toFixed(2)+' (frais ~'+item.fee_per_100.toFixed(0)+'%)</span>';
    if(item.arb_warning)tr+='<div class="trade-warning">&#9888; '+esc(item.arb_warning)+'</div>';
    if(item.spread_warning)tr+='<div class="trade-warning">&#9888; '+esc(item.spread_warning)+'</div>';
    if(item.clob_data&&item.clob_data.executable!==null){
      var clob=item.clob_data;
      var cls=clob.executable?'trade-ev-pos':'trade-ev-zero';
      tr+='<div class="trade-warning" style="color:var(--pm-text-secondary)">CLOB: spread '+clob.total_spread_cost_pct.toFixed(1)+'%, net arb '+clob.effective_arb_pct.toFixed(1)+'%'+(clob.executable?' &#10003;':' &#10007;')+'</div>';
    }
  }else if(item.reason_type==='edge'&&item.edge_analysis){
    var ea=item.edge_analysis;
    var edgePct=(Math.abs(ea.edge)*100).toFixed(0);
    tr+='<span class="trade-edge">EDGE +'+edgePct+'%</span>'
      +'<span class="trade-profit">EV +$'+item.ev_per_100.toFixed(2)+'</span>'
      +'<span class="trade-per">/ $100</span>'
      +'<span class="trade-ann">'+(item.ann_roi_capped?'&ge;':'')+fmtRoi(item.ann_roi)+' ann.</span>'
      +'<span class="trade-ev trade-ev-pos">Net ~$'+item.net_per_100.toFixed(2)+' (frais ~'+item.fee_per_100.toFixed(0)+'%)</span>';
  }else{
    tr+='<span class="trade-speculative">SPECULATIF</span>'
      +'<span class="trade-ev trade-ev-zero">EV ~$0 &middot; Risque -$'+item.risk_per_100.toFixed(0)+'</span>';
  }
  /* Kelly sizing */
  if(item.kelly_pct>0){
    tr+='<span class="trade-kelly">Kelly: '+item.kelly_pct.toFixed(1)+'%</span>';
  }

  /* edge analysis box */
  var eaHtml='';
  if(item.edge_analysis){
    var ea=item.edge_analysis;
    var confCls=ea.confidence||'medium';
    eaHtml='<div class="edge-analysis">'
      +'<strong>'+esc(ea.source)+'</strong>: '+esc(ea.analysis)
      +'<br>Estimation: <strong>'+(ea.estimated_prob*100).toFixed(0)+'%</strong>'
      +' vs march\u00e9 <strong>'+(ea.market_price*100).toFixed(0)+'%</strong>'
      +'<span class="edge-confidence '+confCls+'">'+confCls+'</span>'
      +'</div>';
  }

  /* stats */
  var st='<span>Vol <strong>'+fmtMoney(item.volume)+'</strong></span>';
  if(item.volume_24h>0)st+='<span>24h <strong>'+fmtMoney(item.volume_24h)+'</strong></span>';
  st+='<span>Liq <strong>'+fmtMoney(item.liquidity)+'</strong></span>';
  st+='<span>V/L <strong>'+item.vol_liq+'</strong></span>';
  if(item.thin)st+='<span class="thin-flag">Thin</span>';

  /* tags */
  var tg='<span class="category-tag">'+esc(item.category)+'</span>'
    +'<span class="reason-tag">'+esc(item.reason)+'</span>';

  return '<div class="market-card" data-category="'+esc(item.category)+'" data-id="'+esc(item.id)+'">'
    +'<div class="card-top"><div class="card-question">'+link+'</div><div class="card-badges">'+bdg+'</div></div>'
    +'<div class="card-prices">'
    +'<div class="price-box price-box-yes"><span class="price-label">'+esc(item.yes_label)+'</span><span class="price-pct">'+yP+'%</span></div>'
    +'<div class="price-box price-box-no"><span class="price-label">'+esc(item.no_label)+'</span><span class="price-pct">'+nP+'%</span></div>'
    +'</div>'
    +'<div class="prob-bar"><div class="prob-bar-fill" style="width:'+yP+'%"></div></div>'
    +'<div class="card-trade">'+tr+'</div>'
    +eaHtml
    +'<div class="card-footer"><div class="card-stats">'+st+'</div><div class="card-tags">'+tg+'</div></div>'
    +'</div>';
}

function renderTier(gridId,countId,items){
  var g=document.getElementById(gridId);
  var c=document.getElementById(countId);
  c.textContent=items?items.length:0;
  if(!items||items.length===0){
    g.innerHTML='<div class="empty-state">No opportunities in this tier right now.</div>';
    return;
  }
  g.innerHTML=items.map(renderCard).join('');
  /* re-apply filter */
  if(currentFilter!=='all'){
    g.querySelectorAll('.market-card').forEach(function(card){
      if(card.dataset.category!==currentFilter)card.style.display='none';
    });
  }
}

/* --- Main refresh --- */
function refresh(){
  if(isLoading)return;isLoading=true;
  var btn=document.getElementById('refreshBtn');
  var sp=document.getElementById('spinner');
  var lb=document.getElementById('refreshLabel');
  btn.disabled=true;sp.style.display='inline-block';lb.textContent='Loading';

  fetch('/api/scan').then(function(r){return r.json()}).then(function(d){
    document.getElementById('skeleton').style.display='none';
    document.getElementById('content').style.display='block';

    /* stats */
    document.getElementById('refreshed').textContent=d.refreshed_at;
    document.getElementById('st-fetched').textContent=d.fetched;
    document.getElementById('st-qualified').textContent=d.qualified;
    document.getElementById('st-opps').textContent=d.total_opps;
    document.getElementById('topbar-opps').textContent=d.total_opps+' opportunit'+(d.total_opps!==1?'ies':'y');

    /* analyzer health bar */
    var hb=document.getElementById('health-bar');
    if(hb&&d.analyzer_health){
      var hhtml='<span style="color:var(--pm-text-tertiary);font-weight:600">Analyzers:</span>';
      var names=['weather','finance','earnings','sports','fed_macro','elections','crypto'];
      var labels={weather:'Meteo',finance:'Finance',earnings:'Earnings',sports:'Sports',fed_macro:'Fed',elections:'Elections',crypto:'Crypto'};
      names.forEach(function(n){
        var h=d.analyzer_health[n];
        var cls=h?((h.status==='ok')?'health-ok':'health-err'):'health-na';
        hhtml+='<span class="health-item"><span class="health-dot '+cls+'"></span>'+labels[n]+'</span>';
      });
      hb.innerHTML=hhtml;
    }

    /* error */
    var ea=document.getElementById('error-area');
    if(d.error){ea.innerHTML='<div class="error-banner">API Error: '+esc(d.error)+'</div>'}
    else{ea.innerHTML=''}

    /* filters */
    buildFilters(d.categories||[]);

    /* mark NEW */
    var curIds=new Set();
    var hasNewSuper=false;
    ['edge','super','interesting','watch'].forEach(function(t){
      (d.tiers[t]||[]).forEach(function(item){
        curIds.add(item.id);
        item.is_new=(previousIds.size>0&&!previousIds.has(item.id));
        if(item.is_new&&(t==='super'||t==='edge'))hasNewSuper=true;
      });
    });
    previousIds=curIds;
    if(hasNewSuper)playAlert();

    /* render */
    renderTier('tier-edge','cnt-edge',d.tiers.edge||[]);
    renderTier('tier-super','cnt-super',d.tiers['super']);
    renderTier('tier-int','cnt-int',d.tiers.interesting);
    renderTier('tier-watch','cnt-watch',d.tiers.watch);

  }).catch(function(e){
    document.getElementById('error-area').innerHTML='<div class="error-banner">Fetch error: '+esc(String(e))+'</div>';
  }).finally(function(){
    btn.disabled=false;sp.style.display='none';lb.textContent='Refresh';isLoading=false;
  });
}

/* --- Init --- */
document.getElementById('refreshBtn').addEventListener('click',refresh);
document.getElementById('soundBtn').addEventListener('click',toggleSound);
refresh();
setInterval(refresh,30000);
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(HTML_TEMPLATE, content_type="text/html")


_scan_lock = threading.Lock()


@app.route("/api/scan")
def api_scan():
    now = _time.time()
    with _cache_lock:
        if _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
            return jsonify(_cache["data"])

    # Single-flight: only one scan() at a time, others wait and get cached result
    with _scan_lock:
        # Re-check cache — another thread may have just populated it
        with _cache_lock:
            now2 = _time.time()
            if _cache["data"] is not None and (now2 - _cache["ts"]) < CACHE_TTL:
                return jsonify(_cache["data"])

        data = scan()

        with _cache_lock:
            _cache["data"] = data
            _cache["ts"] = _time.time()

    return jsonify(data)


@app.route("/api/health")
def api_health():
    """Return analyzer health status."""
    with _health_lock:
        health = dict(_analyzer_health)
    return jsonify({
        "analyzers": health,
        "active_count": sum(1 for v in health.values() if v.get("status") == "ok"),
        "total_count": len(_ANALYZER_NAMES),
    })


@app.route("/api/history")
def api_history():
    """Return recent scan history."""
    with _history_lock:
        return jsonify(list(_scan_history))


@app.route("/api/portfolio", methods=["GET"])
def api_portfolio_get():
    """Return current portfolio."""
    return jsonify(_load_portfolio())


@app.route("/api/portfolio", methods=["POST"])
def api_portfolio_add():
    """Add position to portfolio."""
    from flask import request as flask_request
    data = flask_request.get_json(silent=True)
    if not data or "market_id" not in data:
        return jsonify({"error": "market_id required"}), 400
    portfolio = _load_portfolio()
    entry = {
        "market_id": data["market_id"],
        "question": data.get("question", ""),
        "side": data.get("side", ""),
        "price": data.get("price", 0),
        "amount": data.get("amount", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }
    portfolio.append(entry)
    _save_portfolio(portfolio)
    return jsonify({"ok": True, "count": len(portfolio)})


@app.route("/api/portfolio/<market_id>", methods=["DELETE"])
def api_portfolio_close(market_id):
    """Close a portfolio position."""
    portfolio = _load_portfolio()
    for p in portfolio:
        if p.get("market_id") == market_id and p.get("status") == "open":
            p["status"] = "closed"
            p["closed_at"] = datetime.now(timezone.utc).isoformat()
    _save_portfolio(portfolio)
    return jsonify({"ok": True})


@app.route("/api/predictions")
def api_predictions():
    """Return recent predictions for backtest review."""
    try:
        lines = []
        with open(PREDICTIONS_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        # Return last 100
        return jsonify(lines[-100:])
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify([])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _validate_api_keys():
    """Log status of all optional API keys at startup."""
    # --- OpenWeatherMap ---
    if OWM_API_KEY:
        print(f"  [OK] OpenWeatherMap key ({OWM_API_KEY[:4]}...)")
        try:
            resp = requests.get(OWM_FORECAST_URL, params={
                "lat": 40.71, "lon": -74.01, "appid": OWM_API_KEY,
                "units": "imperial", "cnt": 1,
            }, timeout=5)
            if resp.status_code == 200:
                print("       Validated — weather edge active")
            elif resp.status_code == 401:
                print("       [WARN] Key INVALID (401)")
            else:
                print(f"       [WARN] API returned {resp.status_code}")
        except Exception as exc:
            print(f"       [WARN] API unreachable: {exc}")
    else:
        print("  [--] No OpenWeatherMap key (OPENWEATHERMAP_API_KEY)")
    # --- Yahoo Finance ---
    print("  [OK] Yahoo Finance — no key needed (finance + earnings)")
    # --- CoinGecko ---
    print("  [OK] CoinGecko — no key needed (crypto prices)")
    # --- The Odds API ---
    if ODDS_API_KEY:
        print(f"  [OK] The Odds API key ({ODDS_API_KEY[:4]}...)")
        try:
            resp = requests.get(f"{ODDS_API_BASE}/sports", params={
                "apiKey": ODDS_API_KEY,
            }, timeout=5)
            if resp.status_code == 200:
                sports = resp.json()
                active = sum(1 for s in sports if s.get("active"))
                print(f"       Validated — {active} active sports")
            elif resp.status_code == 401:
                print("       [WARN] Key INVALID (401)")
            else:
                print(f"       [WARN] API returned {resp.status_code}")
        except Exception as exc:
            print(f"       [WARN] API unreachable: {exc}")
    else:
        print("  [--] No Odds API key (THE_ODDS_API_KEY for sports edge)")
    # --- FRED ---
    if FRED_API_KEY:
        print(f"  [OK] FRED API key ({FRED_API_KEY[:4]}...)")
        try:
            resp = requests.get(FRED_API_URL, params={
                "series_id": "DFEDTARU",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc",
            }, timeout=5)
            if resp.status_code == 200:
                print("       Validated — macro edge active")
            elif resp.status_code in (400, 401):
                print("       [WARN] Key INVALID")
            else:
                print(f"       [WARN] API returned {resp.status_code}")
        except Exception as exc:
            print(f"       [WARN] API unreachable: {exc}")
    else:
        print("  [--] No FRED key (FRED_API_KEY for macro edge)")
    # --- RealClearPolitics ---
    print("  [OK] Elections/Polls — no key needed (RCP + base rates)")
    # --- CLOB API ---
    print("  [OK] Polymarket CLOB — no key needed (spread verification)")
    # --- Notifications ---
    if DISCORD_WEBHOOK_URL:
        print(f"  [OK] Discord webhook configured")
    else:
        print("  [--] No Discord webhook (DISCORD_WEBHOOK_URL)")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(f"  [OK] Telegram bot configured")
    else:
        print("  [--] No Telegram bot (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
    # --- Logging ---
    print(f"  [OK] Prediction logging → {PREDICTIONS_LOG_FILE}")
    print(f"  [OK] Portfolio tracking → {PORTFOLIO_FILE}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Polymarket Live Opportunity Scanner v6")
    print("  http://localhost:5000")
    print("=" * 60)
    _validate_api_keys()
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
