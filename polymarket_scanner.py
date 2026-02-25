#!/usr/bin/env python3
"""
Polymarket Live Opportunity Scanner
====================================
Fetches open Polymarket markets via the Gamma API and detects
short-term trading opportunities, ranked by attractiveness.

Run:  python polymarket_scanner.py
Then open http://localhost:5000
"""

import json
import re
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, Response

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GAMMA_API = "https://gamma-api.polymarket.com/markets"
API_PARAMS = {
    "active": "true",
    "closed": "false",
    "limit": 100,
    "order": "volume",
    "ascending": "false",
}
MAX_DAYS = 60
MIN_VOLUME = 1000
MIN_LIQUIDITY = 500
CRYPTO_KEYWORDS = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|doge|xrp|bnb|cardano|ada|"
    r"avalanche|avax|polygon|matic|litecoin|ltc|chainlink|link|polkadot|dot|"
    r"shiba|pepe|memecoin|token price|coin price|crypto price)\b",
    re.IGNORECASE,
)
REFRESH_SECONDS = 30
PORT = 5000

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data fetching & scoring
# ---------------------------------------------------------------------------

def fetch_markets() -> list[dict]:
    """Fetch markets from the Gamma API."""
    resp = requests.get(GAMMA_API, params=API_PARAMS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    # Handle various ISO formats
    date_str = date_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def is_crypto(question: str) -> bool:
    return bool(CRYPTO_KEYWORDS.search(question))


def classify_opportunity(yes_price: float, no_price: float) -> tuple[str | None, str]:
    """Return (tier, reason) or (None, '') if not interesting."""
    price_sum = yes_price + no_price
    deviation = abs(price_sum - 1.0)
    max_price = max(yes_price, no_price)

    # --- SUPER INTERESSANT ---
    if max_price > 0.90:
        return ("super", "Quasi-certain outcome (>{:.0f}%)".format(max_price * 100))
    if deviation > 0.04:
        return ("super", "Mispricing detected (sum={:.2f}, dev={:.1f}%)".format(
            price_sum, deviation * 100))

    # --- INTERESSANT ---
    if 0.80 < max_price <= 0.90:
        return ("interesting", "High-confidence outcome ({:.0f}%)".format(max_price * 100))
    if 0.02 < deviation <= 0.04:
        return ("interesting", "Moderate mispricing (sum={:.2f}, dev={:.1f}%)".format(
            price_sum, deviation * 100))

    # --- A REGARDER ---
    if 0.70 < max_price <= 0.80:
        return ("watch", "Notable imbalance ({:.0f}%)".format(max_price * 100))
    if 0.01 < deviation <= 0.02:
        return ("watch", "Slight mispricing (sum={:.2f}, dev={:.1f}%)".format(
            price_sum, deviation * 100))

    return (None, "")


def scan() -> dict:
    """Scan markets and return structured results."""
    now = datetime.now(timezone.utc)

    try:
        raw_markets = fetch_markets()
    except Exception as exc:
        return {
            "error": str(exc),
            "scanned": 0,
            "refreshed_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "tiers": {"super": [], "interesting": [], "watch": []},
        }

    results: dict[str, list] = {"super": [], "interesting": [], "watch": []}
    scanned = 0

    for m in raw_markets:
        question = m.get("question", "") or m.get("title", "")
        slug = m.get("slug", "")
        end_date_str = m.get("endDate")
        volume = parse_float(m.get("volume"))
        liquidity = parse_float(m.get("liquidity"))
        outcome_prices_raw = m.get("outcomePrices", "[]")

        # Parse outcome prices
        try:
            prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
        except (json.JSONDecodeError, TypeError):
            continue

        if not prices or len(prices) < 2:
            continue

        yes_price = parse_float(prices[0])
        no_price = parse_float(prices[1])

        # Parse end date
        end_dt = parse_date(end_date_str)
        if end_dt is None:
            continue
        days_left = (end_dt - now).days
        if days_left < 0 or days_left > MAX_DAYS:
            continue

        # Volume / liquidity filters
        if volume < MIN_VOLUME or liquidity < MIN_LIQUIDITY:
            continue

        # Exclude crypto
        if is_crypto(question):
            continue

        scanned += 1

        tier, reason = classify_opportunity(yes_price, no_price)
        if tier is None:
            continue

        results[tier].append({
            "question": question[:80],
            "yes": yes_price,
            "no": no_price,
            "volume": volume,
            "liquidity": liquidity,
            "days_left": days_left,
            "reason": reason,
            "url": f"https://polymarket.com/event/{slug}" if slug else "",
        })

    # Sort each tier by days_left ascending
    for tier in results:
        results[tier].sort(key=lambda x: x["days_left"])

    return {
        "error": None,
        "scanned": scanned,
        "refreshed_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tiers": results,
    }


# ---------------------------------------------------------------------------
# HTML template — Polymarket brand identity, flat design, dark theme
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
/* ============================================================
   POLYMARKET DESIGN SYSTEM — Design Tokens
   ============================================================ */
:root {
  /* ---- Backgrounds ---- */
  --pm-bg-page:       #12151f;
  --pm-bg-surface:    #181b28;
  --pm-bg-card:       #1c2030;
  --pm-bg-card-hover: #222639;
  --pm-bg-elevated:   #262a3d;
  --pm-bg-input:      #1c2030;

  /* ---- Borders ---- */
  --pm-border:        #262a3b;
  --pm-border-light:  #2e3348;
  --pm-border-focus:  #2e5cff;

  /* ---- Text ---- */
  --pm-text-primary:   #f0f2f5;
  --pm-text-secondary: #858d9d;
  --pm-text-tertiary:  #5d6577;
  --pm-text-inverse:   #12151f;

  /* ---- Brand colors ---- */
  --pm-blue:           #2e5cff;
  --pm-blue-hover:     #4d75ff;
  --pm-blue-pressed:   #2249d6;
  --pm-blue-bg:        rgba(46, 92, 255, 0.10);
  --pm-blue-bg-hover:  rgba(46, 92, 255, 0.18);

  /* ---- Outcome colors ---- */
  --pm-green:          #47c97a;
  --pm-green-soft:     rgba(71, 201, 122, 0.12);
  --pm-green-mid:      rgba(71, 201, 122, 0.22);
  --pm-red:            #ff6464;
  --pm-red-soft:       rgba(255, 100, 100, 0.10);
  --pm-red-mid:        rgba(255, 100, 100, 0.20);

  /* ---- Accent ---- */
  --pm-orange:         #ff9332;
  --pm-orange-soft:    rgba(255, 147, 50, 0.12);
  --pm-purple:         #9b6dff;
  --pm-purple-soft:    rgba(155, 109, 255, 0.12);
  --pm-cyan:           #3dc2ec;
  --pm-cyan-soft:      rgba(61, 194, 236, 0.10);

  /* ---- Tier accents ---- */
  --tier-super-color:       #ff9332;
  --tier-super-bg:          rgba(255, 147, 50, 0.08);
  --tier-super-border:      rgba(255, 147, 50, 0.25);
  --tier-interesting-color:  #9b6dff;
  --tier-interesting-bg:     rgba(155, 109, 255, 0.06);
  --tier-interesting-border: rgba(155, 109, 255, 0.20);
  --tier-watch-color:        #3dc2ec;
  --tier-watch-bg:           rgba(61, 194, 236, 0.05);
  --tier-watch-border:       rgba(61, 194, 236, 0.18);

  /* ---- Radius ---- */
  --pm-radius-xs:  4px;
  --pm-radius-sm:  6px;
  --pm-radius-md:  8px;
  --pm-radius-lg:  12px;
  --pm-radius-xl:  16px;
  --pm-radius-full: 100px;

  /* ---- Typography ---- */
  --pm-font: "Open Sauce One", "Inter", -apple-system, BlinkMacSystemFont,
             "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --pm-mono: "SF Mono", "Fira Code", "Fira Mono", "Roboto Mono", monospace;
}

/* ============================================================
   BASE RESET & GLOBALS
   ============================================================ */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html {
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  font-family: var(--pm-font);
  background: var(--pm-bg-page);
  color: var(--pm-text-primary);
  line-height: 1.5;
  min-height: 100vh;
}

a { color: inherit; text-decoration: none; }
button { font-family: var(--pm-font); }

/* ============================================================
   LAYOUT SHELL
   ============================================================ */
.app-shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 24px;
}

/* ============================================================
   TOP BAR — sticky navigation bar
   ============================================================ */
.topbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(18, 21, 31, 0.82);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--pm-border);
}

.topbar-inner {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 24px;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.topbar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.topbar-logo {
  width: 28px;
  height: 28px;
  background: var(--pm-blue);
  border-radius: var(--pm-radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-weight: 700;
  font-size: 15px;
}

.topbar-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--pm-text-primary);
  letter-spacing: -0.2px;
}

.topbar-title span {
  color: var(--pm-text-tertiary);
  font-weight: 400;
  margin-left: 6px;
}

.topbar-meta {
  display: flex;
  align-items: center;
  gap: 20px;
  font-size: 12.5px;
  color: var(--pm-text-tertiary);
}

.topbar-meta strong {
  color: var(--pm-text-secondary);
  font-weight: 600;
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

/* ============================================================
   BUTTONS
   ============================================================ */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 14px;
  border: none;
  border-radius: var(--pm-radius-sm);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s ease, opacity 0.15s ease;
  white-space: nowrap;
}

.btn-primary {
  background: var(--pm-blue);
  color: #fff;
}
.btn-primary:hover { background: var(--pm-blue-hover); }
.btn-primary:active { background: var(--pm-blue-pressed); }
.btn-primary:disabled {
  background: var(--pm-bg-elevated);
  color: var(--pm-text-tertiary);
  cursor: not-allowed;
}

.btn-ghost {
  background: transparent;
  color: var(--pm-text-secondary);
  border: 1px solid var(--pm-border);
}
.btn-ghost:hover {
  background: var(--pm-bg-card-hover);
  color: var(--pm-text-primary);
  border-color: var(--pm-border-light);
}

.btn-icon {
  padding: 7px;
  width: 32px;
  height: 32px;
  justify-content: center;
}

/* live dot animation */
.live-dot {
  width: 6px;
  height: 6px;
  background: var(--pm-green);
  border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(71, 201, 122, 0.5); }
  50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(71, 201, 122, 0); }
}

/* spinner inside button */
.btn .spinner-sm {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255,255,255,0.25);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
  display: none;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* ============================================================
   ERROR BANNER
   ============================================================ */
.error-banner {
  background: var(--pm-red-soft);
  border: 1px solid rgba(255, 100, 100, 0.30);
  color: var(--pm-red);
  padding: 12px 16px;
  border-radius: var(--pm-radius-md);
  margin: 16px 0;
  font-size: 13px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ============================================================
   TIER SECTIONS
   ============================================================ */
.tier-section {
  margin-bottom: 36px;
}

.tier-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 0 12px;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--pm-border);
}

.tier-icon {
  font-size: 18px;
  line-height: 1;
}

.tier-label {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}

.tier-count {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: var(--pm-radius-full);
  line-height: 1.5;
}

/* Tier color variants */
.tier-super .tier-label  { color: var(--tier-super-color); }
.tier-super .tier-count  { background: var(--tier-super-bg); color: var(--tier-super-color); }

.tier-interesting .tier-label { color: var(--tier-interesting-color); }
.tier-interesting .tier-count { background: var(--tier-interesting-bg); color: var(--tier-interesting-color); }

.tier-watch .tier-label { color: var(--tier-watch-color); }
.tier-watch .tier-count { background: var(--tier-watch-bg); color: var(--tier-watch-color); }

/* ============================================================
   CARD GRID — responsive 2-col on desktop, 1-col on mobile
   ============================================================ */
.card-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

@media (max-width: 820px) {
  .card-grid { grid-template-columns: 1fr; }
}

@media (min-width: 1100px) {
  .card-grid { grid-template-columns: repeat(3, 1fr); }
}

/* ============================================================
   MARKET CARD — Polymarket flat style
   ============================================================ */
.market-card {
  background: var(--pm-bg-card);
  border: 1px solid var(--pm-border);
  border-radius: var(--pm-radius-lg);
  padding: 16px;
  transition: border-color 0.15s ease, background 0.15s ease;
  display: flex;
  flex-direction: column;
  gap: 12px;
  position: relative;
  overflow: hidden;
}

.market-card:hover {
  border-color: var(--pm-border-light);
  background: var(--pm-bg-card-hover);
}

/* Tier left accent stripe */
.market-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  border-radius: 3px 0 0 3px;
}
.tier-super .market-card::before    { background: var(--tier-super-color); }
.tier-interesting .market-card::before { background: var(--tier-interesting-color); }
.tier-watch .market-card::before    { background: var(--tier-watch-color); }

/* ---- Card top row: question + countdown ---- */
.card-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.card-question {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--pm-text-primary);
  line-height: 1.4;
  flex: 1;
  min-width: 0;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-question a {
  color: inherit;
  text-decoration: none;
  transition: color 0.12s ease;
}
.card-question a:hover { color: var(--pm-blue); }

.card-countdown {
  flex-shrink: 0;
  font-size: 12px;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: var(--pm-radius-xs);
  white-space: nowrap;
}
.countdown-urgent {
  background: var(--pm-orange-soft);
  color: var(--pm-orange);
}
.countdown-normal {
  background: var(--pm-bg-elevated);
  color: var(--pm-text-secondary);
}

/* ---- Price display: big Yes/No percentages ---- */
.card-prices {
  display: flex;
  gap: 8px;
}

.price-box {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 8px;
  border-radius: var(--pm-radius-md);
  font-weight: 700;
}

.price-box-yes {
  background: var(--pm-green-soft);
  color: var(--pm-green);
}
.price-box-no {
  background: var(--pm-red-soft);
  color: var(--pm-red);
}

.price-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  opacity: 0.7;
}

.price-pct {
  font-size: 20px;
  letter-spacing: -0.5px;
  font-variant-numeric: tabular-nums;
}

/* ---- Probability bar: visual representation ---- */
.prob-bar {
  height: 4px;
  border-radius: 2px;
  background: var(--pm-red-mid);
  overflow: hidden;
}

.prob-bar-fill {
  height: 100%;
  border-radius: 2px;
  background: var(--pm-green);
  transition: width 0.4s ease;
}

/* ---- Card footer: stats + reason ---- */
.card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}

.card-stats {
  display: flex;
  gap: 14px;
  font-size: 12px;
  color: var(--pm-text-tertiary);
}

.card-stats span {
  display: flex;
  align-items: center;
  gap: 3px;
}

.card-stats strong {
  color: var(--pm-text-secondary);
  font-weight: 600;
}

.reason-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: var(--pm-radius-full);
  white-space: nowrap;
}

.tier-super .reason-tag {
  background: var(--tier-super-bg);
  color: var(--tier-super-color);
}
.tier-interesting .reason-tag {
  background: var(--tier-interesting-bg);
  color: var(--tier-interesting-color);
}
.tier-watch .reason-tag {
  background: var(--tier-watch-bg);
  color: var(--tier-watch-color);
}

/* ============================================================
   EMPTY STATE
   ============================================================ */
.empty-state {
  text-align: center;
  padding: 32px 16px;
  color: var(--pm-text-tertiary);
  font-size: 13px;
  border: 1px dashed var(--pm-border);
  border-radius: var(--pm-radius-lg);
}

.empty-state-icon {
  font-size: 28px;
  margin-bottom: 8px;
  opacity: 0.5;
}

/* ============================================================
   SKELETON LOADING
   ============================================================ */
.skeleton-card {
  background: var(--pm-bg-card);
  border: 1px solid var(--pm-border);
  border-radius: var(--pm-radius-lg);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.skeleton-line {
  height: 14px;
  border-radius: 4px;
  background: linear-gradient(90deg,
    var(--pm-bg-elevated) 25%,
    var(--pm-bg-card-hover) 50%,
    var(--pm-bg-elevated) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}

.skeleton-line-short { width: 60%; }
.skeleton-line-long  { width: 90%; }
.skeleton-block {
  height: 48px;
  border-radius: var(--pm-radius-md);
  background: linear-gradient(90deg,
    var(--pm-bg-elevated) 25%,
    var(--pm-bg-card-hover) 50%,
    var(--pm-bg-elevated) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* ============================================================
   FOOTER
   ============================================================ */
.app-footer {
  text-align: center;
  color: var(--pm-text-tertiary);
  font-size: 12px;
  padding: 32px 0 20px;
  border-top: 1px solid var(--pm-border);
  margin-top: 20px;
}

.app-footer a {
  color: var(--pm-text-secondary);
  transition: color 0.12s ease;
}
.app-footer a:hover { color: var(--pm-blue); }

/* ============================================================
   SCROLLBAR
   ============================================================ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--pm-border-light);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--pm-text-tertiary); }

/* ============================================================
   RESPONSIVE ADJUSTMENTS
   ============================================================ */
@media (max-width: 600px) {
  .app-shell { padding: 0 14px; }
  .topbar-inner { padding: 0 14px; height: 50px; }
  .topbar-meta { display: none; }
  .price-pct { font-size: 17px; }
  .card-stats { gap: 10px; }
}
</style>
</head>
<body>

<!-- ==================== TOP BAR ==================== -->
<div class="topbar">
  <div class="topbar-inner">
    <div class="topbar-brand">
      <div class="topbar-logo">P</div>
      <div class="topbar-title">Polymarket <span>Scanner</span></div>
    </div>
    <div class="topbar-meta">
      <span>Markets scanned: <strong id="scanned">—</strong></span>
      <span>Last refresh: <strong id="refreshed">—</strong></span>
    </div>
    <div class="topbar-actions">
      <div class="live-dot" id="liveDot"></div>
      <button class="btn btn-primary" id="refreshBtn" onclick="manualRefresh()">
        <span class="spinner-sm" id="spinner"></span>
        <span id="refreshLabel">Refresh</span>
      </button>
    </div>
  </div>
</div>

<!-- ==================== MAIN CONTENT ==================== -->
<div class="app-shell">

  <div id="error-area"></div>

  <!-- initial skeleton -->
  <div id="skeleton">
    <div class="tier-section">
      <div class="tier-header"><div class="skeleton-line skeleton-line-short"></div></div>
      <div class="card-grid">
        <div class="skeleton-card"><div class="skeleton-line skeleton-line-long"></div><div class="skeleton-block"></div><div class="skeleton-line skeleton-line-short"></div></div>
        <div class="skeleton-card"><div class="skeleton-line skeleton-line-long"></div><div class="skeleton-block"></div><div class="skeleton-line skeleton-line-short"></div></div>
        <div class="skeleton-card"><div class="skeleton-line skeleton-line-long"></div><div class="skeleton-block"></div><div class="skeleton-line skeleton-line-short"></div></div>
      </div>
    </div>
  </div>

  <div id="content" style="display:none;">
    <div class="tier-section tier-super" id="section-super">
      <div class="tier-header">
        <span class="tier-icon">🔥</span>
        <span class="tier-label">Super Interessant</span>
        <span class="tier-count" id="count-super">0</span>
      </div>
      <div class="card-grid" id="tier-super"></div>
    </div>

    <div class="tier-section tier-interesting" id="section-interesting">
      <div class="tier-header">
        <span class="tier-icon">👀</span>
        <span class="tier-label">Interessant</span>
        <span class="tier-count" id="count-interesting">0</span>
      </div>
      <div class="card-grid" id="tier-interesting"></div>
    </div>

    <div class="tier-section tier-watch" id="section-watch">
      <div class="tier-header">
        <span class="tier-icon">📋</span>
        <span class="tier-label">A Regarder</span>
        <span class="tier-count" id="count-watch">0</span>
      </div>
      <div class="card-grid" id="tier-watch"></div>
    </div>
  </div>

  <footer class="app-footer">
    Auto-refresh every 30 s &middot; Data from
    <a href="https://polymarket.com" target="_blank" rel="noopener">Polymarket</a>
    Gamma API
  </footer>
</div>

<script>
/* ==============================================================
   JS — Fetch, Render, Auto-refresh
   ============================================================== */

function esc(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function fmtMoney(n) {
  if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
  return '$' + Math.round(n);
}

function renderCard(item) {
  var yesPct = (item.yes * 100).toFixed(1);
  var noPct  = (item.no  * 100).toFixed(1);
  var isUrgent = item.days_left <= 7;
  var title = item.url
    ? '<a href="' + item.url + '" target="_blank" rel="noopener">' + esc(item.question) + '</a>'
    : esc(item.question);

  return ''
    + '<div class="market-card">'
    +   '<div class="card-top">'
    +     '<div class="card-question">' + title + '</div>'
    +     '<div class="card-countdown ' + (isUrgent ? 'countdown-urgent' : 'countdown-normal') + '">'
    +       item.days_left + 'd'
    +     '</div>'
    +   '</div>'
    +   '<div class="card-prices">'
    +     '<div class="price-box price-box-yes">'
    +       '<span class="price-label">Yes</span>'
    +       '<span class="price-pct">' + yesPct + '%</span>'
    +     '</div>'
    +     '<div class="price-box price-box-no">'
    +       '<span class="price-label">No</span>'
    +       '<span class="price-pct">' + noPct + '%</span>'
    +     '</div>'
    +   '</div>'
    +   '<div class="prob-bar"><div class="prob-bar-fill" style="width:' + yesPct + '%"></div></div>'
    +   '<div class="card-footer">'
    +     '<div class="card-stats">'
    +       '<span>Vol <strong>' + fmtMoney(item.volume) + '</strong></span>'
    +       '<span>Liq <strong>' + fmtMoney(item.liquidity) + '</strong></span>'
    +     '</div>'
    +     '<span class="reason-tag">' + esc(item.reason) + '</span>'
    +   '</div>'
    + '</div>';
}

function renderTier(containerId, countId, items) {
  var el = document.getElementById(containerId);
  var countEl = document.getElementById(countId);
  countEl.textContent = items ? items.length : 0;

  if (!items || items.length === 0) {
    el.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">'
      + '<div class="empty-state-icon">—</div>'
      + 'No opportunities in this tier right now.'
      + '</div>';
    return;
  }
  el.innerHTML = items.map(renderCard).join('');
}

var isLoading = false;

function refresh() {
  if (isLoading) return;
  isLoading = true;

  var btn = document.getElementById('refreshBtn');
  var spin = document.getElementById('spinner');
  var label = document.getElementById('refreshLabel');
  btn.disabled = true;
  spin.style.display = 'inline-block';
  label.textContent = 'Loading';

  fetch('/api/scan')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      /* Hide skeleton, show content */
      document.getElementById('skeleton').style.display = 'none';
      document.getElementById('content').style.display = 'block';

      document.getElementById('scanned').textContent = data.scanned;
      document.getElementById('refreshed').textContent = data.refreshed_at;

      var errArea = document.getElementById('error-area');
      if (data.error) {
        errArea.innerHTML = '<div class="error-banner">⚠ API Error: ' + esc(data.error) + '</div>';
      } else {
        errArea.innerHTML = '';
      }

      renderTier('tier-super', 'count-super', data.tiers['super']);
      renderTier('tier-interesting', 'count-interesting', data.tiers.interesting);
      renderTier('tier-watch', 'count-watch', data.tiers.watch);
    })
    .catch(function(err) {
      document.getElementById('error-area').innerHTML =
        '<div class="error-banner">⚠ Fetch error: ' + esc(String(err)) + '</div>';
    })
    .finally(function() {
      btn.disabled = false;
      spin.style.display = 'none';
      label.textContent = 'Refresh';
      isLoading = false;
    });
}

function manualRefresh() { refresh(); }

/* Initial load + auto-refresh */
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(HTML_TEMPLATE, content_type="text/html")


@app.route("/api/scan")
def api_scan():
    data = scan()
    return jsonify(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Polymarket Live Opportunity Scanner")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
