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
# HTML template (inline, dark UI, no external CSS)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polymarket Scanner</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.6;
    padding: 20px; max-width: 1200px; margin: 0 auto;
  }
  header {
    text-align: center; padding: 24px 0 16px; border-bottom: 1px solid #21262d;
    margin-bottom: 24px;
  }
  header h1 { font-size: 1.8rem; color: #58a6ff; margin-bottom: 8px; }
  .meta { color: #8b949e; font-size: 0.9rem; margin-bottom: 12px; }
  .meta span { margin: 0 12px; }
  .refresh-btn {
    background: #238636; color: #fff; border: none; padding: 8px 20px;
    border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600;
    transition: background 0.2s;
  }
  .refresh-btn:hover { background: #2ea043; }
  .refresh-btn:disabled { background: #21262d; color: #484f58; cursor: wait; }
  .error-banner {
    background: #3d1f28; border: 1px solid #f85149; color: #f85149;
    padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;
    font-size: 0.9rem;
  }
  .tier { margin-bottom: 32px; }
  .tier-header {
    font-size: 1.3rem; font-weight: 700; padding: 8px 0 12px;
    border-bottom: 1px solid #21262d; margin-bottom: 16px;
  }
  .tier-super .tier-header { color: #f0883e; }
  .tier-interesting .tier-header { color: #d2a8ff; }
  .tier-watch .tier-header { color: #79c0ff; }
  .empty { color: #484f58; font-style: italic; padding: 8px 0; }
  .card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 16px; margin-bottom: 12px; transition: border-color 0.2s;
  }
  .card:hover { border-color: #388bfd; }
  .card-title {
    font-size: 1rem; font-weight: 600; color: #e6edf3; margin-bottom: 8px;
  }
  .card-title a { color: inherit; text-decoration: none; }
  .card-title a:hover { text-decoration: underline; color: #58a6ff; }
  .card-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px; margin-bottom: 8px;
  }
  .card-stat { font-size: 0.85rem; }
  .card-stat .label { color: #8b949e; }
  .card-stat .value { font-weight: 600; }
  .yes-val { color: #3fb950; }
  .no-val { color: #f85149; }
  .reason-tag {
    display: inline-block; background: #1f2937; color: #d2a8ff;
    font-size: 0.8rem; padding: 3px 10px; border-radius: 12px;
    margin-top: 4px;
  }
  .countdown { color: #f0883e; font-weight: 700; }
  .spinner {
    display: none; width: 16px; height: 16px; border: 2px solid #484f58;
    border-top-color: #58a6ff; border-radius: 50%;
    animation: spin 0.6s linear infinite; vertical-align: middle; margin-left: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  footer {
    text-align: center; color: #484f58; font-size: 0.8rem;
    padding: 24px 0 8px; border-top: 1px solid #21262d; margin-top: 16px;
  }
</style>
</head>
<body>

<header>
  <h1>Polymarket Live Scanner</h1>
  <div class="meta">
    <span>Markets scanned: <strong id="scanned">—</strong></span>
    <span>Last refresh: <strong id="refreshed">—</strong></span>
  </div>
  <button class="refresh-btn" id="refreshBtn" onclick="manualRefresh()">
    REFRESH NOW
  </button>
  <span class="spinner" id="spinner"></span>
</header>

<div id="error-area"></div>

<div id="content">
  <div class="tier tier-super">
    <div class="tier-header">🔥 SUPER INTERESSANT</div>
    <div id="tier-super"></div>
  </div>
  <div class="tier tier-interesting">
    <div class="tier-header">👀 INTERESSANT</div>
    <div id="tier-interesting"></div>
  </div>
  <div class="tier tier-watch">
    <div class="tier-header">📋 A REGARDER</div>
    <div id="tier-watch"></div>
  </div>
</div>

<footer>
  Auto-refreshes every 30 seconds &middot; Polymarket Opportunity Scanner
</footer>

<script>
function fmtMoney(n) {
  if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
  return '$' + n.toFixed(0);
}

function renderCard(item) {
  const urlTag = item.url
    ? '<a href="' + item.url + '" target="_blank" rel="noopener">' + escHtml(item.question) + '</a>'
    : escHtml(item.question);
  return '<div class="card">'
    + '<div class="card-title">' + urlTag + '</div>'
    + '<div class="card-grid">'
    + stat('Yes', (item.yes * 100).toFixed(1) + '%', 'yes-val')
    + stat('No', (item.no * 100).toFixed(1) + '%', 'no-val')
    + stat('Volume', fmtMoney(item.volume))
    + stat('Liquidity', fmtMoney(item.liquidity))
    + stat('Resolves in', '<span class="countdown">' + item.days_left + 'd</span>')
    + '</div>'
    + '<span class="reason-tag">' + escHtml(item.reason) + '</span>'
    + '</div>';
}

function stat(label, value, cls) {
  return '<div class="card-stat"><span class="label">' + label + '</span><br>'
    + '<span class="value' + (cls ? ' ' + cls : '') + '">' + value + '</span></div>';
}

function escHtml(s) {
  var d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

function renderTier(containerId, items) {
  var el = document.getElementById(containerId);
  if (!items || items.length === 0) {
    el.innerHTML = '<div class="empty">No opportunities in this tier right now.</div>';
    return;
  }
  el.innerHTML = items.map(renderCard).join('');
}

function refresh() {
  var btn = document.getElementById('refreshBtn');
  var spin = document.getElementById('spinner');
  btn.disabled = true;
  spin.style.display = 'inline-block';

  fetch('/api/scan')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      document.getElementById('scanned').textContent = data.scanned;
      document.getElementById('refreshed').textContent = data.refreshed_at;

      var errArea = document.getElementById('error-area');
      if (data.error) {
        errArea.innerHTML = '<div class="error-banner">API Error: ' + escHtml(data.error) + '</div>';
      } else {
        errArea.innerHTML = '';
      }

      renderTier('tier-super', data.tiers['super']);
      renderTier('tier-interesting', data.tiers.interesting);
      renderTier('tier-watch', data.tiers.watch);
    })
    .catch(function(err) {
      document.getElementById('error-area').innerHTML =
        '<div class="error-banner">Fetch error: ' + escHtml(String(err)) + '</div>';
    })
    .finally(function() {
      btn.disabled = false;
      spin.style.display = 'none';
    });
}

function manualRefresh() { refresh(); }

// Initial load + auto-refresh
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
