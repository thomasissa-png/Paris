#!/usr/bin/env python3
"""
Polymarket Live Opportunity Scanner v2
=======================================
Full trading intelligence: ROI, annualized returns, mispricing direction,
multi-outcome support, category filtering, smart scoring.

Run:  pip install -r requirements.txt && python polymarket_scanner.py
Open: http://localhost:5000
"""

import json
import math
import re
import threading
import time as _time
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
PORT = 5000

# Crypto filter — ONLY unambiguous tokens
# Removed: sol, eth, ada, link, dot, bnb, matic, ltc (too many false positives)
CRYPTO_RE = re.compile(
    r"\b("
    r"bitcoin|btc|ethereum|solana|crypto(?:currenc(?:y|ies))?|"
    r"dogecoin|doge|ripple|xrp|cardano|avalanche|avax|litecoin|"
    r"chainlink|polkadot|shiba\s*inu|pepe\s*coin|memecoin|"
    r"token\s+price|coin\s+price|crypto\s+price|"
    r"defi|nft|blockchain|stablecoin|altcoin|"
    r"binance|coinbase|uniswap|aave|satoshi|gwei"
    r")\b",
    re.IGNORECASE,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_cache: dict = {"data": None, "ts": 0.0}

# ---------------------------------------------------------------------------
# API — paginated fetch
# ---------------------------------------------------------------------------

def fetch_all_markets() -> list[dict]:
    """Fetch up to MAX_PAGES * BATCH_SIZE markets with pagination."""
    all_markets: list[dict] = []
    for page in range(MAX_PAGES):
        try:
            params = {**API_PARAMS, "limit": BATCH_SIZE, "offset": page * BATCH_SIZE}
            resp = requests.get(GAMMA_API, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < BATCH_SIZE:
                break
        except Exception:
            break  # use what we have
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


def is_crypto(question: str) -> bool:
    return bool(CRYPTO_RE.search(question))


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
                       "supreme court", "scotus", "parliament", "prime minister"]),
        ("Sports", ["nfl", "nba", "mlb", "nhl", "soccer", "football",
                     "basketball", "baseball", "tennis", "ufc", "boxing",
                     "super bowl", "world cup", "championship", "playoffs",
                     "grand slam", "formula 1", "f1 ", "olympics"]),
        ("Economics", ["fed ", "interest rate", "gdp", "inflation", "stock",
                       "s&p", "nasdaq", "dow jones", "recession", "cpi",
                       "jobs report", "unemployment", "treasury", "tariff",
                       "oil price", "gold price"]),
        ("Tech", [" ai ", "openai", "google", "apple", "microsoft", "tesla",
                  "spacex", "chatgpt", "artificial intelligence", "llm "]),
        ("Entertainment", ["oscar", "grammy", "emmy", "movie", "album",
                          "celebrity", "tiktok", "youtube", "netflix",
                          "disney", "box office", "billboard"]),
        ("Geopolitics", ["war ", "ukraine", "russia", "china", "nato",
                        "sanction", "missile", "military", "iran", "israel",
                        "ceasefire", "invasion", "north korea"]),
        ("Weather", ["weather", "hurricane", "earthquake", "temperature",
                     "climate", "wildfire", "flood", "tornado"]),
        ("Science", ["fda", "vaccine", "covid", "trial", "study",
                     "nasa", "mars ", "space station"]),
    ]
    for cat, keywords in rules:
        if any(kw in q for kw in keywords):
            return cat
    return "Other"

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def classify(max_price: float, deviation: float,
             price_sum: float) -> tuple[str | None, str, str]:
    """Classify opportunity: (tier, reason_label, reason_type) or (None,'','').

    Philosophy: only REAL edge, no luck.
    - arbitrage (sum < 1.0, >2%) = guaranteed profit → super/interesting
    - near_certain (>90%) = EV ≈ $0, involves luck → watch only
    - overround (sum > 1.0) = excluded (margin against you)
    - arbs <2% excluded (unprofitable after ~2% fees)
    """
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


def compute_score(deviation: float, price_sum: float, max_price: float,
                  days_left: float, liquidity: float) -> float:
    """Score by attractiveness. Arbs (real edge) >> near-certain (no edge)."""
    if price_sum < 1.0 and deviation > 0.02:
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

    # --- Crypto gate ---
    if is_crypto(question):
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

    # --- Classify ---
    tier, reason_label, reason_type = classify(max_price, deviation, price_sum)
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
        fee_per_100 = round(EST_FEE_PCT, 2)
        net_per_100 = round(ev_per_100 - fee_per_100, 2)
        arb_warning = f"{num_outcomes} trades requis"
    else:
        ev_per_100 = 0.0
        risk_per_100 = 100.0
        fee_per_100 = 0.0
        net_per_100 = 0.0
        arb_warning = ""

    # --- Vol/Liq ratio ---
    vol_liq = round(volume / liquidity, 1) if liquidity > 0 else 999
    thin = liquidity < 10000 or vol_liq > 20

    # --- Composite score ---
    score = compute_score(deviation, price_sum, max_price, days_left, liquidity)

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
            "tiers": {"super": [], "interesting": [], "watch": []},
        }

    fetched = len(raw)
    tiers: dict[str, list] = {"super": [], "interesting": [], "watch": []}
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

    return {
        "error": None,
        "fetched": fetched,
        "qualified": qualified,
        "total_opps": total_opps,
        "refreshed_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "categories": sorted(cats),
        "tiers": tiers,
    }


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

/* footer */
.card-footer{display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap}
.card-stats{display:flex;gap:10px;font-size:11.5px;color:var(--pm-text-tertiary)}
.card-stats span{display:flex;align-items:center;gap:2px}
.card-stats strong{color:var(--pm-text-secondary);font-weight:600}
.thin-flag{color:var(--pm-orange);font-weight:600;font-size:10px}
.card-tags{display:flex;gap:4px;flex-wrap:wrap}
.category-tag{font-size:10px;font-weight:600;padding:2px 7px;border-radius:var(--r-full);background:var(--pm-bg-elevated);color:var(--pm-text-secondary)}
.reason-tag{font-size:10px;font-weight:600;padding:2px 7px;border-radius:var(--r-full);white-space:nowrap}
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
  ['super','int','watch'].forEach(function(t){
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
  }else{
    tr+='<span class="trade-speculative">SPECULATIF</span>'
      +'<span class="trade-ev trade-ev-zero">EV ~$0 &middot; Risque -$'+item.risk_per_100.toFixed(0)+'</span>';
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

    /* error */
    var ea=document.getElementById('error-area');
    if(d.error){ea.innerHTML='<div class="error-banner">API Error: '+esc(d.error)+'</div>'}
    else{ea.innerHTML=''}

    /* filters */
    buildFilters(d.categories||[]);

    /* mark NEW */
    var curIds=new Set();
    var hasNewSuper=false;
    ['super','interesting','watch'].forEach(function(t){
      (d.tiers[t]||[]).forEach(function(item){
        curIds.add(item.id);
        item.is_new=(previousIds.size>0&&!previousIds.has(item.id));
        if(item.is_new&&t==='super')hasNewSuper=true;
      });
    });
    previousIds=curIds;
    if(hasNewSuper)playAlert();

    /* render */
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


@app.route("/api/scan")
def api_scan():
    now = _time.time()
    with _cache_lock:
        if _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
            return jsonify(_cache["data"])

    data = scan()

    with _cache_lock:
        _cache["data"] = data
        _cache["ts"] = _time.time()

    return jsonify(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Polymarket Live Opportunity Scanner v2")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
