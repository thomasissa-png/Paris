# Polymarket Live Opportunity Scanner

## Project Overview
Single-file Flask web app (`polymarket_scanner.py`) that scans Polymarket prediction markets via the Gamma API and detects short-term trading opportunities ranked by attractiveness.

**Stack**: Python 3.10+, Flask, requests. No external CSS frameworks — all inline styles matching Polymarket's brand identity.

**Run**: `pip install -r requirements.txt && python polymarket_scanner.py` → http://localhost:5000

## Architecture
- **Single file**: `polymarket_scanner.py` contains backend (Flask API) + frontend (HTML/CSS/JS template)
- **API**: GET `https://gamma-api.polymarket.com/markets` with pagination (5 pages × 100 = up to 500 markets)
- **Cache**: Server-side dict with 25s TTL + thread lock — prevents API spam
- **Frontend**: Vanilla JS, no build step, auto-refreshes every 30s via `/api/scan` endpoint

## Scoring Philosophy — EV-First
The scanner prioritizes **real mathematical edge** over raw probability:
- **Arbitrage (sum < 1.0)** = guaranteed profit, no luck involved → always ranked first
- **Near-certain (>90%)** = high probability but EV ≈ $0 on a fairly priced market → shown with risk warning
- **Overround (sum > 1.0)** = market margin, NOT an opportunity → informational only

### 3 Tiers
| Tier | Label | Criteria | Edge |
|------|-------|----------|------|
| **super** | Gain Garanti | Arb with deviation > 3% | Guaranteed profit |
| **interesting** | Forte Probabilite | Arb 1-3% OR max_price > 90% | Arb = guaranteed, near-certain = speculative |
| **watch** | A Surveiller | Micro arb 0.5-1% OR 80-90% OR overround > 4% | Low edge or informational |

### Scoring Formula
`score = edge × (1/days_left) × log(liquidity)`
- **Arb edge** = `deviation × 20` (bigger gap = better)
- **Near-certain edge** = `(max_price - 0.70) × 0.3` (reduced weight — no real edge)
- **Overround edge** = `deviation × 2` (low priority)

### Key Thresholds
- `MIN_LIQUIDITY = 5000` — markets below $5K liquidity are unexecutable
- `MIN_VOLUME = 1000` — minimum lifetime volume
- `MAX_ANN_ROI = 1000.0` — cap annualized ROI to avoid absurd display
- `max_price > 0.995` → filtered (negligible profit)
- Volume 24h = 0 (when data available) → filtered as dead market

### EV Display
- **Arbs**: `EV +$X.XX garanti` (guaranteed, risk = $0)
- **Non-arbs**: `EV ~$0 · Risque -$100` (honest risk/reward)
- **GARANTI** badge for arbs, **SPECULATIF** badge for everything else

## Key Design Decisions
- **Crypto filter**: Regex with ONLY unambiguous tokens — short tokens like `sol`, `eth`, `ada`, `link`, `dot` were intentionally REMOVED because they cause false positives on words like "resolution", "whether", "Canada"
- **Question NOT truncated** in backend — CSS `-webkit-line-clamp` handles display truncation
- **Design system**: Polymarket brand identity — primary blue `#2e5cff`, green `#47c97a` (Yes), red `#ff6464` (No), dark bg `#12151f`, font Open Sauce One / Inter
- **Overround ≠ mispricing**: sum > 1.0 is the market's margin (like a bookmaker's vig), NOT an arbitrage opportunity

## Testing
**ALWAYS run tests before committing**:
```bash
python -m pytest tests/ -v
```

Test file: `tests/test_scanner.py` (108 tests) — covers:
- classify(): arb tiers (super/interesting/watch), near-certain, overround, boundaries, priority rules
- compute_score(): arb >> near-certain, edge-based ranking, time/liquidity weighting
- Crypto regex: 10 true positives + 10 false-positive guards
- Category extraction: 8 categories + tag priority + fallback
- process_market() pipeline: arb detection + EV, near-certain + risk, overround, ROI cap, volume 24h filter, thin detection, multi-outcome, all filters, edge cases

## File Structure
```
polymarket_scanner.py   # Main app (backend + frontend)
requirements.txt        # flask>=3.0, requests>=2.31
tests/
  test_scanner.py       # pytest test suite — run before every commit
run_tests.sh            # Quick test runner script
CLAUDE.md               # This file
```

## Common Pitfalls
- The Gamma API field `outcomePrices` is a JSON **string** like `'["0.94","0.06"]'`, not an array — must be parsed with `json.loads()`
- The `outcomes` field is also a JSON string: `'["Yes","No"]'`
- `volume` from the API is **lifetime** volume, not 24h — display context matters
- Multi-outcome markets have >2 prices — top outcome shown as "Yes", rest aggregated as "Others (N-1)"
- `endDate` uses ISO format with `Z` suffix — replaced with `+00:00` for `fromisoformat()`
- `sum(prices) < 1.0` = arb opportunity (guaranteed). `sum > 1.0` = overround (market margin, NOT opportunity)
- Floating-point: `0.49 + 0.50 = 0.99` but `abs(0.99 - 1.0) = 0.010000000000000009` — boundary tests must account for this
