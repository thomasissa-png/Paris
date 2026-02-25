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

## Key Design Decisions
- **Scoring**: Composite score = `certainty × (1/days_left) × log(liquidity)` — balances probability, urgency, and market depth
- **3 tiers**: Super Interessant (>90% or >4% mispricing), Interessant (80-90% or 2-4%), A Regarder (70-80% or 1-2%)
- **Trade recommendations**: Each card shows BUY direction + entry price + profit per $100 + annualized ROI. Arbitrage opportunities (sum < 1.0) are labeled GARANTI
- **Crypto filter**: Regex with ONLY unambiguous tokens — short tokens like `sol`, `eth`, `ada`, `link`, `dot` were intentionally REMOVED because they cause false positives on words like "resolution", "whether", "Canada"
- **Question NOT truncated** in backend — CSS `-webkit-line-clamp` handles display truncation
- **Design system**: Polymarket brand identity — primary blue `#2e5cff`, green `#47c97a` (Yes), red `#ff6464` (No), dark bg `#12151f`, font Open Sauce One / Inter

## Testing
**ALWAYS run tests before committing**:
```bash
python -m pytest tests/ -v
```

Test file: `tests/test_scanner.py` — covers:
- Classification logic (all 3 tiers + edge cases)
- Score computation (time urgency, liquidity weighting)
- Crypto regex (no false positives on sol/eth/ada/link/dot)
- Category extraction (heuristic keyword matching)
- Full market processing pipeline (binary + multi-outcome + arb detection)
- ROI / profit calculations
- Edge cases (missing fields, malformed data, zero prices)

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
