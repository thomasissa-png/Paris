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

## Scoring Philosophy — EV-First + Edge Informationnel
The scanner prioritizes **real mathematical edge** over raw probability:
- **Edge informationnel** = external data (weather, finance, sports) disagrees with market → top priority
- **Arbitrage (sum < 1.0)** = guaranteed profit, no luck involved → always ranked high
- **Near-certain (>90%)** = high probability but EV ≈ $0 on a fairly priced market → shown with risk warning
- **Overround (sum > 1.0)** = market margin, NOT an opportunity → excluded

### 4 Tiers
| Tier | Label | Criteria | Edge |
|------|-------|----------|------|
| **edge** | Edge Informationnel | External data divergence > 10% | Real EV from better information |
| **super** | Gain Garanti | Arb with deviation > 4% | Guaranteed profit after fees |
| **interesting** | Petit Arb | Arb 2-4% | Guaranteed profit, tighter margin |
| **watch** | Speculatif | max_price > 90% (near-certain) | EV ≈ $0, involves luck |

**Removed**: Overround (sum > 1.0) excluded entirely — margin against you, not an opportunity.
**Removed**: Arbs < 2% — unprofitable after ~2% trading fees.
**Removed**: 80-90% probability — too speculative for "no luck" objective.

### Scoring Formula
`score = edge × (1/days_left) × log(liquidity)`
- **External edge** = `ext_edge × conf_mult × 40` (highest weight — modulated by forecast confidence)
- **Arb edge** = `deviation × 50` (dominant weight — real mechanical edge)
- **Near-certain edge** = `(max_price - 0.70) × 0.1` (minimal weight — no real edge)
- **Confidence multiplier**: `high=1.0, medium=0.7, low=0.4` — degrades score for distant forecasts

### Key Thresholds
- `MIN_LIQUIDITY = 5000` — markets below $5K liquidity are unexecutable
- `MIN_VOLUME = 1000` — minimum lifetime volume
- `MAX_ANN_ROI = 1000.0` — cap annualized ROI to avoid absurd display
- `EST_FEE_PCT = 2.0` — estimated round-trip trading fees
- `MIN_EDGE = 0.10` — minimum 10% divergence for edge tier
- `max_price > 0.995` → filtered (negligible profit) — **except** for edge tier (edge checked first)
- Volume 24h = 0 (when data available) → filtered as dead market
- Arb deviation < 2% → filtered (unprofitable after fees)

### EV Display
- **Arbs**: `+$X.XX / $100` gross profit + `Net ~$Y.YY (frais ~2%)` after fees
- **Edge**: `EDGE +X%` badge + `EV +$Y.YY / $100` + analysis box with source data
- **Non-arbs**: no profit shown — only `EV ~$0 · Risque -$100`
- **GARANTI** badge (green) for arbs, **EDGE** badge (green) for edge, **SPECULATIF** badge (orange) for non-arbs
- **Warning**: `⚠ N trades requis` for arbs (execution risk)

## External Analyzers (Edge Informationnel)
The scanner runs 3 analyzers to detect informational edge. Analyzers are **optional** — weather and sports activate with API keys, finance is always active (no key needed).

### Weather Analyzer
- **API**: OpenWeatherMap 5-day/3h forecast (free tier: 1000 calls/day)
- **Env var**: `OPENWEATHERMAP_API_KEY`
- **Startup validation**: API key tested at startup, warns if invalid/missing
- **Detects**: temperature threshold markets, rain/snow probability markets
- **Question parsing**: keyword-based detection (flexible order), not rigid regex
- **Cities**: 30+ pre-mapped US + major world cities (lat/lon lookup). "la" alias removed (false positives)
- **Cache**: 10 min TTL per city, expired entries purged automatically, thread-safe
- **Temperature**: sigmoid calibration `P = 1/(1+exp(-margin/3))` where margin = max_forecast - threshold
- **Rain**: composite formula `P(at least one) = 1 - ∏(1 - pop_i)` — not max(PoP)
- **Confidence**: degrades by forecast horizon (J+1 = high, J+2-3 = medium, J+4-5 = low)

### Finance Analyzer
- **API**: Yahoo Finance public chart endpoint (no API key needed — always active)
- **Detects**: stock price threshold markets ("Will Tesla reach $X?"), index targets, commodity prices
- **Tickers**: 20+ pre-mapped US stocks, indices (S&P, Nasdaq, Dow), commodities (gold, oil, silver)
- **Model**: Log-normal diffusion `P(S>K) = Φ(-z)` where `z = ln(K/S) / (σ√T)` — Black-Scholes style
- **Volatility**: calculated from 3-month daily log returns (RMS)
- **Cache**: 5 min TTL per ticker, expired entries purged automatically, thread-safe
- **Confidence**: 0-7d = high, 7-30d = medium, 30d+ = low

### Sports Analyzer
- **API**: The Odds API v4 (free tier: 500 requests/month)
- **Env var**: `THE_ODDS_API_KEY`
- **Startup validation**: API key tested at startup, counts active sports
- **Detects**: sports outcome markets (NBA, NFL, MLB, NHL, UFC, soccer leagues, F1, etc.)
- **Sports**: 20+ sport keys covering major US + international leagues
- **Team matching**: keyword overlap between Polymarket question and event teams (min 4-char match)
- **Probability**: devigged consensus across multiple bookmakers (raw_prob / overround)
- **Minimum**: requires >= 2 bookmakers for reliable consensus
- **Cache**: 5 min TTL per sport, expired entries purged automatically, thread-safe

### Common Edge Properties
- **Edge calc**: `estimated_prob - yes_price` (binary: float_prices[0]) → if |edge| > 10%, classified as "edge" tier
- **EV formula**: `(win_prob / buy_price - 1) × 100` where win_prob = est_prob (buy Yes) or 1-est_prob (buy No)
- **Analyzer call order**: weather → finance → sports (deterministic, first match wins)

### Future Analyzers (architecture ready)
- **Polls**: Polling aggregators — election/political probability vs. Polymarket
- **Resolution**: News APIs — detect already-resolved markets not yet settled

## Key Design Decisions
- **Crypto filter**: Regex with ONLY unambiguous tokens — short tokens like `sol`, `eth`, `ada`, `link`, `dot` were intentionally REMOVED because they cause false positives on words like "resolution", "whether", "Canada"
- **Question NOT truncated** in backend — CSS `-webkit-line-clamp` handles display truncation
- **Design system**: Polymarket brand identity — primary blue `#2e5cff`, green `#47c97a` (Yes), red `#ff6464` (No), dark bg `#12151f`, font Open Sauce One / Inter
- **Overround excluded**: sum > 1.0 is the market's margin — excluded entirely (not shown)
- **Fees estimated at ~2%**: arb profit shown net of estimated fees
- **External analyzers optional**: zero-config without API keys, enhanced with keys via env vars
- **Weather cache**: 10 min TTL per city, thread-safe, auto-purge expired entries, prevents API key exhaustion
- **Finance cache**: 5 min TTL per ticker, same purge pattern
- **Odds cache**: 5 min TTL per sport, same purge pattern
- **Analyzer call order**: weather → finance → sports via `run_analyzers()`. Order is deterministic and documented
- **All API calls synchronous**: acceptable because caches deduplicate requests. Async would add complexity for minimal gain at current scale
- **Finance always active**: Yahoo Finance needs no key — every scan checks stock/index markets for free

## Testing
**ALWAYS run tests before committing**:
```bash
python -m pytest tests/ -v
```

Test file: `tests/test_scanner.py` (203 tests) — covers:
- classify(): arb tiers (super/interesting), edge tier (before 0.995), near-certain watch-only, overround excluded, boundaries
- compute_score(): edge > arb >> near-certain, time/liquidity weighting, confidence multiplier
- Weather analyzer: city detection, sigmoid temp, composite rain, flexible keywords, horizon confidence, mocked OWM API
- Finance analyzer: ticker detection, log-normal probability model, threshold parsing, above/below direction, confidence by horizon
- Sports analyzer: sport detection, event matching by team names, devigged consensus probability, bookmaker count, mocked Odds API
- run_analyzers(): priority chain (weather → finance → sports), fallback behavior
- Edge pipeline: Yes price comparison, positive/negative edge trade recs, EV formula, confidence scoring
- Crypto regex: 10 true positives + 10 false-positive guards
- Category extraction: 8 categories + tag priority + fallback
- process_market() pipeline: arb detection + EV + fees + net, near-certain risk, ROI cap, volume 24h filter, thin detection, multi-outcome arb labels, all filters, edge cases
- Cleanup: "la" alias removed, OWM cache purge, normal CDF helper

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
