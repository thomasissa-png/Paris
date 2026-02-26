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
| **edge** | Edge Informationnel | External data divergence > 5% | Real EV from better information |
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
- `EST_FEE_PCT = 2.0` — estimated round-trip trading fees per trade (scales with num_outcomes)
- `MIN_EDGE = 0.05` — minimum 5% divergence for edge tier
- `max_price > 0.995` → filtered (negligible profit) — **except** for edge tier (edge checked first)
- Volume 24h = 0 (when data available) → filtered as dead market
- Arb deviation < 2% → filtered (unprofitable after fees)

### EV Display
- **Arbs**: `+$X.XX / $100` gross profit + `Net ~$Y.YY (frais ~2%)` after fees
- **Edge**: `EDGE +X%` badge + `EV +$Y.YY / $100` + `Net ~$Z.ZZ (frais ~N%)` + analysis box with source data
- **Non-arbs**: no profit shown — only `EV ~$0 · Risque -$100`
- **Spread warning**: `⚠ Faible liquidité` for low-liquidity arbs, per-outcome liquidity check for multi-outcome
- **GARANTI** badge (green) for arbs, **EDGE** badge (green) for edge, **SPECULATIF** badge (orange) for non-arbs
- **Warning**: `⚠ N trades requis` for arbs (execution risk)

## External Analyzers (Edge Informationnel)
The scanner runs 7 analyzers to detect informational edge. Analyzers are **optional** — some activate with API keys, others are always active (no key needed).

### Weather Analyzer
- **API**: OpenWeatherMap 5-day/3h forecast (free tier: 1000 calls/day)
- **Env var**: `OPENWEATHERMAP_API_KEY`
- **Startup validation**: API key tested at startup, warns if invalid/missing
- **Detects**: temperature threshold markets, rain/snow probability markets
- **Question parsing**: keyword-based detection (flexible order), not rigid regex. Target date extraction ("March 5") narrows forecast to specific day
- **Snow vs Rain**: differentiated — snow questions use OWM weather type + temperature-based probability scaling
- **Cities**: 50+ pre-mapped US + major world cities (lat/lon lookup). "la" alias removed (false positives)
- **Cache**: 10 min TTL per city, expired entries purged automatically, thread-safe
- **Temperature**: sigmoid calibration `P = 1/(1+exp(-margin/3))` where margin = max_forecast - threshold
- **Rain**: composite formula `P(at least one) = 1 - ∏(1 - pop_i)` — not max(PoP)
- **Confidence**: degrades by forecast horizon (J+1 = high, J+2-3 = medium, J+4-5 = low)

### Finance Analyzer
- **API**: Yahoo Finance public chart endpoint (no API key needed — always active)
- **Detects**: stock price threshold markets ("Will Tesla reach $X?"), index targets, commodity prices
- **Tickers**: 50+ pre-mapped US stocks, indices (S&P, Nasdaq, Dow), commodities (gold, oil, silver, copper, platinum), forex (EUR/USD, GBP/USD, USD/JPY, USD/CNY, DXY), crypto-adjacent stocks (MSTR, MARA, RIOT)
- **Model**: Log-normal diffusion with drift `P(S>K) = Φ(-z)` where `z = (ln(K/S) - (r-σ²/2)T) / (σ√T)`, r=4.5% annual
- **Volatility**: EWMA (λ=0.94, RiskMetrics) from 3-month daily log returns — weights recent data more heavily
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
- **Draw handling**: 3-way markets (soccer, boxing) detected — draw noted in analysis
- **Confidence**: hybrid score combining bookmaker count AND time horizon (not just books)
- **Cache**: 5 min TTL per sport, expired entries purged automatically, thread-safe

### Common Edge Properties
- **Edge calc**: `estimated_prob - yes_price` (binary: float_prices[0]) → if |edge| > 10%, classified as "edge" tier
- **EV formula**: `(win_prob / buy_price - 1) × 100` where win_prob = est_prob (buy Yes) or 1-est_prob (buy No)
- **Analyzer call order**: weather → finance → earnings → sports → fed/macro → elections → crypto (deterministic, first match wins)

### Earnings Analyzer
- **API**: Yahoo Finance quoteSummary endpoint (no API key needed — always active)
- **Detects**: "Will X beat/miss earnings?" style markets
- **Question parsing**: keyword detection (earnings, revenue, EPS, quarterly, beat, miss, guidance)
- **Model**: Historical beat rate from last 4 quarters of actual vs. estimated EPS
- **Beat direction**: "beat" → use beat_rate, "miss" → 1 - beat_rate
- **Confidence**: degrades by data quality (4Q=high, 2Q=medium, <2Q=low) AND horizon (>30d=low)
- **Cache**: 10 min TTL per ticker, expired entries purged automatically, thread-safe

### Fed/Macro Analyzer
- **API**: FRED (Federal Reserve Economic Data) — free API key (unlimited calls)
- **Env var**: `FRED_API_KEY`
- **Detects**: Fed rate decisions, CPI/inflation, unemployment, GDP/recession markets
- **Series**: DFEDTARU (Fed rate), CPALTT01USM657N (CPI YoY), UNRATE (unemployment), A191RL1Q225SBEA (GDP)
- **Rate model**: Base rates — hold=75%, cut=30%, hike=10% (adjusted with threshold if present)
- **Inflation/Unemployment**: Sigmoid calibration `P = 1/(1+exp(-margin/k))` where margin = data - threshold
- **Recession**: GDP-based — negative=55%, slow(<1%)=30%, healthy=15%
- **Works without key**: rate direction analysis still available (generic base rates)
- **Confidence**: 0-7d = high, 7-30d = medium, 30d+ = low
- **Cache**: 30 min TTL per series, thread-safe

### Crypto Price Analyzer
- **API**: CoinGecko public API (no API key needed — always active)
- **Detects**: crypto price target markets ("Will Bitcoin reach $X?")
- **Coins**: 25+ pre-mapped (bitcoin, ethereum, solana, cardano, dogecoin, etc.)
- **Model**: Log-normal diffusion (same as finance) but with crypto-appropriate higher volatility
- **Volatility**: Derived from 30d price change (CoinGecko market_data)
- **Confidence**: Always medium or low (crypto too volatile for "high" confidence)
- **Crypto gate**: Modified — crypto markets WITH price targets pass through for analysis
- **Cache**: 5 min TTL per coin, thread-safe

### Elections/Polls Analyzer
- **API**: RealClearPolitics public polling endpoint (no API key needed)
- **Detects**: election outcomes, approval ratings, party control markets
- **Figures**: 10+ pre-mapped political figures (Trump, Biden, Harris, DeSantis, Newsom, etc.)
- **Approval model**: Sigmoid `P = 1/(1+exp(-margin/2))` where margin = approval - threshold
- **Party control**: Base rates — Senate R=55%, D=45%; House 50/50
- **Incumbent advantage**: 55% base rate for re-election markets
- **Confidence**: Generally low (elections are inherently uncertain)
- **Cache**: 30 min TTL (polls update slowly), thread-safe

### Future Analyzers (architecture ready)
- **Resolution**: News APIs — detect already-resolved markets not yet settled

## Key Design Decisions
- **Crypto filter**: Regex with ONLY unambiguous tokens — short tokens like `sol`, `eth`, `ada`, `link`, `dot` removed (false positives). `coinbase` removed from regex — handled separately to allow "Coinbase stock" (COIN ticker) through finance analyzer while blocking "Coinbase exchange" (crypto context)
- **Question NOT truncated** in backend — CSS `-webkit-line-clamp` handles display truncation
- **Design system**: Polymarket brand identity — primary blue `#2e5cff`, green `#47c97a` (Yes), red `#ff6464` (No), dark bg `#12151f`, font Open Sauce One / Inter
- **Overround excluded**: sum > 1.0 is the market's margin — excluded entirely (not shown)
- **Fees estimated at ~2%**: arb profit shown net of estimated fees
- **External analyzers optional**: zero-config without API keys, enhanced with keys via env vars. 4 analyzers need no key (Finance, Earnings, Crypto, Elections), 3 optional keys (Weather, Sports, FRED)
- **Weather cache**: 10 min TTL per city, thread-safe, auto-purge expired entries, prevents API key exhaustion
- **Finance cache**: 5 min TTL per ticker, same purge pattern
- **Odds cache**: 5 min TTL per sport, same purge pattern
- **Analyzer call order**: weather → finance → earnings → sports → fed/macro → elections → crypto via `run_analyzers()`. Order is deterministic and documented
- **All API calls synchronous**: acceptable because caches deduplicate requests. Async would add complexity for minimal gain at current scale
- **Finance always active**: Yahoo Finance needs no key — every scan checks stock/index/forex/commodity/earnings markets for free
- **Crypto analyzer active**: CoinGecko needs no key — crypto price target markets analyzed for free
- **Elections analyzer active**: RealClearPolitics + base rates need no key — political markets analyzed for free
- **Earnings cache**: 10 min TTL per ticker, same purge pattern
- **FRED cache**: 30 min TTL per series (macro data moves slowly), same purge pattern
- **Polls cache**: 30 min TTL (polls update slowly), same purge pattern
- **Crypto cache**: 5 min TTL per coin, same purge pattern
- **Gamma API retry**: 3 attempts per page with 1s backoff, logging warnings on partial fetches
- **Cache single-flight**: `_scan_lock` ensures only one `scan()` runs at a time; concurrent requests wait for cached result
- **Proportional fees**: `fee = EST_FEE_PCT × num_outcomes / 2` — multi-outcome arbs pay more fees per outcome
- **Spread warnings**: per-outcome liquidity check (`liquidity / num_outcomes`) warns when execution risk is high
- **Temperature regex**: anchored — prefers `80F` (number+unit) over `5` (bare number). Fallback to number after directional keyword
- **Yahoo Finance logging**: HTTP errors, timeouts logged via `_logger` — no silent failures

## Testing
**ALWAYS run tests before committing**:
```bash
python -m pytest tests/ -v
```

Test file: `tests/test_scanner.py` (361 tests) — covers:
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
- Forex/Commodities: ticker mapping (EUR/USD, GBP, YEN, DXY, copper, platinum), additional stocks (Uber, Disney, Boeing, etc.)
- Earnings analyzer: keyword detection, beat/miss direction, confidence by data quality, mocked Yahoo Finance
- Fed/Macro analyzer: rate decision (hold/cut/hike), CPI threshold sigmoid, unemployment, GDP/recession, FRED series config, mocked FRED API
- Crypto analyzer: ticker mapping (25+ coins), CoinGecko price model, above/below direction, confidence always medium/low, mocked CoinGecko
- Elections analyzer: keyword detection (election, midterm, senate, approval), political figures, party control base rates, approval threshold sigmoid, mocked RCP API
- Updated category extraction: Finance, Crypto, Commodities, Forex, expanded Politics/Sports/Weather
- Updated run_analyzers: 7-analyzer chain order verification, all-None fallback
- Crypto gate: price target markets pass through, generic crypto still filtered
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

## Polymarket Market Taxonomy (researched Feb 2026)
Polymarket has ~7,767 active markets across 10+ top-level categories. Coverage analysis:

### Categories by volume and scanner coverage
| Category | Active Markets | Weekly Volume | Scanner Coverage |
|----------|---------------|---------------|-----------------|
| **Sports** | Largest | ~$721M/wk (39.6%) | Sports analyzer (Odds API) |
| **Politics** | ~1,460 | $2.7B+ total | Elections analyzer (RCP + base rates) |
| **Crypto** | ~2,128 | Large | Crypto price analyzer (CoinGecko) |
| **Finance** | ~253 | Medium | Finance analyzer (Yahoo) + Earnings analyzer |
| **Economy** | ~162 | $228.9M total | Fed/Macro analyzer (FRED) |
| **Pop Culture** | ~330 | $112M total | Not covered (no reliable data source) |
| **Tech** | ~86 AI + more | Medium | Partially via Finance (stock prices) |
| **World/Geopolitics** | 191+ new/month | Growing fast | Not covered (no prediction data source) |
| **Climate/Science** | ~30 weather | Small | Weather analyzer (OWM) |

### Key market structures on Polymarket
- **Yes/No** — most common: "Will X happen by Y date?"
- **Either/Or** — two options: "Will SpaceX or OpenAI IPO first?"
- **Over/Under** — threshold: "Will S&P close above X?"
- **Multi-Outcome** — N candidates: "Who will win Best Actor?" (5+ options)

### Trending tags (Feb 2026)
Trump, Iran, ZachXBT, Tweet Markets, Texas Senate, Cuba, Acquisitions, Tariffs, Oscars, Nepal Election, Midterms, Primaries, Epstein, Daily Temperature, Gov Shutdown, Mexico Cartel War, AI, Derivatives, Equities, Fed, SpaceX, IPOs, Earnings, Venezuela, Ukraine, China, Movies, Global Elections

### Not yet covered (potential future analyzers)
- **Pop Culture / Entertainment**: Oscars, Grammys, box office — no reliable free prediction API
- **Geopolitics**: War, sanctions, territorial — Metaculus or Manifold as potential sources
- **Tweet/Mention markets**: Social media activity predictions — would need Twitter API
- **Resolution detection**: Already-resolved markets not yet settled — News APIs

## Analyzer Prioritization Framework
When adding new analyzers, prioritize by: **(markets covered × data reliability) / implementation effort**

| Priority | Criterion |
|----------|-----------|
| S-tier | Free API + many markets + high reliability (Finance, Earnings, FRED) |
| A-tier | Easy extension of existing infra (adding tickers) or free API + decent market count |
| B-tier | Free API but lower reliability or fewer markets |
| C-tier | Requires paid API or scraping or very few markets |

## Implementation Gotchas Learned
- **Regex word boundaries** with plurals: `\bdemocrat\b` does NOT match "Democrats" — the `s` continues the word. Use `democrats?` pattern
- **"raise" vs "hike"**: In Fed context, users say "raise rates" not "hike rates" — both patterns needed in regex
- **Crypto gate modification**: When allowing crypto price markets through, must check BOTH `has_price_target` AND `has_crypto_ticker` to avoid false positives
- **CoinGecko coin IDs**: Not always the ticker — e.g., "avalanche-2" not "avax", "matic-network" not "polygon"
- **FRED series IDs**: Non-obvious — e.g., `CPALTT01USM657N` for CPI YoY%, `DFEDTARU` for Fed upper target
- **Yahoo Finance earnings**: The `quoteSummary` endpoint with `earningsTrend,earnings` modules gives both historical and forward estimates
- **Test mocking pattern**: Use `@patch("polymarket_scanner._fetch_X")` to mock the data fetch, not the analyzer itself — tests the analysis logic
- **Election regex needs**: `elections?`, `democrats?(?:ic)?`, `republicans?`, `midterms?` — plural forms matter
- **Fed rate regex**: Must handle both "rate cut" (rate + action) AND "cut rates" (action + rate) — two separate alternations
- **Confidence for crypto**: Should NEVER be "high" — crypto is inherently too volatile
- **Base rates for elections**: Useful even without polling data — incumbent advantage ~55%, Senate/House control have historical base rates

## User Preferences
- **Language**: User communicates in French — respond in French for explanations, English for code/docs
- **Scope**: User prefers ambitious implementations ("implémente absolument tout") — don't hold back
- **Testing**: Always run `python -m pytest tests/ -v` before committing (CLAUDE.md rule)

## Common Pitfalls
- The Gamma API field `outcomePrices` is a JSON **string** like `'["0.94","0.06"]'`, not an array — must be parsed with `json.loads()`
- The `outcomes` field is also a JSON string: `'["Yes","No"]'`
- `volume` from the API is **lifetime** volume, not 24h — display context matters
- Multi-outcome markets have >2 prices — top outcome shown as "Yes", rest aggregated as "Others (N-1)"
- `endDate` uses ISO format with `Z` suffix — replaced with `+00:00` for `fromisoformat()`
- `sum(prices) < 1.0` = arb opportunity (guaranteed). `sum > 1.0` = overround (market margin, NOT opportunity)
- Floating-point: `0.49 + 0.50 = 0.99` but `abs(0.99 - 1.0) = 0.010000000000000009` — boundary tests must account for this
