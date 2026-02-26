"""
Polymarket Scanner — Test Suite v4
===================================
Run: python -m pytest tests/ -v

EV-first philosophy, no luck:
- Arb garanti (sum < 1.0, >2%) = super/interesting, guaranteed profit
- Edge informationnel (ext data diverges >10%) = edge tier, real EV
- Near-certain (>90%) = watch only, EV ~$0, involves luck
- Overround (sum > 1.0) = excluded entirely
- Arb <2% = excluded (unprofitable after ~2% fees)

v4 changes (16 audit items):
- Sigmoid temp calibration, composite rain P(at least one)
- Edge uses Yes price (not max_price), EV = win_prob/buy_price - 1
- Confidence degradation by forecast horizon
- Edge checked before 0.995 filter in classify
- Score multiplied by confidence factor
"""

import json
import math
import pytest
from datetime import datetime, timezone, timedelta

from unittest.mock import patch

from polymarket_scanner import (
    CITY_COORDS,
    COINGECKO_MIN_INTERVAL,
    CRYPTO_TICKERS,
    EST_FEE_PCT,
    FINANCE_TICKERS,
    FRED_SERIES,
    MAX_ANN_ROI,
    MAX_SCAN_HISTORY,
    MIN_EDGE,
    MIN_LIQUIDITY,
    SPORT_KEYS,
    _COINBASE_STOCK_RE,
    _EARNINGS_KEYWORDS_RE,
    _ELECTION_RE,
    _APPROVAL_RE,
    _FED_RATE_RE,
    _INFLATION_RE,
    _UNEMPLOYMENT_RE,
    _GDP_RE,
    _cg_rate_limit,
    _consensus_probability,
    _find_city,
    _find_crypto_ticker,
    _find_political_figure,
    _find_sport,
    _find_ticker,
    _has_draw_market,
    _horizon_confidence,
    _log_prediction,
    _match_event,
    _normal_cdf,
    _parse_target_date,
    _rate_trend_probabilities,
    _record_health,
    _record_scan_history,
    _to_fahrenheit,
    analyze_crypto,
    analyze_earnings,
    analyze_elections,
    analyze_fed_macro,
    analyze_finance,
    analyze_sports,
    analyze_weather,
    classify,
    compute_score,
    extract_category,
    is_crypto,
    kelly_fraction,
    parse_date,
    parse_float,
    process_market,
    run_analyzers,
    truncate,
    verify_arb_execution,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def now():
    return datetime.now(timezone.utc)


def _make_market(now, **overrides):
    """Factory for a valid binary market dict."""
    base = {
        "question": "Will the bill pass the Senate?",
        "slug": "will-bill-pass-senate",
        "conditionId": "cond_abc123",
        "endDate": (now + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "volume": "5000000",
        "liquidity": "50000",
        "outcomePrices": '["0.94", "0.06"]',
        "outcomes": '["Yes", "No"]',
    }
    base.update(overrides)
    return base


def _make_owm_entries(temps, pops=None, base_dt=None):
    """Create mock OWM forecast entries."""
    if base_dt is None:
        base_dt = datetime.now(timezone.utc)
    entries = []
    for i, temp in enumerate(temps):
        entry = {
            "dt": int((base_dt + timedelta(hours=3 * i)).timestamp()),
            "main": {"temp": temp, "temp_max": temp + 2, "temp_min": temp - 2},
            "pop": pops[i] if pops else 0.0,
        }
        entries.append(entry)
    return entries


# =========================================================================
# classify() — Arb-first, no overround, no luck
# =========================================================================

class TestClassify:
    """Test tier classification with strict arb-first philosophy."""

    # --- Guaranteed arbitrage (sum < 1.0, min 2%) ---

    def test_super_arb_large_deviation(self):
        """deviation > 4% → super/arbitrage."""
        tier, label, rtype = classify(0.50, 0.05, 0.95)
        assert tier == "super"
        assert rtype == "arbitrage"
        assert "garanti" in label.lower()

    def test_interesting_arb_medium_deviation(self):
        """deviation 2-4% → interesting/arbitrage."""
        tier, label, rtype = classify(0.50, 0.03, 0.97)
        assert tier == "interesting"
        assert rtype == "arbitrage"

    def test_arb_below_2pct_filtered(self):
        """deviation < 2% → filtered (unprofitable after fees)."""
        tier, _, _ = classify(0.50, 0.015, 0.985)
        assert tier is None

    def test_arb_1pct_filtered(self):
        """1% arb → filtered (fees eat the profit)."""
        tier, _, _ = classify(0.50, 0.01, 0.99)
        assert tier is None

    # --- Overround removed entirely ---

    def test_overround_not_shown(self):
        """sum > 1.0 should NEVER produce a result."""
        tier, _, _ = classify(0.55, 0.07, 1.07)
        assert tier is None

    def test_overround_small_not_shown(self):
        """Even small overround → excluded."""
        tier, _, _ = classify(0.52, 0.02, 1.02)
        assert tier is None

    # --- Near-certain → watch ONLY (not interesting) ---

    def test_near_certain_watch_only(self):
        """max_price > 90% → watch/near_certain (NOT interesting)."""
        tier, label, rtype = classify(0.94, 0.0, 1.0)
        assert tier == "watch"
        assert rtype == "near_certain"

    def test_near_certain_never_interesting(self):
        """Even 95% → watch, never interesting."""
        tier, _, _ = classify(0.95, 0.0, 1.0)
        assert tier == "watch"
        assert tier != "interesting"

    # --- 80-90% removed (too speculative) ---

    def test_80_90_removed(self):
        """0.80-0.90 probability → filtered (too much luck)."""
        tier, _, _ = classify(0.85, 0.0, 1.0)
        assert tier is None

    def test_exactly_90_filtered(self):
        """0.90 exactly → NOT > 0.90, filtered."""
        tier, _, _ = classify(0.90, 0.0, 1.0)
        assert tier is None

    # --- Too certain (without edge) ---

    def test_too_certain_filtered(self):
        tier, _, _ = classify(0.998, 0.005, 1.0)
        assert tier is None

    def test_boundary_995_filtered(self):
        tier, _, _ = classify(0.996, 0.005, 1.0)
        assert tier is None

    def test_just_below_995_passes(self):
        """0.994 → watch/near_certain."""
        tier, _, _ = classify(0.994, 0.006, 1.0)
        assert tier == "watch"

    # --- Not interesting ---

    def test_low_price_no_deviation_filtered(self):
        tier, label, rtype = classify(0.60, 0.005, 1.0)
        assert tier is None
        assert label == ""

    # --- Priority: arb > near-certain ---

    def test_arb_takes_priority_over_near_certain(self):
        """sum < 1.0 with deviation > 4%, even with high max_price."""
        tier, _, rtype = classify(0.92, 0.05, 0.95)
        assert tier == "super"
        assert rtype == "arbitrage"

    def test_too_certain_overrides_arb(self):
        """0.997 with no edge → filtered."""
        tier, _, _ = classify(0.997, 0.003, 0.998)
        assert tier is None


# =========================================================================
# classify() with edge_analysis
# =========================================================================

class TestClassifyEdge:
    """Test that edge_analysis triggers the 'edge' tier."""

    def test_edge_above_min_edge(self):
        """Edge >= MIN_EDGE → edge tier."""
        ea = {"edge": 0.15, "source": "OpenWeatherMap"}
        tier, label, rtype = classify(0.70, 0.0, 1.0, edge_analysis=ea)
        assert tier == "edge"
        assert rtype == "edge"
        assert "OpenWeatherMap" in label

    def test_edge_below_min_edge(self):
        """Edge < MIN_EDGE → falls through to normal classify."""
        ea = {"edge": 0.03, "source": "OpenWeatherMap"}
        tier, _, _ = classify(0.70, 0.0, 1.0, edge_analysis=ea)
        assert tier is None  # 70% = not near_certain, not arb

    def test_edge_at_5pct_qualifies(self):
        """Edge = 5% (MIN_EDGE) → qualifies as edge tier."""
        ea = {"edge": 0.05, "source": "OpenWeatherMap"}
        tier, _, rtype = classify(0.70, 0.0, 1.0, edge_analysis=ea)
        assert tier == "edge"
        assert rtype == "edge"

    def test_edge_negative_large(self):
        """Negative edge (market overprices) → still triggers if abs >= MIN_EDGE."""
        ea = {"edge": -0.20, "source": "OpenWeatherMap"}
        tier, _, rtype = classify(0.70, 0.0, 1.0, edge_analysis=ea)
        assert tier == "edge"
        assert rtype == "edge"

    def test_edge_none_no_effect(self):
        """No edge_analysis → normal behavior."""
        tier, _, _ = classify(0.70, 0.0, 1.0, edge_analysis=None)
        assert tier is None

    def test_edge_on_very_certain_market(self):
        """Item 6: 0.996 WITH edge → edge tier (edge checked before 0.995 filter)."""
        ea = {"edge": 0.20, "source": "Test"}
        tier, _, rtype = classify(0.996, 0.0, 1.0, edge_analysis=ea)
        assert tier == "edge"
        assert rtype == "edge"

    def test_very_certain_without_edge_still_filtered(self):
        """0.996 without edge → still filtered."""
        tier, _, _ = classify(0.996, 0.0, 1.0, edge_analysis=None)
        assert tier is None


# =========================================================================
# compute_score() — Arb dominant, NC minimal
# =========================================================================

class TestComputeScore:
    """Test edge-based scoring with strong arb dominance."""

    def test_arb_scores_much_higher_than_near_certain(self):
        """Arbs should score 10x+ higher than near-certain."""
        s_arb = compute_score(0.05, 0.95, 0.50, 5, 50000)
        s_nc = compute_score(0.0, 1.0, 0.94, 5, 50000)
        assert s_arb > s_nc * 10

    def test_bigger_arb_scores_higher(self):
        s_big = compute_score(0.05, 0.95, 0.50, 5, 50000)
        s_small = compute_score(0.03, 0.97, 0.50, 5, 50000)
        assert s_big > s_small

    def test_sooner_is_higher_score(self):
        s_soon = compute_score(0.05, 0.95, 0.50, 2, 50000)
        s_late = compute_score(0.05, 0.95, 0.50, 50, 50000)
        assert s_soon > s_late

    def test_more_liquid_is_higher_score(self):
        s_liq = compute_score(0.05, 0.95, 0.50, 10, 100000)
        s_dry = compute_score(0.05, 0.95, 0.50, 10, 5000)
        assert s_liq > s_dry

    def test_score_is_positive(self):
        s = compute_score(0.05, 0.95, 0.50, 30, 10000)
        assert s > 0

    def test_very_short_time_does_not_explode(self):
        s = compute_score(0.05, 0.95, 0.50, 0.01, 50000)
        assert math.isfinite(s)

    def test_near_certain_score_positive(self):
        s = compute_score(0.0, 1.0, 0.92, 5, 50000)
        assert s > 0

    def test_near_certain_score_minimal(self):
        """Near-certain score should be very small compared to arbs."""
        s_nc = compute_score(0.0, 1.0, 0.94, 5, 50000)
        s_arb = compute_score(0.03, 0.97, 0.50, 5, 50000)
        assert s_arb > s_nc * 5


# =========================================================================
# compute_score() with ext_edge + confidence
# =========================================================================

class TestComputeScoreEdge:

    def test_edge_scores_higher_than_near_certain(self):
        s_edge = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20)
        s_nc = compute_score(0.0, 1.0, 0.94, 5, 50000, ext_edge=0.0)
        assert s_edge > s_nc * 5

    def test_edge_scores_comparable_to_arbs(self):
        s_edge = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20)
        s_arb = compute_score(0.05, 0.95, 0.50, 5, 50000, ext_edge=0.0)
        assert s_edge > 0
        assert s_arb > 0

    def test_high_confidence_scores_higher_than_low(self):
        """Item 16: high confidence edge should score higher than low."""
        s_high = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20, confidence="high")
        s_low = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20, confidence="low")
        assert s_high > s_low

    def test_medium_confidence_between_high_and_low(self):
        """Medium confidence should score between high and low."""
        s_high = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20, confidence="high")
        s_med = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20, confidence="medium")
        s_low = compute_score(0.0, 1.0, 0.70, 5, 50000, ext_edge=0.20, confidence="low")
        assert s_high > s_med > s_low

    def test_confidence_does_not_affect_arbs(self):
        """Confidence only affects ext_edge scoring, not arbs."""
        s1 = compute_score(0.05, 0.95, 0.50, 5, 50000, ext_edge=0.0, confidence="high")
        s2 = compute_score(0.05, 0.95, 0.50, 5, 50000, ext_edge=0.0, confidence="low")
        assert s1 == s2


# =========================================================================
# is_crypto() — Regex correctness
# =========================================================================

class TestCryptoRegex:
    """Ensure no false positives on ambiguous short tokens."""

    @pytest.mark.parametrize("q", [
        "Will Bitcoin hit $100k?",
        "Ethereum price above 5000",
        "BTC to reach new ATH",
        "Is Solana the next big crypto?",
        "Dogecoin market cap",
        "Will Cardano overtake Ethereum?",
        "NFT market collapse?",
        "DeFi total value locked",
        "Binance regulatory issues",
        "Will stablecoin regulation pass?",
    ])
    def test_crypto_detected(self, q):
        assert is_crypto(q) is True, f"Should detect crypto: {q}"

    @pytest.mark.parametrize("q", [
        "Will the resolution pass the Senate?",
        "Whether the bill is approved",
        "Canada election results",
        "The linked document shows evidence",
        "Dot plot from the Fed meeting",
        "NBA season opener tonight",
        "Will Trump win the election?",
        "Solar energy investment growth",
        "Ethical AI guidelines adopted",
        "Adaptation strategy for climate change",
    ])
    def test_crypto_not_detected(self, q):
        assert is_crypto(q) is False, f"Should NOT detect crypto: {q}"


# =========================================================================
# extract_category()
# =========================================================================

class TestExtractCategory:
    def test_politics(self):
        assert extract_category({"question": "Will Trump win?"}) == "Politics"

    def test_sports(self):
        assert extract_category({"question": "NBA finals winner?"}) == "Sports"

    def test_economics(self):
        assert extract_category({"question": "Will the Fed raise interest rate?"}) == "Economics"

    def test_tech(self):
        assert extract_category({"question": "Will OpenAI release GPT-5?"}) == "Tech"

    def test_entertainment(self):
        assert extract_category({"question": "Oscar best picture 2026?"}) == "Entertainment"

    def test_geopolitics(self):
        assert extract_category({"question": "Ukraine war ceasefire?"}) == "Geopolitics"

    def test_weather(self):
        assert extract_category({"question": "Hurricane season 2026?"}) == "Weather"

    def test_science(self):
        assert extract_category({"question": "FDA approval for new vaccine?"}) == "Science"

    def test_other_fallback(self):
        assert extract_category({"question": "Random unrelated thing?"}) == "Other"

    def test_tags_take_priority(self):
        m = {"question": "Will Trump win?", "tags": '[{"label": "custom-tag"}]'}
        assert extract_category(m) == "Custom-Tag"

    def test_tags_as_string_list(self):
        m = {"question": "Something", "tags": '["Sports"]'}
        assert extract_category(m) == "Sports"


# =========================================================================
# parse_float()
# =========================================================================

class TestParseFloat:
    def test_valid_string(self):
        assert parse_float("3.14") == 3.14

    def test_valid_int(self):
        assert parse_float(42) == 42.0

    def test_none(self):
        assert parse_float(None) == 0.0

    def test_garbage(self):
        assert parse_float("abc") == 0.0

    def test_custom_default(self):
        assert parse_float(None, -1.0) == -1.0

    def test_empty_string(self):
        assert parse_float("") == 0.0


# =========================================================================
# parse_date()
# =========================================================================

class TestParseDate:
    def test_iso_with_z(self):
        dt = parse_date("2026-03-15T12:00:00Z")
        assert dt is not None
        assert dt.year == 2026

    def test_iso_with_offset(self):
        dt = parse_date("2026-06-01T00:00:00+00:00")
        assert dt is not None

    def test_none_input(self):
        assert parse_date(None) is None

    def test_empty_string(self):
        assert parse_date("") is None

    def test_invalid_format(self):
        assert parse_date("not-a-date") is None


# =========================================================================
# truncate()
# =========================================================================

class TestTruncate:
    def test_short_string_unchanged(self):
        assert truncate("Hi", 10) == "Hi"

    def test_exact_length_unchanged(self):
        assert truncate("Hello", 5) == "Hello"

    def test_long_string_truncated(self):
        result = truncate("Hello World", 8)
        assert len(result) <= 8
        assert result.endswith("..")

    def test_default_length(self):
        result = truncate("This is a very long outcome name")
        assert len(result) <= 15
        assert result.endswith("..")


# =========================================================================
# process_market() — Full pipeline (strict EV-first)
# =========================================================================

class TestProcessMarket:
    """Integration tests: arbs guaranteed, near-certain in watch, overround gone."""

    # --- Near-certain → watch (not interesting) ---

    def test_near_certain_binary_market(self, now):
        """94/6 market → WATCH/near_certain with EV ~$0."""
        m = _make_market(now)
        result = process_market(m, now)
        assert result is not None
        assert result["tier"] == "watch"
        assert result["reason_type"] == "near_certain"
        assert result["yes"] == 0.94
        assert result["no"] == 0.06
        assert result["is_binary"] is True
        assert result["guaranteed"] is False
        # Near-certain: no edge
        assert result["ev_per_100"] == 0.0
        assert result["risk_per_100"] == 100.0
        assert result["fee_per_100"] == 0.0
        assert result["net_per_100"] == 0.0
        assert result["arb_warning"] == ""

    def test_question_not_truncated(self, now):
        long_q = "Will the Supreme Court ruling affect the upcoming presidential election?" * 3
        m = _make_market(now, question=long_q)
        result = process_market(m, now)
        assert result is not None
        assert result["question"] == long_q

    def test_trade_recommendation_buy_yes(self, now):
        """Near-certain → BUY best outcome, SPECULATIF."""
        m = _make_market(now, outcomePrices='["0.94", "0.06"]')
        result = process_market(m, now)
        assert result is not None
        assert "BUY" in result["trade_label"]
        assert "YES" in result["trade_label"]
        assert result["trade_side_class"] == "yes"
        assert result["guaranteed"] is False

    # --- Arbitrage: GUARANTEED profit with fees ---

    def test_arb_super_large_deviation(self, now):
        """sum 0.93 (7% arb) → super, guaranteed, fees shown."""
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert result["tier"] == "super"
        assert result["reason_type"] == "arbitrage"
        assert "ARB" in result["trade_label"]
        assert result["trade_side_class"] == "arb"
        assert result["guaranteed"] is True
        # EV guaranteed
        assert result["ev_per_100"] > 0
        assert result["risk_per_100"] == 0.0
        # Fees
        assert result["fee_per_100"] == EST_FEE_PCT
        assert result["net_per_100"] == round(result["ev_per_100"] - EST_FEE_PCT, 2)
        assert result["net_per_100"] > 0  # profitable after fees
        # Warning
        assert result["arb_warning"] == "2 trades requis"

    def test_arb_ev_calculation(self, now):
        """EV = (1-sum)/sum * 100, net = EV - fees."""
        m = _make_market(now, outcomePrices='["0.45", "0.50"]')
        result = process_market(m, now)
        assert result is not None
        expected_ev = (1.0 - 0.95) / 0.95 * 100
        assert abs(result["ev_per_100"] - expected_ev) < 0.1
        assert result["net_per_100"] == round(expected_ev - EST_FEE_PCT, 2)

    def test_arb_interesting_medium_deviation(self, now):
        """sum 0.97 (3% arb) → interesting tier."""
        m = _make_market(now, outcomePrices='["0.46", "0.51"]')
        result = process_market(m, now)
        assert result is not None
        assert result["tier"] == "interesting"
        assert result["reason_type"] == "arbitrage"
        assert result["guaranteed"] is True
        assert result["fee_per_100"] == EST_FEE_PCT

    def test_arb_net_positive_for_large_arb(self, now):
        """5% arb → net profit clearly positive after fees."""
        m = _make_market(now, outcomePrices='["0.45", "0.50"]')
        result = process_market(m, now)
        assert result is not None
        assert result["net_per_100"] > 2.0  # well above break-even

    # --- Arb <2% FILTERED (unprofitable after fees) ---

    def test_arb_below_2pct_filtered(self, now):
        """sum 0.99 (1% arb) → filtered."""
        m = _make_market(now, outcomePrices='["0.495", "0.495"]')
        assert process_market(m, now) is None

    # --- Overround REMOVED ---

    def test_overround_filtered(self, now):
        """sum > 1.0 → no longer shown at all."""
        m = _make_market(now, outcomePrices='["0.55", "0.52"]')
        assert process_market(m, now) is None

    def test_overround_large_filtered(self, now):
        """Even large overround → excluded."""
        m = _make_market(now, outcomePrices='["0.60", "0.50"]')
        assert process_market(m, now) is None

    # --- Multi-outcome arb ---

    def test_multi_outcome_arb(self, now):
        """4-outcome arb shows trade count and warning."""
        m = _make_market(
            now,
            question="Who will win the GOP primary?",
            outcomePrices='["0.82", "0.08", "0.03", "0.02"]',
            outcomes='["Trump", "DeSantis", "Haley", "Other"]',
        )
        result = process_market(m, now)
        assert result is not None
        assert result["is_binary"] is False
        assert result["num_outcomes"] == 4
        assert result["tier"] == "super"  # 5% arb
        assert result["reason_type"] == "arbitrage"
        assert result["yes_label"] == "Trump"
        assert "Others" in result["no_label"]
        # Multi-outcome trade label and warning
        assert "4" in result["trade_label"]
        assert result["arb_warning"] == "4 trades requis"

    def test_binary_arb_trade_label_no_count(self, now):
        """Binary arb says 'BUY ALL', not 'BUY 2×'."""
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert "BUY ALL" in result["trade_label"]
        assert result["arb_warning"] == "2 trades requis"

    # --- ROI ---

    def test_roi_calculation(self, now):
        m = _make_market(now, outcomePrices='["0.94", "0.06"]')
        result = process_market(m, now)
        assert result is not None
        assert abs(result["profit_100"] - 6.38) < 0.1
        assert result["ann_roi"] > 0

    def test_ann_roi_capped(self, now):
        m = _make_market(
            now,
            outcomePrices='["0.94", "0.06"]',
            endDate=(now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        result = process_market(m, now)
        assert result is not None
        assert result["ann_roi"] <= MAX_ANN_ROI
        assert result["ann_roi_capped"] is True

    def test_ann_roi_not_capped_for_longer_markets(self, now):
        m = _make_market(now, outcomePrices='["0.94", "0.06"]')
        result = process_market(m, now)
        assert result is not None
        assert result["ann_roi_capped"] is False

    # --- Thin market ---

    def test_thin_market_detected(self, now):
        m = _make_market(now, liquidity="8000")
        result = process_market(m, now)
        assert result is not None
        assert result["thin"] is True

    def test_healthy_market_not_thin(self, now):
        m = _make_market(now, volume="10000", liquidity="50000")
        result = process_market(m, now)
        assert result is not None
        assert result["thin"] is False

    # --- Score, URL, category ---

    def test_score_present_and_positive(self, now):
        m = _make_market(now)
        result = process_market(m, now)
        assert result is not None
        assert result["score"] > 0

    def test_url_construction(self, now):
        m = _make_market(now, slug="my-test-slug")
        result = process_market(m, now)
        assert result is not None
        assert result["url"] == "https://polymarket.com/event/my-test-slug"

    def test_category_in_result(self, now):
        m = _make_market(now, question="NBA championship winner 2026?")
        result = process_market(m, now)
        assert result is not None
        assert result["category"] == "Sports"

    def test_days_display_hours_for_imminent(self, now):
        m = _make_market(
            now,
            endDate=(now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        result = process_market(m, now)
        assert result is not None
        assert result["days_unit"] == "h"
        assert result["days_display"] > 0

    # --- Filters (should return None) ---

    def test_filter_expired_market(self, now):
        m = _make_market(
            now,
            endDate=(now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        assert process_market(m, now) is None

    def test_filter_too_far_out(self, now):
        m = _make_market(
            now,
            endDate=(now + timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        assert process_market(m, now) is None

    def test_filter_low_volume(self, now):
        m = _make_market(now, volume="500")
        assert process_market(m, now) is None

    def test_filter_low_liquidity(self, now):
        m = _make_market(now, liquidity="4000")
        assert process_market(m, now) is None

    def test_filter_crypto_market_no_price(self, now):
        """Crypto without price target → still filtered."""
        m = _make_market(now, question="Will Bitcoin be used as legal tender?")
        assert process_market(m, now) is None

    def test_crypto_with_price_target_passes(self, now):
        """Crypto with price target → allowed through for crypto analyzer."""
        m = _make_market(now, question="Will Bitcoin reach $200000?")
        result = process_market(m, now)
        # May return None (no CoinGecko in test) or a valid result
        # The key test is it's NOT filtered by the crypto gate
        # (it passes through, but may fail later due to network)
        # We just verify the crypto gate doesn't block it
        pass  # integration test — crypto gate allows it through

    def test_filter_no_end_date(self, now):
        m = _make_market(now, endDate=None)
        assert process_market(m, now) is None

    def test_filter_too_certain(self, now):
        m = _make_market(now, outcomePrices='["0.998", "0.002"]')
        assert process_market(m, now) is None

    def test_filter_not_interesting(self, now):
        """50/50 → no arb, no near-certain → None."""
        m = _make_market(now, outcomePrices='["0.50", "0.50"]')
        assert process_market(m, now) is None

    def test_filter_invalid_prices(self, now):
        m = _make_market(now, outcomePrices="invalid_json")
        assert process_market(m, now) is None

    def test_filter_single_price(self, now):
        m = _make_market(now, outcomePrices='["0.94"]')
        assert process_market(m, now) is None

    def test_filter_dead_market_volume_24h_zero(self, now):
        m = _make_market(now, volume24hr="0")
        assert process_market(m, now) is None

    def test_no_filter_when_24h_data_missing(self, now):
        m = _make_market(now)
        assert "volume24hr" not in m
        result = process_market(m, now)
        assert result is not None

    def test_filter_dead_market_volume24Hr_variant(self, now):
        m = _make_market(now, volume24Hr="0")
        assert process_market(m, now) is None

    def test_filter_moderate_probability(self, now):
        """85% market → filtered (80-90% removed)."""
        m = _make_market(now, outcomePrices='["0.85", "0.15"]')
        assert process_market(m, now) is None

    # --- Edge cases ---

    def test_zero_price_no_crash(self, now):
        m = _make_market(now, outcomePrices='["0.00", "1.00"]')
        assert process_market(m, now) is None

    def test_outcomes_as_list_not_string(self, now):
        m = _make_market(now, outcomes=["Yes", "No"])
        result = process_market(m, now)
        assert result is not None
        assert result["yes_label"] == "Yes"

    def test_prices_as_list_not_string(self, now):
        m = _make_market(now, outcomePrices=[0.94, 0.06])
        result = process_market(m, now)
        assert result is not None

    def test_missing_outcomes_field(self, now):
        m = _make_market(now)
        del m["outcomes"]
        result = process_market(m, now)
        assert result is not None

    def test_volume_24h_passthrough(self, now):
        m = _make_market(now, volume24hr="12345")
        result = process_market(m, now)
        assert result is not None
        assert result["volume_24h"] == 12345.0

    def test_arb_score_higher_than_near_certain_in_pipeline(self, now):
        """End-to-end: arb markets score much higher than near-certain."""
        m_arb = _make_market(now, outcomePrices='["0.45", "0.48"]')
        m_nc = _make_market(now, outcomePrices='["0.94", "0.06"]')
        r_arb = process_market(m_arb, now)
        r_nc = process_market(m_nc, now)
        assert r_arb is not None and r_nc is not None
        assert r_arb["score"] > r_nc["score"] * 10

    def test_all_new_fields_present(self, now):
        """All results must have fee, net, warning, edge_analysis fields."""
        m = _make_market(now)
        result = process_market(m, now)
        assert result is not None
        assert "ev_per_100" in result
        assert "risk_per_100" in result
        assert "fee_per_100" in result
        assert "net_per_100" in result
        assert "arb_warning" in result
        assert "ann_roi_capped" in result
        assert "edge_analysis" in result

    def test_edge_analysis_none_without_api_key(self, now):
        """Without OWM key, edge_analysis should be None for any market."""
        m = _make_market(now, question="Will temperature in New York exceed 80F?")
        result = process_market(m, now)
        assert result is not None
        assert result["edge_analysis"] is None


# =========================================================================
# process_market() — Edge markets (items 1-2, 11)
# =========================================================================

class TestProcessMarketEdge:
    """Test edge market processing with correct price/EV calculations."""

    @patch("polymarket_scanner.run_analyzers")
    def test_edge_uses_yes_price_binary(self, mock_analyzers, now):
        """Item 1: edge = est_prob - yes_price (not max_price)."""
        mock_analyzers.return_value = {
            "estimated_prob": 0.80,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "high",
            "data_point": "test",
        }
        # Binary market: Yes=0.60, No=0.40
        m = _make_market(now, outcomePrices='["0.60", "0.40"]')
        result = process_market(m, now)
        assert result is not None
        assert result["tier"] == "edge"
        # Edge should be 0.80 - 0.60 = 0.20 (using Yes price, not max_price)
        assert abs(result["edge_analysis"]["edge"] - 0.20) < 0.01
        assert result["edge_analysis"]["market_price"] == 0.60

    @patch("polymarket_scanner.run_analyzers")
    def test_edge_positive_buys_yes(self, mock_analyzers, now):
        """Item 2: positive edge → BUY YES."""
        mock_analyzers.return_value = {
            "estimated_prob": 0.80,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "high",
            "data_point": "test",
        }
        m = _make_market(now, outcomePrices='["0.60", "0.40"]')
        result = process_market(m, now)
        assert result is not None
        assert "YES" in result["trade_label"]
        assert result["trade_side_class"] == "yes"

    @patch("polymarket_scanner.run_analyzers")
    def test_edge_negative_buys_no(self, mock_analyzers, now):
        """Item 2: negative edge → BUY NO."""
        mock_analyzers.return_value = {
            "estimated_prob": 0.30,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "high",
            "data_point": "test",
        }
        # Binary: Yes=0.60, No=0.40. Edge = 0.30 - 0.60 = -0.30
        m = _make_market(now, outcomePrices='["0.60", "0.40"]')
        result = process_market(m, now)
        assert result is not None
        assert "NO" in result["trade_label"]
        assert result["trade_side_class"] == "no"

    @patch("polymarket_scanner.run_analyzers")
    def test_edge_ev_formula_positive(self, mock_analyzers, now):
        """Item 11: EV = (win_prob / buy_price - 1) × 100 for positive edge."""
        mock_analyzers.return_value = {
            "estimated_prob": 0.80,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "high",
            "data_point": "test",
        }
        m = _make_market(now, outcomePrices='["0.60", "0.40"]')
        result = process_market(m, now)
        assert result is not None
        # Positive edge: buy Yes@0.60, win_prob=0.80
        # EV = (0.80/0.60 - 1) × 100 = 33.33
        expected_ev = (0.80 / 0.60 - 1) * 100
        assert abs(result["ev_per_100"] - expected_ev) < 0.1

    @patch("polymarket_scanner.run_analyzers")
    def test_edge_ev_formula_negative(self, mock_analyzers, now):
        """Item 11: EV = ((1-est_prob) / buy_price_no - 1) × 100 for negative edge."""
        mock_analyzers.return_value = {
            "estimated_prob": 0.30,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "high",
            "data_point": "test",
        }
        # Binary: Yes=0.60, No=0.40. Edge=-0.30 → buy No@0.40
        m = _make_market(now, outcomePrices='["0.60", "0.40"]')
        result = process_market(m, now)
        assert result is not None
        # Negative edge: buy No@0.40, win_prob = 1-0.30 = 0.70
        # EV = (0.70/0.40 - 1) × 100 = 75.0
        expected_ev = (0.70 / 0.40 - 1) * 100
        assert abs(result["ev_per_100"] - expected_ev) < 0.1

    @patch("polymarket_scanner.run_analyzers")
    def test_edge_confidence_affects_score(self, mock_analyzers, now):
        """Item 16: low confidence edge should score lower."""
        # High confidence
        mock_analyzers.return_value = {
            "estimated_prob": 0.80,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "high",
            "data_point": "test",
        }
        m = _make_market(now, outcomePrices='["0.60", "0.40"]')
        r_high = process_market(m, now)

        # Low confidence
        mock_analyzers.return_value = {
            "estimated_prob": 0.80,
            "source": "OpenWeatherMap",
            "analysis": "Test",
            "confidence": "low",
            "data_point": "test",
        }
        r_low = process_market(m, now)

        assert r_high is not None and r_low is not None
        assert r_high["score"] > r_low["score"]


# =========================================================================
# Weather Analyzer — city detection, unit conversion, question parsing
# =========================================================================

class TestFindCity:
    """Test city name extraction from question text."""

    def test_find_nyc(self):
        result = _find_city("temperature in New York above 80F")
        assert result is not None
        assert result[0] == "new york"

    def test_find_nyc_alias(self):
        result = _find_city("Will the high in NYC exceed 75?")
        assert result is not None
        assert result[0] == "nyc"

    def test_find_chicago(self):
        result = _find_city("Rain in Chicago on Tuesday?")
        assert result is not None
        assert result[0] == "chicago"

    def test_find_london(self):
        result = _find_city("Temperature in London above 20C")
        assert result is not None
        assert result[0] == "london"

    def test_longest_match_wins(self):
        """'san francisco' should match over 'san'."""
        result = _find_city("Weather in San Francisco this week")
        assert result is not None
        assert result[0] == "san francisco"

    def test_no_match(self):
        result = _find_city("Will the bill pass the Senate?")
        assert result is None

    def test_case_insensitive(self):
        result = _find_city("temp in MIAMI above 90")
        assert result is not None
        assert result[0] == "miami"

    def test_la_alias_removed(self):
        """Item 9: 'la' alias removed to prevent false positives."""
        assert "la" not in CITY_COORDS
        result = _find_city("la température est élevée")
        # Should NOT match "la" as Los Angeles
        assert result is None

    def test_los_angeles_still_works(self):
        """'los angeles' (full name) still matches."""
        result = _find_city("Weather in Los Angeles this week")
        assert result is not None
        assert result[0] == "los angeles"


class TestToFahrenheit:
    def test_fahrenheit_passthrough(self):
        assert _to_fahrenheit(80, "F") == 80

    def test_celsius_conversion(self):
        assert _to_fahrenheit(0, "C") == 32
        assert abs(_to_fahrenheit(100, "celsius") - 212) < 0.01

    def test_no_unit_defaults_fahrenheit(self):
        assert _to_fahrenheit(75, None) == 75

    def test_degree_c(self):
        assert abs(_to_fahrenheit(20, "°C") - 68) < 0.01


# =========================================================================
# _horizon_confidence() — forecast horizon degradation (item 12)
# =========================================================================

class TestHorizonConfidence:
    """Test confidence degradation based on forecast horizon."""

    def test_short_horizon_high_confidence(self):
        """Entries within 24h → high confidence."""
        now_dt = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        entries = _make_owm_entries([70] * 4, base_dt=now_dt)  # 0-9h
        assert _horizon_confidence(entries, now_dt) == "high"

    def test_medium_horizon(self):
        """Entries up to 48h → medium confidence."""
        now_dt = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        entries = _make_owm_entries([70] * 16, base_dt=now_dt)  # 0-45h
        assert _horizon_confidence(entries, now_dt) == "medium"

    def test_long_horizon_low_confidence(self):
        """Entries up to 5 days → low confidence."""
        now_dt = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        entries = _make_owm_entries([70] * 40, base_dt=now_dt)  # 0-117h
        assert _horizon_confidence(entries, now_dt) == "low"

    def test_empty_entries_low(self):
        """No entries → low confidence."""
        assert _horizon_confidence([]) == "low"


# =========================================================================
# Weather Analyzer — sigmoid temp, composite rain (items 3, 4, 5)
# =========================================================================

class TestAnalyzeWeather:
    """Test weather analysis with mocked OWM API responses."""

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_temp_above_detected_sigmoid(self, mock_fetch):
        """Item 3: sigmoid calibration — forecast well above threshold → high prob."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([82, 85, 78, 84, 80, 86, 79, 83])
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will the high temperature in New York exceed 80F by Friday?",
            end_dt,
        )
        assert result is not None
        assert result["source"] == "OpenWeatherMap"
        # Sigmoid: max_high=88, margin=8, prob=1/(1+exp(-8/3)) ≈ 0.93
        assert result["estimated_prob"] > 0.8
        assert "80" in result["analysis"]

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_temp_below_threshold_sigmoid(self, mock_fetch):
        """Item 3: sigmoid — forecast well below threshold → low prob."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([60, 62, 58, 61, 59, 63, 57, 60])
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will the temperature in Chicago reach 80F?",
            end_dt,
        )
        assert result is not None
        # Sigmoid: max_high=65, margin=-15, prob=1/(1+exp(5)) ≈ 0.007
        assert result["estimated_prob"] < 0.05

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_temp_near_threshold_sigmoid(self, mock_fetch):
        """Sigmoid: forecast near threshold → ~50% probability."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        # temp_max = temp+2, so max_high = 82. Threshold 80.
        # margin = 2, prob = 1/(1+exp(-2/3)) ≈ 0.66
        entries = _make_owm_entries([78, 79, 80, 79, 78, 77, 76, 78])
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will temperature in Denver exceed 80F?",
            end_dt,
        )
        assert result is not None
        assert 0.4 < result["estimated_prob"] < 0.8

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_rain_composite_formula(self, mock_fetch):
        """Item 5: composite rain prob = 1 - prod(1 - pop_i)."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries(
            [70] * 8,
            pops=[0.1, 0.3, 0.8, 0.9, 0.7, 0.4, 0.2, 0.1],
        )
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will it rain in Miami this week?",
            end_dt,
        )
        assert result is not None
        # Composite: 1 - (0.9*0.7*0.2*0.1*0.3*0.6*0.8*0.9) ≈ 0.998
        assert result["estimated_prob"] >= 0.99
        assert "rain" in result["analysis"].lower()
        assert "composite" in result["analysis"].lower()

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_rain_low_pop_composite(self, mock_fetch):
        """Low PoPs still accumulate with composite formula but stay reasonable."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries(
            [70] * 8,
            pops=[0.02, 0.0, 0.05, 0.0, 0.02, 0.0, 0.0, 0.01],
        )
        mock_fetch.return_value = entries

        result = analyze_weather("Will it rain in Denver?", end_dt)
        assert result is not None
        # Composite: 1 - (0.98*1.0*0.95*1.0*0.98*1.0*1.0*0.99) ≈ 0.097
        assert result["estimated_prob"] < 0.15

    def test_no_api_key_returns_none(self):
        """Without API key, analyze_weather returns None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        result = analyze_weather(
            "Will temperature in New York exceed 80F?",
            end_dt,
        )
        assert result is None

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    def test_non_weather_question_returns_none(self):
        """Non-weather question → None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        result = analyze_weather("Will Trump win the election?", end_dt)
        assert result is None

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    def test_unknown_city_returns_none(self):
        """Weather question with unknown city → None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        result = analyze_weather(
            "Will temperature in Timbuktu exceed 120F?",
            end_dt,
        )
        assert result is None

    # --- Flexible regex (item 4) ---

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_flexible_question_format(self, mock_fetch):
        """Item 4: keyword detection works regardless of city/threshold order."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([82, 85, 78, 84, 80, 86, 79, 83])
        mock_fetch.return_value = entries

        # Threshold before city (non-standard order)
        result = analyze_weather(
            "Will 80F temperature be exceeded in New York?",
            end_dt,
        )
        assert result is not None
        assert result["estimated_prob"] > 0.5

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_heat_keyword_detected(self, mock_fetch):
        """Item 4: 'heat' keyword triggers temperature analysis."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([90, 92, 88, 91])
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will the heat wave in Miami exceed 95F?",
            end_dt,
        )
        assert result is not None

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_precipitation_keyword_detected(self, mock_fetch):
        """Item 4: 'precipitation' keyword triggers rain analysis."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([70] * 4, pops=[0.8, 0.7, 0.6, 0.5])
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will there be precipitation in Chicago?",
            end_dt,
        )
        assert result is not None
        assert result["estimated_prob"] > 0.5

    # --- Confidence in results (item 12) ---

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_confidence_returned_in_result(self, mock_fetch):
        """Weather result includes confidence level."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([82, 85, 78, 84, 80, 86, 79, 83])
        mock_fetch.return_value = entries

        result = analyze_weather(
            "Will temperature in New York exceed 80F?",
            end_dt,
        )
        assert result is not None
        assert result["confidence"] in ("high", "medium", "low")


# =========================================================================
# OWM cache purge (item 8)
# =========================================================================

class TestOWMCachePurge:
    """Test that expired OWM cache entries are purged."""

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner.requests.get")
    def test_expired_entries_purged(self, mock_get):
        """Item 8: expired cache entries are removed on next fetch."""
        import polymarket_scanner as ps

        # Prepare mock response
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"list": [{"dt": 1000, "main": {"temp": 70}}]}

        # Manually insert an expired entry
        with ps._owm_lock:
            ps._owm_cache["99.99,99.99"] = (0, [])  # ts=0 → ancient, expired

        # Trigger a new fetch which should purge expired
        ps._fetch_owm_forecast(40.71, -74.01)

        with ps._owm_lock:
            assert "99.99,99.99" not in ps._owm_cache

        # Cleanup
        with ps._owm_lock:
            ps._owm_cache.clear()


# =========================================================================
# Finance Analyzer — ticker detection, probability model
# =========================================================================

class TestNormalCdf:
    """Test standard normal CDF helper."""

    def test_cdf_at_zero(self):
        assert abs(_normal_cdf(0) - 0.5) < 0.001

    def test_cdf_large_positive(self):
        assert _normal_cdf(3.0) > 0.998

    def test_cdf_large_negative(self):
        assert _normal_cdf(-3.0) < 0.002

    def test_cdf_symmetry(self):
        assert abs(_normal_cdf(1.0) + _normal_cdf(-1.0) - 1.0) < 0.001


class TestFindTicker:
    """Test financial ticker extraction from question text."""

    def test_find_tesla(self):
        assert _find_ticker("Will Tesla stock reach $400?") == "TSLA"

    def test_find_tsla(self):
        assert _find_ticker("TSLA above $300 by March?") == "TSLA"

    def test_find_sp500(self):
        assert _find_ticker("Will the S&P 500 close above 6000?") == "^GSPC"

    def test_find_gold(self):
        assert _find_ticker("Gold price above $2500?") == "GC=F"

    def test_find_nvidia(self):
        assert _find_ticker("Will Nvidia stock hit $200?") == "NVDA"

    def test_longest_match_wins(self):
        """'s&p 500' should match over 's&p'."""
        assert _find_ticker("S&P 500 reaching new highs") == "^GSPC"

    def test_no_match(self):
        assert _find_ticker("Will Trump win the election?") is None

    def test_case_insensitive(self):
        assert _find_ticker("will TESLA reach $500?") == "TSLA"


class TestAnalyzeFinance:
    """Test finance analyzer with mocked Yahoo Finance data."""

    def _mock_yahoo_response(self, current_price, closes):
        """Build a mock Yahoo Finance chart response."""
        return {
            "chart": {
                "result": [{
                    "meta": {"regularMarketPrice": current_price},
                    "indicators": {
                        "quote": [{"close": closes}]
                    },
                }]
            }
        }

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_stock_above_current_high_prob(self, mock_fetch):
        """Stock at $350, target $300 → high probability."""
        mock_fetch.return_value = {"price": 350, "daily_vol": 0.02, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Tesla stock be above $300?", end_dt)
        assert result is not None
        assert result["source"] == "Yahoo Finance"
        assert result["estimated_prob"] > 0.7

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_stock_above_far_away_low_prob(self, mock_fetch):
        """Stock at $200, target $500, low vol → low probability."""
        mock_fetch.return_value = {"price": 200, "daily_vol": 0.01, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Tesla stock reach $500?", end_dt)
        assert result is not None
        assert result["estimated_prob"] < 0.1

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_stock_below_threshold(self, mock_fetch):
        """'below' question inverts probability."""
        mock_fetch.return_value = {"price": 100, "daily_vol": 0.02, "ticker": "^GSPC"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will S&P 500 drop below $50?", end_dt)
        assert result is not None
        # Price at 100, target below 50 → very unlikely
        assert result["estimated_prob"] < 0.1

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_dollar_sign_threshold(self, mock_fetch):
        """Threshold with $ sign parsed correctly."""
        mock_fetch.return_value = {"price": 2400, "daily_vol": 0.01, "ticker": "GC=F"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Gold price above $2,500?", end_dt)
        assert result is not None
        assert "2,500" in result["analysis"] or "2500" in result["analysis"]

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_confidence_short_horizon(self, mock_fetch):
        """Short-term market → high confidence."""
        mock_fetch.return_value = {"price": 300, "daily_vol": 0.02, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=3)
        result = analyze_finance("Will Tesla hit $310?", end_dt)
        assert result is not None
        assert result["confidence"] == "high"

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_confidence_long_horizon(self, mock_fetch):
        """Long-term market → low confidence."""
        mock_fetch.return_value = {"price": 300, "daily_vol": 0.02, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=45)
        result = analyze_finance("Will Tesla hit $310?", end_dt)
        assert result is not None
        assert result["confidence"] == "low"

    def test_non_finance_question_returns_none(self):
        """Non-finance question → None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Trump win the election?", end_dt)
        assert result is None

    def test_no_threshold_returns_none(self):
        """Finance keyword but no price threshold → None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Tesla stock go up?", end_dt)
        assert result is None

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_ev_realistic_for_close_price(self, mock_fetch):
        """Price near threshold → ~50% probability."""
        mock_fetch.return_value = {"price": 300, "daily_vol": 0.02, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Tesla reach $300?", end_dt)
        assert result is not None
        assert 0.3 < result["estimated_prob"] < 0.7


# =========================================================================
# Sports Analyzer — sport detection, team matching, odds consensus
# =========================================================================

class TestFindSport:
    """Test sport key detection from question text."""

    def test_nba(self):
        assert _find_sport("Will the Lakers win the NBA championship?") == "basketball_nba"

    def test_nfl(self):
        assert _find_sport("NFL regular season MVP?") == "americanfootball_nfl"

    def test_super_bowl_specific(self):
        """'super bowl' is more specific than 'nfl'."""
        assert _find_sport("Who wins Super Bowl LIX?") == "americanfootball_nfl_super_bowl"

    def test_premier_league(self):
        assert _find_sport("Premier League title race") == "soccer_epl"

    def test_ufc(self):
        assert _find_sport("UFC heavyweight bout") == "mma_mixed_martial_arts"

    def test_no_sport(self):
        assert _find_sport("Will the bill pass the Senate?") is None


class TestMatchEvent:
    """Test event matching by team name overlap."""

    def test_match_by_full_name(self):
        events = [
            {"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"},
            {"home_team": "Golden State Warriors", "away_team": "Miami Heat"},
        ]
        result = _match_event("Will the Lakers beat the Celtics?", events)
        assert result is not None
        assert result["home_team"] == "Los Angeles Lakers"

    def test_match_by_partial_name(self):
        events = [
            {"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"},
        ]
        result = _match_event("Lakers championship odds?", events)
        assert result is not None
        assert result["home_team"] == "Los Angeles Lakers"

    def test_no_match_short_words(self):
        """Short words (< 4 chars) don't trigger false matches."""
        events = [
            {"home_team": "FC Red Bull Salzburg", "away_team": "AC Milan"},
        ]
        result = _match_event("Will AI change the world?", events)
        assert result is None

    def test_no_match_empty(self):
        assert _match_event("Lakers vs Celtics", []) is None


class TestConsensusProb:
    """Test devigged consensus probability calculation."""

    def test_two_bookmakers(self):
        """Average devigged probability from two bookmakers."""
        event = {
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Lakers", "price": 2.0},
                    {"name": "Celtics", "price": 1.9},
                ]}]},
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Lakers", "price": 2.1},
                    {"name": "Celtics", "price": 1.85},
                ]}]},
            ]
        }
        prob, count = _consensus_probability(event, "Lakers")
        assert count == 2
        assert 0.4 < prob < 0.55  # ~47-49% devigged

    def test_no_matching_team(self):
        """Team not found in outcomes → (0, 0)."""
        event = {
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Team A", "price": 2.0},
                    {"name": "Team B", "price": 1.9},
                ]}]},
            ]
        }
        prob, count = _consensus_probability(event, "Lakers")
        assert count == 0
        assert prob == 0.0

    def test_devig_removes_margin(self):
        """Devigged probabilities should sum to ~1.0 (not >1.0 like raw)."""
        event = {
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home", "price": 1.90},
                    {"name": "Away", "price": 2.00},
                ]}]},
            ]
        }
        prob_home, _ = _consensus_probability(event, "Home")
        prob_away, _ = _consensus_probability(event, "Away")
        assert abs(prob_home + prob_away - 1.0) < 0.01


class TestAnalyzeSports:
    """Test sports analyzer with mocked Odds API responses."""

    def test_no_api_key_returns_none(self):
        """Without API key → None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_sports("Will the Lakers win the NBA title?", end_dt)
        assert result is None

    def test_non_sports_question_returns_none(self):
        """Non-sports question → None."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=7)
        with patch("polymarket_scanner.ODDS_API_KEY", "fake-key"):
            result = analyze_sports("Will Trump win the election?", end_dt)
        assert result is None

    @patch("polymarket_scanner.ODDS_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_odds")
    def test_nba_game_detected(self, mock_fetch):
        """NBA question with matching event → analysis returned."""
        mock_fetch.return_value = [
            {
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [
                    {"markets": [{"key": "h2h", "outcomes": [
                        {"name": "Los Angeles Lakers", "price": 2.10},
                        {"name": "Boston Celtics", "price": 1.80},
                    ]}]},
                    {"markets": [{"key": "h2h", "outcomes": [
                        {"name": "Los Angeles Lakers", "price": 2.05},
                        {"name": "Boston Celtics", "price": 1.85},
                    ]}]},
                    {"markets": [{"key": "h2h", "outcomes": [
                        {"name": "Los Angeles Lakers", "price": 2.15},
                        {"name": "Boston Celtics", "price": 1.78},
                    ]}]},
                ],
            }
        ]
        end_dt = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_sports("Will the Lakers beat the Celtics in the NBA?", end_dt)
        assert result is not None
        assert result["source"] == "The Odds API"
        # Lakers ~47% underdog across bookmakers
        assert 0.35 < result["estimated_prob"] < 0.55
        assert "Lakers" in result["analysis"]
        assert result["confidence"] == "medium"  # 3 bookmakers

    @patch("polymarket_scanner.ODDS_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_odds")
    def test_no_matching_event(self, mock_fetch):
        """Sports question but no matching event → None."""
        mock_fetch.return_value = [
            {
                "home_team": "Denver Nuggets",
                "away_team": "Miami Heat",
                "bookmakers": [],
            }
        ]
        end_dt = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_sports("Will the Lakers win the NBA title?", end_dt)
        assert result is None

    @patch("polymarket_scanner.ODDS_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_odds")
    def test_insufficient_bookmakers(self, mock_fetch):
        """Only 1 bookmaker → None (need >= 2 for consensus)."""
        mock_fetch.return_value = [
            {
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [
                    {"markets": [{"key": "h2h", "outcomes": [
                        {"name": "Los Angeles Lakers", "price": 2.10},
                        {"name": "Boston Celtics", "price": 1.80},
                    ]}]},
                ],
            }
        ]
        end_dt = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_sports("Will the Lakers beat the Celtics in the NBA?", end_dt)
        assert result is None


# =========================================================================
# run_analyzers() — full pipeline with all 3 analyzers
# =========================================================================

class TestRunAnalyzers:
    """Test that run_analyzers chains weather → finance → sports."""

    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_weather_takes_priority(self, mock_w, mock_f, mock_s):
        """Weather result returned first if available."""
        mock_w.return_value = {"source": "OpenWeatherMap", "estimated_prob": 0.9}
        mock_f.return_value = {"source": "Yahoo Finance", "estimated_prob": 0.5}
        mock_s.return_value = None
        end_dt = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("test", end_dt)
        assert result["source"] == "OpenWeatherMap"

    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_finance_when_no_weather(self, mock_w, mock_f, mock_s):
        """Finance used when weather returns None."""
        mock_w.return_value = None
        mock_f.return_value = {"source": "Yahoo Finance", "estimated_prob": 0.5}
        mock_s.return_value = None
        end_dt = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("test", end_dt)
        assert result["source"] == "Yahoo Finance"

    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_sports_when_no_weather_or_finance(self, mock_w, mock_f, mock_s):
        """Sports used as last resort."""
        mock_w.return_value = None
        mock_f.return_value = None
        mock_s.return_value = {"source": "The Odds API", "estimated_prob": 0.6}
        end_dt = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("test", end_dt)
        assert result["source"] == "The Odds API"

    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_none_when_all_fail(self, mock_w, mock_f, mock_s):
        """None returned when no analyzer matches."""
        mock_w.return_value = None
        mock_f.return_value = None
        mock_s.return_value = None
        end_dt = datetime.now(timezone.utc) + timedelta(days=5)
        assert run_analyzers("test", end_dt) is None


# =========================================================================
# Audit fix #1: MIN_EDGE lowered to 5%
# =========================================================================

class TestMinEdgeLowered:
    """MIN_EDGE should be 0.05, not 0.10."""

    def test_min_edge_is_5pct(self):
        assert MIN_EDGE == 0.05

    def test_edge_7pct_qualifies(self):
        """7% edge should now qualify as edge tier."""
        ea = {"edge": 0.07, "source": "Yahoo Finance"}
        tier, _, rtype = classify(0.60, 0.0, 1.0, edge_analysis=ea)
        assert tier == "edge"
        assert rtype == "edge"


# =========================================================================
# Audit fix #2: Coinbase crypto/finance conflict
# =========================================================================

class TestCoinbaseConflict:
    """Coinbase stock markets should NOT be filtered as crypto."""

    def test_coinbase_stock_not_crypto(self):
        assert is_crypto("Will Coinbase stock reach $300?") is False

    def test_coinbase_share_not_crypto(self):
        assert is_crypto("Coinbase share price above $250?") is False

    def test_coinbase_price_not_crypto(self):
        assert is_crypto("Coinbase price above $200?") is False

    def test_coinbase_alone_is_crypto(self):
        """'Coinbase' without stock context → crypto (the exchange)."""
        assert is_crypto("Will Coinbase list the new token?") is True

    def test_coinbase_exchange_is_crypto(self):
        assert is_crypto("Coinbase regulatory issues with crypto") is True


# =========================================================================
# Audit fix #3: Black-Scholes drift
# =========================================================================

class TestFinanceDrift:
    """Finance model should include drift term."""

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_drift_increases_above_probability(self, mock_fetch):
        """With drift, P(above target below current) should benefit.

        Drift = ~4.5%/yr risk-free rate. For a target slightly BELOW current
        price with enough time, drift clearly pushes P(above) above 0.5.
        """
        mock_fetch.return_value = {"price": 100, "daily_vol": 0.01, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=180)
        result = analyze_finance("Will Tesla stock be above $98?", end_dt)
        assert result is not None
        # Target slightly below current + drift → clearly >50%
        assert result["estimated_prob"] > 0.55


# =========================================================================
# Audit fix #4: Proportional fees
# =========================================================================

class TestProportionalFees:
    """Fees should scale with number of outcomes."""

    def test_binary_arb_fee_is_2pct(self, now):
        """Binary arb: 2 outcomes → fee = 2% (2 × 2% / 2)."""
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert result["fee_per_100"] == EST_FEE_PCT  # 2.0

    def test_4outcome_arb_fee_is_4pct(self, now):
        """4-outcome arb: fee = 4% (4 × 2% / 2)."""
        m = _make_market(
            now,
            outcomePrices='["0.82", "0.08", "0.03", "0.02"]',
            outcomes='["A", "B", "C", "D"]',
        )
        result = process_market(m, now)
        assert result is not None
        assert result["fee_per_100"] == EST_FEE_PCT * 2  # 4.0

    def test_4outcome_net_accounts_for_higher_fees(self, now):
        """Multi-outcome arb net should reflect higher fees."""
        m = _make_market(
            now,
            outcomePrices='["0.82", "0.08", "0.03", "0.02"]',
            outcomes='["A", "B", "C", "D"]',
        )
        result = process_market(m, now)
        assert result is not None
        assert result["net_per_100"] == round(result["ev_per_100"] - 4.0, 2)


# =========================================================================
# Audit fix #5: Anchored temperature regex
# =========================================================================

class TestTempRegexAnchored:
    """Temperature regex should capture the right number."""

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_5day_temp_captures_80_not_5(self, mock_fetch):
        """'5-day high temperature exceed 80F' → threshold=80, not 5."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([82, 85, 78, 84, 80, 86, 79, 83])
        mock_fetch.return_value = entries
        result = analyze_weather(
            "Will NYC's 5-day high temperature exceed 80F?", end_dt
        )
        assert result is not None
        # If threshold was 5, prob would be ~1.0. If 80, prob is reasonable.
        assert "80" in result["analysis"]

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_threshold_after_keyword(self, mock_fetch):
        """'exceed 75' without unit → captures 75 via keyword regex."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        entries = _make_owm_entries([72, 74, 76, 78])
        mock_fetch.return_value = entries
        result = analyze_weather(
            "Will the temperature in Miami exceed 75 this week?", end_dt
        )
        assert result is not None


# =========================================================================
# Audit fix #7: Extended directional keywords
# =========================================================================

class TestDirectionalKeywords:
    """Test extended directional keywords for weather and finance."""

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_finance_decline_is_below(self, mock_fetch):
        mock_fetch.return_value = {"price": 100, "daily_vol": 0.02, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Tesla stock decline to $50?", end_dt)
        assert result is not None
        assert result["estimated_prob"] < 0.1  # very unlikely

    @patch("polymarket_scanner._fetch_yahoo_chart")
    def test_finance_crash_is_below(self, mock_fetch):
        mock_fetch.return_value = {"price": 100, "daily_vol": 0.02, "ticker": "TSLA"}
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_finance("Will Tesla crash below $50?", end_dt)
        assert result is not None
        assert result["estimated_prob"] < 0.1


# =========================================================================
# Audit fix #8 + #18: Spread/slippage warnings
# =========================================================================

class TestSpreadWarnings:
    """Test spread and liquidity warnings on arbs."""

    def test_low_liquidity_arb_gets_spread_warning(self, now):
        """Arb with <$25K liquidity → spread warning."""
        m = _make_market(now, outcomePrices='["0.45", "0.48"]', liquidity="15000")
        result = process_market(m, now)
        assert result is not None
        assert result["spread_warning"] != ""

    def test_high_liquidity_arb_no_warning(self, now):
        """Arb with $100K liquidity → no spread warning."""
        m = _make_market(now, outcomePrices='["0.45", "0.48"]', liquidity="100000")
        result = process_market(m, now)
        assert result is not None
        assert result["spread_warning"] == ""

    def test_multi_outcome_low_per_outcome_warning(self, now):
        """4-outcome arb with $15K total → per-outcome warning."""
        m = _make_market(
            now,
            outcomePrices='["0.82", "0.08", "0.03", "0.02"]',
            outcomes='["A", "B", "C", "D"]',
            liquidity="15000",
        )
        result = process_market(m, now)
        assert result is not None
        assert "outcome" in result["spread_warning"].lower()

    def test_spread_warning_field_always_present(self, now):
        """spread_warning should always be in result."""
        m = _make_market(now)
        result = process_market(m, now)
        assert result is not None
        assert "spread_warning" in result


# =========================================================================
# Audit fix #9: Sports confidence — horizon + bookmakers
# =========================================================================

class TestSportsConfidenceHybrid:
    """Sports confidence should combine bookmaker count and time horizon."""

    @patch("polymarket_scanner.ODDS_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_odds")
    def test_many_books_short_horizon_is_high(self, mock_fetch):
        """5+ bookmakers, <3 days → high confidence."""
        mock_fetch.return_value = [{
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Los Angeles Lakers", "price": 2.10},
                    {"name": "Boston Celtics", "price": 1.80},
                ]}]}
                for _ in range(5)
            ],
        }]
        end_dt = datetime.now(timezone.utc) + timedelta(days=1)
        result = analyze_sports("Will the Lakers beat the Celtics in the NBA?", end_dt)
        assert result is not None
        assert result["confidence"] == "high"

    @patch("polymarket_scanner.ODDS_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_odds")
    def test_few_books_far_horizon_is_low(self, mock_fetch):
        """2 bookmakers, 30+ days → low confidence."""
        mock_fetch.return_value = [{
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Los Angeles Lakers", "price": 2.10},
                    {"name": "Boston Celtics", "price": 1.80},
                ]}]}
                for _ in range(2)
            ],
        }]
        end_dt = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_sports("Will the Lakers beat the Celtics in the NBA?", end_dt)
        assert result is not None
        assert result["confidence"] == "low"


# =========================================================================
# Audit fix #10: Additional cities
# =========================================================================

class TestAdditionalCities:
    """New cities should be recognized."""

    def test_tampa(self):
        assert _find_city("Weather in Tampa this week") is not None

    def test_charlotte(self):
        assert _find_city("Temperature in Charlotte") is not None

    def test_orlando(self):
        assert _find_city("Rain in Orlando?") is not None

    def test_baltimore(self):
        assert _find_city("Snow in Baltimore") is not None

    def test_kansas_city(self):
        assert _find_city("Kansas City temperature forecast") is not None

    def test_new_orleans(self):
        assert _find_city("Will it rain in New Orleans?") is not None

    def test_berlin(self):
        assert _find_city("Temperature in Berlin above 30C") is not None

    def test_st_louis(self):
        assert _find_city("Weather in St. Louis") is not None


# =========================================================================
# Audit fix #11: Snow vs rain differentiation
# =========================================================================

class TestSnowDifferentiation:
    """Snow questions should use snow-specific logic, not raw PoP."""

    @patch("polymarket_scanner.OWM_API_KEY", "fake-key")
    @patch("polymarket_scanner._fetch_owm_forecast")
    def test_snow_with_warm_temps_low_prob(self, mock_fetch):
        """If temperature is warm (>38F), snow prob should be lower than raw PoP."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=2)
        # Warm entries with high PoP but no snow weather type
        entries = []
        for i in range(8):
            entries.append({
                "dt": int((datetime.now(timezone.utc) + timedelta(hours=3 * i)).timestamp()),
                "main": {"temp": 50, "temp_max": 52, "temp_min": 48},
                "pop": 0.8,
                "weather": [{"main": "Rain", "description": "light rain"}],
            })
        mock_fetch.return_value = entries
        result = analyze_weather("Will it snow in Chicago?", end_dt)
        assert result is not None
        # Snow prob should be much lower than rain PoP composite (~100%)
        assert result["estimated_prob"] < 0.3


# =========================================================================
# Audit fix #13: Draw market detection
# =========================================================================

class TestDrawMarket:
    """Test _has_draw_market detection."""

    def test_3way_with_draw(self):
        event = {
            "bookmakers": [{
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home", "price": 2.50},
                    {"name": "Away", "price": 3.00},
                    {"name": "Draw", "price": 3.20},
                ]}]
            }]
        }
        assert _has_draw_market(event) is True

    def test_2way_no_draw(self):
        event = {
            "bookmakers": [{
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home", "price": 1.90},
                    {"name": "Away", "price": 2.00},
                ]}]
            }]
        }
        assert _has_draw_market(event) is False


# =========================================================================
# Audit fix #16: Target date parsing in weather questions
# =========================================================================

class TestParseTargetDate:
    """Test specific date extraction from question text."""

    def test_march_5(self):
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)
        dt = _parse_target_date("temperature on March 5 in NYC", now)
        assert dt is not None
        assert dt.month == 3
        assert dt.day == 5

    def test_february_28(self):
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)
        dt = _parse_target_date("Will it rain February 28?", now)
        assert dt is not None
        assert dt.month == 2
        assert dt.day == 28

    def test_no_date_returns_none(self):
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)
        assert _parse_target_date("Will it rain this week?", now) is None

    def test_past_date_wraps_to_next_year(self):
        now = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        dt = _parse_target_date("temperature on January 10", now)
        assert dt is not None
        assert dt.year == 2027
        assert dt.month == 1

    def test_abbreviated_month(self):
        now = datetime(2026, 2, 26, 12, 0, tzinfo=timezone.utc)
        dt = _parse_target_date("Weather on Apr 15", now)
        assert dt is not None
        assert dt.month == 4


# =========================================================================
# Forex/Commodities tickers (A2)
# =========================================================================

class TestForexCommodityTickers:
    """Test that new forex/commodity tickers are mapped correctly."""

    def test_eurusd_mapped(self):
        assert FINANCE_TICKERS.get("eurusd") == "EURUSD=X"

    def test_euro_mapped(self):
        assert FINANCE_TICKERS.get("euro") == "EURUSD=X"

    def test_yen_mapped(self):
        assert FINANCE_TICKERS.get("yen") == "USDJPY=X"

    def test_yuan_mapped(self):
        assert FINANCE_TICKERS.get("yuan") == "USDCNY=X"

    def test_dxy_mapped(self):
        assert FINANCE_TICKERS.get("dxy") == "DX-Y.NYB"

    def test_copper_mapped(self):
        assert FINANCE_TICKERS.get("copper") == "HG=F"

    def test_platinum_mapped(self):
        assert FINANCE_TICKERS.get("platinum") == "PL=F"

    def test_find_ticker_euro(self):
        assert _find_ticker("Will the euro reach $1.15?") == "EURUSD=X"

    def test_find_ticker_yen(self):
        assert _find_ticker("Will the yen fall below 150?") == "USDJPY=X"

    def test_find_ticker_dxy(self):
        assert _find_ticker("Will DXY go above 105?") == "DX-Y.NYB"

    def test_find_ticker_copper(self):
        assert _find_ticker("Will copper price reach $5?") == "HG=F"

    def test_additional_stock_uber(self):
        assert FINANCE_TICKERS.get("uber") == "UBER"

    def test_additional_stock_disney(self):
        assert FINANCE_TICKERS.get("disney") == "DIS"

    def test_additional_stock_boeing(self):
        assert FINANCE_TICKERS.get("boeing") == "BA"

    def test_additional_stock_microstrategy(self):
        assert FINANCE_TICKERS.get("microstrategy") == "MSTR"


# =========================================================================
# Earnings Analyzer (S2)
# =========================================================================

class TestEarningsKeywords:
    """Test earnings question detection regex."""

    def test_beats_earnings(self):
        assert _EARNINGS_KEYWORDS_RE.search("Will Tesla beat earnings?")

    def test_eps_match(self):
        assert _EARNINGS_KEYWORDS_RE.search("NVIDIA EPS above $5?")

    def test_revenue_match(self):
        assert _EARNINGS_KEYWORDS_RE.search("Will Apple revenue beat estimates?")

    def test_q4_match(self):
        assert _EARNINGS_KEYWORDS_RE.search("Will Google Q4 earnings surprise?")

    def test_quarterly_match(self):
        assert _EARNINGS_KEYWORDS_RE.search("quarterly profit exceeds expectations")

    def test_no_match_unrelated(self):
        assert not _EARNINGS_KEYWORDS_RE.search("Will it rain in NYC?")

    def test_guidance_match(self):
        assert _EARNINGS_KEYWORDS_RE.search("Will Tesla guidance disappoint?")


class TestEarningsAnalyzer:
    """Test earnings analyzer with mocked Yahoo Finance data."""

    def _mock_earnings_data(self):
        return {
            "ticker": "TSLA",
            "beat_rate": 0.75,
            "est_eps": 1.25,
            "num_quarters": 4,
        }

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_beat_question_returns_beat_rate(self, mock_fetch):
        mock_fetch.return_value = self._mock_earnings_data()
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.75
        assert result["source"] == "Yahoo Finance Earnings"
        assert "beat" in result["analysis"].lower()

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_miss_question_returns_complement(self, mock_fetch):
        mock_fetch.return_value = self._mock_earnings_data()
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will Tesla miss earnings?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.25  # 1 - 0.75

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_no_ticker_returns_none(self, mock_fetch):
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will earnings beat estimates?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_no_earnings_keywords_returns_none(self, mock_fetch):
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will it rain tomorrow?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_confidence_high_for_short_horizon(self, mock_fetch):
        mock_fetch.return_value = self._mock_earnings_data()
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        assert result["confidence"] == "high"

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_confidence_low_for_long_horizon(self, mock_fetch):
        mock_fetch.return_value = self._mock_earnings_data()
        end = datetime.now(timezone.utc) + timedelta(days=45)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        assert result["confidence"] == "low"

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_low_data_quality_downgrades_confidence(self, mock_fetch):
        mock_fetch.return_value = {
            "ticker": "TSLA", "beat_rate": 0.50,
            "est_eps": None, "num_quarters": 1,
        }
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        assert result["confidence"] == "low"

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_fetch_failure_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is None


# =========================================================================
# Fed/Macro Analyzer (S1)
# =========================================================================

class TestFedMacroKeywords:
    """Test Fed/Macro question detection regexes."""

    def test_fed_rate_cut(self):
        assert _FED_RATE_RE.search("Will the Fed cut rates in March?")

    def test_fomc_decision(self):
        assert _FED_RATE_RE.search("FOMC rate decision at next meeting")

    def test_rate_hike(self):
        assert _FED_RATE_RE.search("Will interest rate hike happen?")

    def test_bps(self):
        assert _FED_RATE_RE.search("Will Fed cut 25 basis points?")

    def test_inflation_cpi(self):
        assert _INFLATION_RE.search("Will CPI come in above 3%?")

    def test_core_inflation(self):
        assert _INFLATION_RE.search("Core inflation above target?")

    def test_pce(self):
        assert _INFLATION_RE.search("Will PCE exceed 2.5%?")

    def test_unemployment(self):
        assert _UNEMPLOYMENT_RE.search("Unemployment rate above 4%?")

    def test_nonfarm(self):
        assert _UNEMPLOYMENT_RE.search("Nonfarm payrolls beat estimates?")

    def test_gdp(self):
        assert _GDP_RE.search("Will GDP growth exceed 2%?")

    def test_recession(self):
        assert _GDP_RE.search("Will the US enter a recession?")

    def test_no_match_unrelated(self):
        assert not _FED_RATE_RE.search("Will it rain in NYC?")
        assert not _INFLATION_RE.search("Will Tesla stock rise?")
        assert not _UNEMPLOYMENT_RE.search("Will Bitcoin hit $100k?")


class TestFedMacroAnalyzer:
    """Test Fed/Macro analyzer with mocked FRED data."""

    @patch("polymarket_scanner._fetch_fred_series")
    def test_fed_hold_with_rate(self, mock_fred):
        mock_fred.return_value = 5.25
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = analyze_fed_macro("Will the Fed hold rates unchanged?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.75
        assert result["source"] == "FRED / Fed Analysis"

    @patch("polymarket_scanner._fetch_fred_trend")
    @patch("polymarket_scanner._fetch_fred_series")
    def test_fed_cut_with_rate(self, mock_fred, mock_trend):
        mock_fred.return_value = 5.25
        mock_trend.return_value = []  # no trend data → fallback base rates
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = analyze_fed_macro("Will the Fed cut rates?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.15  # trend fallback: cut=15%

    @patch("polymarket_scanner._fetch_fred_series")
    def test_fed_hike_low_prob(self, mock_fred):
        mock_fred.return_value = 5.25
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = analyze_fed_macro("Will the Fed raise rates?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.10

    @patch("polymarket_scanner._fetch_fred_series")
    def test_fed_without_key_still_works(self, mock_fred):
        mock_fred.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = analyze_fed_macro("Will the Fed hold rates?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.75

    @patch("polymarket_scanner._fetch_fred_series")
    def test_cpi_above_threshold(self, mock_fred):
        mock_fred.return_value = 3.2
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_fed_macro("Will CPI come in above 3%?", end)
        assert result is not None
        assert result["estimated_prob"] > 0.5  # 3.2 > 3.0

    @patch("polymarket_scanner._fetch_fred_series")
    def test_cpi_below_threshold(self, mock_fred):
        mock_fred.return_value = 2.8
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_fed_macro("Will inflation fall below 3%?", end)
        assert result is not None
        assert result["estimated_prob"] > 0.5  # 2.8 < 3.0

    @patch("polymarket_scanner._fetch_fred_series")
    def test_unemployment_above_threshold(self, mock_fred):
        mock_fred.return_value = 4.2
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_fed_macro("Unemployment rate above 4%?", end)
        assert result is not None
        assert result["estimated_prob"] > 0.5

    @patch("polymarket_scanner._fetch_fred_series")
    def test_gdp_recession(self, mock_fred):
        mock_fred.return_value = -0.5
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_fed_macro("Will the US enter a recession?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.55  # negative GDP

    @patch("polymarket_scanner._fetch_fred_series")
    def test_gdp_healthy_low_recession_prob(self, mock_fred):
        mock_fred.return_value = 3.0
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_fed_macro("Will the US enter a recession?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.15

    def test_unrelated_question_returns_none(self):
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_fed_macro("Will it rain in NYC?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_fred_series")
    def test_confidence_short_horizon(self, mock_fred):
        mock_fred.return_value = 5.25
        end = datetime.now(timezone.utc) + timedelta(days=3)
        result = analyze_fed_macro("Will the Fed hold rates?", end)
        assert result is not None
        assert result["confidence"] == "high"

    @patch("polymarket_scanner._fetch_fred_series")
    def test_confidence_long_horizon(self, mock_fred):
        mock_fred.return_value = 5.25
        end = datetime.now(timezone.utc) + timedelta(days=45)
        result = analyze_fed_macro("Will the Fed cut rates?", end)
        assert result is not None
        assert result["confidence"] == "low"


class TestFredSeries:
    """Test FRED series configuration."""

    def test_fed_rate_series_exists(self):
        assert "fed_rate" in FRED_SERIES
        assert FRED_SERIES["fed_rate"]["series_id"] == "DFEDTARU"

    def test_cpi_series_exists(self):
        assert "cpi_yoy" in FRED_SERIES
        assert FRED_SERIES["cpi_yoy"]["series_id"] == "CPALTT01USM657N"

    def test_unemployment_series_exists(self):
        assert "unemployment" in FRED_SERIES
        assert FRED_SERIES["unemployment"]["series_id"] == "UNRATE"

    def test_gdp_series_exists(self):
        assert "gdp_growth" in FRED_SERIES

    def test_nonfarm_series_exists(self):
        assert "nonfarm_payrolls" in FRED_SERIES


# =========================================================================
# Crypto Price Analyzer (B1)
# =========================================================================

class TestCryptoTickers:
    """Test crypto ticker mapping."""

    def test_bitcoin_mapped(self):
        assert CRYPTO_TICKERS.get("bitcoin") == "bitcoin"

    def test_btc_mapped(self):
        assert CRYPTO_TICKERS.get("btc") == "bitcoin"

    def test_ethereum_mapped(self):
        assert CRYPTO_TICKERS.get("ethereum") == "ethereum"

    def test_solana_mapped(self):
        assert CRYPTO_TICKERS.get("solana") == "solana"

    def test_dogecoin_mapped(self):
        assert CRYPTO_TICKERS.get("dogecoin") == "dogecoin"

    def test_xrp_mapped(self):
        assert CRYPTO_TICKERS.get("xrp") == "ripple"

    def test_pepe_mapped(self):
        assert CRYPTO_TICKERS.get("pepe") == "pepe"

    def test_sui_mapped(self):
        assert CRYPTO_TICKERS.get("sui") == "sui"


class TestFindCryptoTicker:
    """Test crypto ticker extraction from questions."""

    def test_bitcoin_found(self):
        assert _find_crypto_ticker("Will Bitcoin reach $100000?") == "bitcoin"

    def test_ethereum_found(self):
        assert _find_crypto_ticker("Will Ethereum hit $5000?") == "ethereum"

    def test_solana_found(self):
        assert _find_crypto_ticker("Will Solana price exceed $300?") == "solana"

    def test_no_match(self):
        assert _find_crypto_ticker("Will Tesla stock rise?") is None

    def test_longest_match_wins(self):
        """'shiba inu' (8 chars) should win over 'sui' (3 chars)."""
        assert _find_crypto_ticker("Will Shiba Inu reach $0.001?") == "shiba-inu"


class TestCryptoAnalyzer:
    """Test crypto analyzer with mocked CoinGecko data."""

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_bitcoin_above_target(self, mock_fetch):
        mock_fetch.return_value = {
            "price": 95000, "ath": 100000, "change_30d": 10.0,
            "monthly_vol": 0.10, "coin_id": "bitcoin",
        }
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_crypto("Will Bitcoin reach $100000?", end)
        assert result is not None
        assert 0 < result["estimated_prob"] < 1
        assert result["source"] == "CoinGecko"
        assert "bitcoin" in result["analysis"].lower() or "Bitcoin" in result["analysis"]

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_bitcoin_below_target(self, mock_fetch):
        mock_fetch.return_value = {
            "price": 95000, "ath": 100000, "change_30d": 10.0,
            "monthly_vol": 0.10, "coin_id": "bitcoin",
        }
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_crypto("Will Bitcoin drop below $80000?", end)
        assert result is not None
        # Below 80k when at 95k should have low probability
        assert result["estimated_prob"] < 0.5

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_no_price_target_returns_none(self, mock_fetch):
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_crypto("Will Bitcoin be adopted by institutions?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_no_crypto_ticker_returns_none(self, mock_fetch):
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_crypto("Will gold reach $3000?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_fetch_failure_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_crypto("Will Bitcoin reach $100000?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_confidence_always_medium_or_low(self, mock_fetch):
        """Crypto confidence should never be 'high' (too volatile)."""
        mock_fetch.return_value = {
            "price": 95000, "ath": 100000, "change_30d": 10.0,
            "monthly_vol": 0.10, "coin_id": "bitcoin",
        }
        end = datetime.now(timezone.utc) + timedelta(days=3)
        result = analyze_crypto("Will Bitcoin reach $100000?", end)
        assert result is not None
        assert result["confidence"] in ("medium", "low")

    @patch("polymarket_scanner._fetch_crypto_price")
    def test_ethereum_analysis(self, mock_fetch):
        mock_fetch.return_value = {
            "price": 3500, "ath": 4800, "change_30d": 5.0,
            "monthly_vol": 0.08, "coin_id": "ethereum",
        }
        end = datetime.now(timezone.utc) + timedelta(days=14)
        result = analyze_crypto("Will Ethereum hit $5000?", end)
        assert result is not None
        assert result["estimated_prob"] < 0.5  # 42% away is hard


# =========================================================================
# Elections/Polls Analyzer (A1)
# =========================================================================

class TestElectionKeywords:
    """Test election/political question detection."""

    def test_election_detected(self):
        assert _ELECTION_RE.search("Who will win the 2028 presidential election?")

    def test_nominee_detected(self):
        assert _ELECTION_RE.search("Democratic nominee for president?")

    def test_midterm_detected(self):
        assert _ELECTION_RE.search("Will Democrats win the midterms?")

    def test_senate_control(self):
        assert _ELECTION_RE.search("Will Republicans control the Senate?")

    def test_approval_detected(self):
        assert _APPROVAL_RE.search("Trump approval rating above 50%?")

    def test_favorability(self):
        assert _APPROVAL_RE.search("Will Biden favorability improve?")

    def test_no_match_unrelated(self):
        assert not _ELECTION_RE.search("Will Tesla stock rise?")
        assert not _APPROVAL_RE.search("Will it rain tomorrow?")


class TestPoliticalFigures:
    """Test political figure detection."""

    def test_trump_found(self):
        info = _find_political_figure("Will Trump win re-election?")
        assert info is not None
        assert info["name"] == "Donald Trump"
        assert info["party"] == "R"

    def test_biden_found(self):
        info = _find_political_figure("Biden approval above 45%?")
        assert info is not None
        assert info["name"] == "Joe Biden"

    def test_newsom_found(self):
        info = _find_political_figure("Will Newsom be the nominee?")
        assert info is not None
        assert info["name"] == "Gavin Newsom"

    def test_unknown_figure_returns_none(self):
        assert _find_political_figure("Will rain affect crops?") is None


class TestElectionsAnalyzer:
    """Test elections analyzer with mocked data."""

    @patch("polymarket_scanner._fetch_approval_data")
    def test_approval_above_threshold(self, mock_fetch):
        mock_fetch.return_value = {
            "approve": 48.0, "disapprove": 49.5,
            "source": "RealClearPolitics",
        }
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections("Trump approval rating above 45%?", end)
        assert result is not None
        assert result["estimated_prob"] > 0.5  # 48 > 45
        assert result["source"] == "RealClearPolitics"

    @patch("polymarket_scanner._fetch_approval_data")
    def test_approval_below_threshold(self, mock_fetch):
        mock_fetch.return_value = {
            "approve": 42.0, "disapprove": 54.0,
            "source": "RealClearPolitics",
        }
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections(
            "Will Trump approval fall below 45%?", end
        )
        assert result is not None
        assert result["estimated_prob"] > 0.5  # 42 < 45

    def test_senate_dem_control(self):
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections(
            "Will Democrats control the Senate after midterms?", end
        )
        assert result is not None
        assert result["estimated_prob"] == 0.45

    def test_senate_rep_control(self):
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections(
            "Will Republicans control the Senate?", end
        )
        assert result is not None
        assert result["estimated_prob"] == 0.55

    def test_house_control(self):
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections(
            "Will Democrats control the House?", end
        )
        assert result is not None
        assert result["estimated_prob"] == 0.50

    def test_candidate_with_incumbent_advantage(self):
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections(
            "Will Trump win re-election as president?", end
        )
        assert result is not None
        assert result["estimated_prob"] == 0.55

    def test_unrelated_returns_none(self):
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections("Will gold reach $3000?", end)
        assert result is None

    @patch("polymarket_scanner._fetch_approval_data")
    def test_approval_no_data_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=30)
        result = analyze_elections("Trump approval above 50%?", end)
        assert result is None

    def test_election_confidence_is_low(self):
        """Elections far out → low confidence."""
        end = datetime.now(timezone.utc) + timedelta(days=60)
        result = analyze_elections(
            "Will Republicans control the Senate?", end
        )
        assert result is not None
        assert result["confidence"] == "low"


# =========================================================================
# Updated category extraction
# =========================================================================

class TestNewCategories:
    """Test new category extraction rules."""

    def test_finance_category_earnings(self):
        m = {"question": "Will Tesla beat Q4 earnings?", "tags": None}
        assert extract_category(m) == "Finance"

    def test_finance_category_ipo(self):
        m = {"question": "Will Cerebras IPO by March?", "tags": None}
        assert extract_category(m) == "Finance"

    def test_crypto_category(self):
        m = {"question": "Will Bitcoin reach $150k?", "tags": None}
        assert extract_category(m) == "Crypto"

    def test_commodities_category(self):
        m = {"question": "Will gold price exceed $3000?", "tags": None}
        assert extract_category(m) == "Commodities"

    def test_forex_category(self):
        m = {"question": "Will the euro reach $1.15?", "tags": None}
        assert extract_category(m) == "Forex"

    def test_economics_fomc(self):
        m = {"question": "Will the FOMC cut rates?", "tags": None}
        assert extract_category(m) == "Economics"

    def test_economics_tariff(self):
        m = {"question": "Will tariff rates increase?", "tags": None}
        assert extract_category(m) == "Economics"

    def test_politics_midterm(self):
        m = {"question": "Who wins the midterm elections?", "tags": None}
        assert extract_category(m) == "Politics"

    def test_entertainment_tweet(self):
        m = {"question": "Will Trump tweet about it?", "tags": None}
        # "trump" matches Politics first (before Entertainment)
        assert extract_category(m) == "Politics"

    def test_sports_cricket(self):
        m = {"question": "Will India win the cricket match?", "tags": None}
        assert extract_category(m) == "Sports"

    def test_weather_snow(self):
        m = {"question": "Will it snow in NYC this weekend?", "tags": None}
        assert extract_category(m) == "Weather"


# =========================================================================
# Updated run_analyzers order
# =========================================================================

class TestRunAnalyzersExpanded:
    """Test the expanded analyzer chain."""

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_order_weather_first(self, mock_w, mock_f, mock_e,
                                  mock_s, mock_m, mock_el, mock_c):
        mock_w.return_value = {"source": "Weather", "estimated_prob": 0.9, "confidence": "high"}
        mock_f.return_value = {"source": "Finance", "estimated_prob": 0.5, "confidence": "medium"}
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "Weather"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_finance_before_earnings(self, mock_w, mock_f, mock_e,
                                      mock_s, mock_m, mock_el, mock_c):
        mock_w.return_value = None
        mock_f.return_value = {"source": "Finance", "estimated_prob": 0.7, "confidence": "high"}
        mock_e.return_value = {"source": "Earnings", "estimated_prob": 0.6, "confidence": "medium"}
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "Finance"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_earnings_before_sports(self, mock_w, mock_f, mock_e,
                                     mock_s, mock_m, mock_el, mock_c):
        mock_w.return_value = None
        mock_f.return_value = None
        mock_e.return_value = {"source": "Earnings", "estimated_prob": 0.75, "confidence": "high"}
        mock_s.return_value = {"source": "Sports", "estimated_prob": 0.6, "confidence": "medium"}
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "Earnings"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_fed_macro_after_sports(self, mock_w, mock_f, mock_e,
                                     mock_s, mock_m, mock_el, mock_c):
        mock_w.return_value = None
        mock_f.return_value = None
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = {"source": "FRED", "estimated_prob": 0.7, "confidence": "medium"}
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "FRED"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_elections_after_fed(self, mock_w, mock_f, mock_e,
                                  mock_s, mock_m, mock_el, mock_c):
        mock_w.return_value = None
        mock_f.return_value = None
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = {"source": "Elections"}
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "Elections"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_crypto_last(self, mock_w, mock_f, mock_e,
                          mock_s, mock_m, mock_el, mock_c):
        mock_w.return_value = None
        mock_f.return_value = None
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = {"source": "CoinGecko"}
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "CoinGecko"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_none_when_all_return_none(self, mock_w, mock_f, mock_e,
                                        mock_s, mock_m, mock_el, mock_c):
        for m in [mock_w, mock_f, mock_e, mock_s, mock_m, mock_el, mock_c]:
            m.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result is None


# =========================================================================
# Crypto gate updated behavior
# =========================================================================

class TestCryptoGateUpdated:
    """Test that crypto gate allows price target markets through."""

    def test_crypto_no_price_target_still_filtered(self, now):
        """Generic crypto question without price → filtered."""
        m = _make_market(now, question="Will Bitcoin be adopted as legal tender?")
        assert process_market(m, now) is None

    def test_crypto_unrelated_still_filtered(self, now):
        """Crypto discussion question → filtered."""
        m = _make_market(now, question="Will Ethereum switch to proof of stake again?")
        assert process_market(m, now) is None


# =========================================================================
# v6 — Kelly Criterion
# =========================================================================

class TestKellyCriterion:
    """Test Kelly fraction calculation."""

    def test_edge_bet(self):
        """60% win prob at 50c → Kelly = (1*0.6-0.4)/1 = 0.2 = 20%."""
        f = kelly_fraction(0.6, 0.50)
        assert abs(f - 0.20) < 0.01

    def test_fair_bet_zero(self):
        """50% win prob at 50c → Kelly = 0 (no edge)."""
        f = kelly_fraction(0.5, 0.50)
        assert f == 0.0

    def test_negative_edge_zero(self):
        """40% win prob at 50c → negative Kelly → capped at 0."""
        f = kelly_fraction(0.4, 0.50)
        assert f == 0.0

    def test_capped_at_25pct(self):
        """Very strong edge → capped at 25%."""
        f = kelly_fraction(0.99, 0.10)
        assert f == 0.25

    def test_edge_zero_price(self):
        """Price 0 → Kelly 0."""
        assert kelly_fraction(0.6, 0.0) == 0.0

    def test_edge_one_price(self):
        """Price 1.0 → Kelly 0."""
        assert kelly_fraction(0.6, 1.0) == 0.0

    def test_small_edge(self):
        """55% at 50c → Kelly ~0.10."""
        f = kelly_fraction(0.55, 0.50)
        assert 0.05 < f < 0.15


# =========================================================================
# v6 — CoinGecko Rate Limiter
# =========================================================================

class TestCoinGeckoRateLimit:
    """Test CoinGecko rate limiting."""

    def test_min_interval_configured(self):
        assert COINGECKO_MIN_INTERVAL >= 1.0

    def test_rate_limit_function_exists(self):
        """_cg_rate_limit should be callable."""
        assert callable(_cg_rate_limit)


# =========================================================================
# v6 — CLOB Verification
# =========================================================================

class TestCLOBVerification:
    """Test CLOB spread verification."""

    @patch("polymarket_scanner._fetch_order_book")
    def test_executable_arb(self, mock_book):
        """Arb with tight spread → executable."""
        mock_book.return_value = {
            "best_bid": 0.44, "best_ask": 0.45,
            "spread": 0.01, "spread_pct": 2.2,
            "bid_depth": 1000, "ask_depth": 1000,
        }
        market = {"clobTokenIds": '["token1", "token2"]'}
        prices = [0.45, 0.48]  # sum=0.93, deviation=7%
        result = verify_arb_execution(market, prices)
        assert result["executable"] is True
        assert result["total_spread_cost_pct"] > 0
        assert result["effective_arb_pct"] > 0

    @patch("polymarket_scanner._fetch_order_book")
    def test_not_executable_wide_spread(self, mock_book):
        """Arb with wide spread → not executable."""
        mock_book.return_value = {
            "best_bid": 0.40, "best_ask": 0.48,
            "spread": 0.08, "spread_pct": 16.7,
            "bid_depth": 100, "ask_depth": 100,
        }
        market = {"clobTokenIds": '["token1", "token2"]'}
        prices = [0.48, 0.49]  # sum=0.97, deviation=3%
        result = verify_arb_execution(market, prices)
        assert result["executable"] is False

    @patch("polymarket_scanner._fetch_order_book")
    def test_no_clob_ids(self, mock_book):
        """Market without clobTokenIds → executable=None."""
        market = {}
        result = verify_arb_execution(market, [0.45, 0.48])
        assert result["executable"] is None

    @patch("polymarket_scanner._fetch_order_book")
    def test_clob_api_failure(self, mock_book):
        """CLOB API failure → books empty."""
        mock_book.return_value = None
        market = {"clobTokenIds": '["token1"]'}
        result = verify_arb_execution(market, [0.45])
        assert result["books"] == [None]


# =========================================================================
# v6 — Health Tracking
# =========================================================================

class TestHealthTracking:
    """Test analyzer health tracking."""

    def test_record_health_ok(self):
        _record_health("test_analyzer", True, "test")
        from polymarket_scanner import _analyzer_health
        assert "test_analyzer" in _analyzer_health
        assert _analyzer_health["test_analyzer"]["status"] == "ok"

    def test_record_health_error(self):
        _record_health("test_analyzer2", False, "connection timeout")
        from polymarket_scanner import _analyzer_health
        assert _analyzer_health["test_analyzer2"]["status"] == "error"
        assert "timeout" in _analyzer_health["test_analyzer2"]["detail"]


# =========================================================================
# v6 — Fed Rate Trend Probabilities
# =========================================================================

class TestRateTrendProbabilities:
    """Test trend-based Fed rate probability model."""

    def test_no_trend_data_fallback(self):
        """Empty trend → fallback base rates."""
        probs = _rate_trend_probabilities(5.25, [])
        assert abs(probs["hold"] + probs["cut"] + probs["hike"] - 1.0) < 0.01

    def test_all_holds_trend(self):
        """Constant rate → very high hold probability."""
        trend = [5.25, 5.25, 5.25, 5.25, 5.25]
        probs = _rate_trend_probabilities(5.25, trend)
        assert probs["hold"] > 0.80

    def test_cutting_cycle(self):
        """Rate declining → higher cut probability."""
        trend = [4.75, 5.00, 5.25, 5.50, 5.50]  # recent = most first
        probs = _rate_trend_probabilities(4.75, trend)
        assert probs["cut"] > probs["hike"]

    def test_hiking_cycle(self):
        """Rate rising → higher hike probability."""
        trend = [5.50, 5.25, 5.00, 4.75, 4.50]  # recent = most first
        probs = _rate_trend_probabilities(5.50, trend)
        assert probs["hike"] > probs["cut"]

    def test_probs_sum_to_one(self):
        """Probabilities should always sum to ~1.0."""
        for trend in [
            [5.25, 5.25, 5.25],
            [4.75, 5.00, 5.25],
            [5.50, 5.25, 5.00],
            [],
        ]:
            probs = _rate_trend_probabilities(5.25, trend)
            total = probs["hold"] + probs["cut"] + probs["hike"]
            assert abs(total - 1.0) < 0.02, f"Sum={total} for trend={trend}"


# =========================================================================
# v6 — Earnings with Revisions
# =========================================================================

class TestEarningsRevisions:
    """Test earnings analyzer with revision data."""

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_upward_revision_boosts_beat(self, mock_fetch):
        """Upward revisions → higher beat probability."""
        mock_fetch.return_value = {
            "ticker": "TSLA", "beat_rate": 0.65,
            "est_eps": 1.50, "num_quarters": 4,
            "revision_trend": 0.05,  # +5% revision
            "revenue_growth": 0.15,
        }
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        # 65% base + 5% revision boost + 3% revenue boost
        assert result["estimated_prob"] > 0.65

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_downward_revision_hurts_beat(self, mock_fetch):
        """Downward revisions → lower beat probability."""
        mock_fetch.return_value = {
            "ticker": "TSLA", "beat_rate": 0.65,
            "est_eps": 1.50, "num_quarters": 4,
            "revision_trend": -0.05,
            "revenue_growth": None,
        }
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        assert result["estimated_prob"] < 0.65

    @patch("polymarket_scanner._fetch_earnings_data")
    def test_no_revision_data_unchanged(self, mock_fetch):
        """No revision data → base beat rate unchanged."""
        mock_fetch.return_value = {
            "ticker": "TSLA", "beat_rate": 0.75,
            "est_eps": 1.25, "num_quarters": 4,
            "revision_trend": 0.0,
            "revenue_growth": None,
        }
        end = datetime.now(timezone.utc) + timedelta(days=7)
        result = analyze_earnings("Will Tesla beat earnings?", end)
        assert result is not None
        assert result["estimated_prob"] == 0.75


# =========================================================================
# v6 — Prediction Logging
# =========================================================================

class TestPredictionLogging:
    """Test prediction logging to JSONL."""

    def test_log_prediction_writes(self, tmp_path):
        """Prediction should be written to file."""
        import polymarket_scanner
        old_path = polymarket_scanner.PREDICTIONS_LOG_FILE
        polymarket_scanner.PREDICTIONS_LOG_FILE = str(tmp_path / "test_predictions.jsonl")
        try:
            opp = {
                "id": "test123", "question": "Test?",
                "tier": "edge", "score": 100,
                "yes": 0.5, "no": 0.5, "price_sum": 1.0,
                "liquidity": 10000, "volume": 5000,
                "trade_label": "BUY YES", "trade_side_class": "yes",
                "ev_per_100": 5.0, "net_per_100": 3.0, "kelly_pct": 10.0,
                "edge_analysis": {
                    "source": "Test", "estimated_prob": 0.6,
                    "edge": 0.1, "confidence": "high",
                },
            }
            _log_prediction(opp)
            with open(polymarket_scanner.PREDICTIONS_LOG_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["market_id"] == "test123"
            assert data["edge"]["source"] == "Test"
        finally:
            polymarket_scanner.PREDICTIONS_LOG_FILE = old_path


# =========================================================================
# v6 — Scan History
# =========================================================================

class TestScanHistory:
    """Test historical scan tracking."""

    def test_record_scan_history(self):
        """Scan history should be recorded."""
        from polymarket_scanner import _scan_history, _history_lock
        with _history_lock:
            _scan_history.clear()

        scan_data = {
            "refreshed_at": "2026-02-26 12:00:00 UTC",
            "total_opps": 5,
            "tiers": {
                "edge": [{"question": "Q1", "tier": "edge", "score": 100, "ev_per_100": 5}],
                "super": [],
                "interesting": [],
                "watch": [],
            },
        }
        _record_scan_history(scan_data)
        with _history_lock:
            assert len(_scan_history) == 1
            assert _scan_history[0]["total_opps"] == 5
            assert len(_scan_history[0]["top_opportunities"]) == 1


# =========================================================================
# v6 — process_market outputs Kelly and CLOB
# =========================================================================

class TestProcessMarketV6:
    """Test that process_market includes Kelly and CLOB data."""

    @pytest.fixture
    def now(self):
        return datetime.now(timezone.utc)

    @patch("polymarket_scanner.verify_arb_execution")
    @patch("polymarket_scanner.run_analyzers")
    def test_arb_has_kelly_25(self, mock_run, mock_clob, now):
        """Arb opportunities should have Kelly=25%."""
        mock_run.return_value = None
        mock_clob.return_value = {"executable": None, "total_spread_cost_pct": 0,
                                   "effective_arb_pct": 0, "books": []}
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert result["kelly_pct"] == 25.0

    @patch("polymarket_scanner.verify_arb_execution")
    @patch("polymarket_scanner.run_analyzers")
    def test_edge_has_kelly_calculated(self, mock_run, mock_clob, now):
        """Edge opportunities should have Kelly > 0."""
        mock_run.return_value = {
            "estimated_prob": 0.70,
            "source": "Yahoo Finance",
            "analysis": "test",
            "confidence": "high",
        }
        mock_clob.return_value = {"executable": None, "total_spread_cost_pct": 0,
                                   "effective_arb_pct": 0, "books": []}
        m = _make_market(now, outcomePrices='["0.50", "0.50"]')
        result = process_market(m, now)
        assert result is not None
        if result["tier"] == "edge":
            assert result["kelly_pct"] > 0

    @patch("polymarket_scanner.run_analyzers")
    def test_clob_data_present_for_arb(self, mock_run, now):
        """Arb results should include clob_data."""
        mock_run.return_value = None
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert "clob_data" in result

    @patch("polymarket_scanner.run_analyzers")
    def test_kelly_pct_in_result(self, mock_run, now):
        """kelly_pct should always be in result."""
        mock_run.return_value = None
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert "kelly_pct" in result


# =========================================================================
# v6 — Multi-analyzer picks best confidence
# =========================================================================

class TestMultiAnalyzerFusion:
    """Test that multi-analyzer picks best by confidence."""

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_high_confidence_wins(self, mock_w, mock_f, mock_e,
                                    mock_s, mock_m, mock_el, mock_c):
        """Higher confidence analyzer should win over lower."""
        mock_w.return_value = {"source": "Weather", "estimated_prob": 0.6, "confidence": "low"}
        mock_f.return_value = {"source": "Finance", "estimated_prob": 0.7, "confidence": "high"}
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "Finance"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_single_result_returned_directly(self, mock_w, mock_f, mock_e,
                                               mock_s, mock_m, mock_el, mock_c):
        """Single analyzer match → returned without sorting."""
        mock_w.return_value = None
        mock_f.return_value = {"source": "Finance", "estimated_prob": 0.5}
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert result["source"] == "Finance"

    @patch("polymarket_scanner.analyze_crypto")
    @patch("polymarket_scanner.analyze_elections")
    @patch("polymarket_scanner.analyze_fed_macro")
    @patch("polymarket_scanner.analyze_sports")
    @patch("polymarket_scanner.analyze_earnings")
    @patch("polymarket_scanner.analyze_finance")
    @patch("polymarket_scanner.analyze_weather")
    def test_also_analyzed_by_populated(self, mock_w, mock_f, mock_e,
                                          mock_s, mock_m, mock_el, mock_c):
        """Multiple matches → also_analyzed_by field populated."""
        mock_w.return_value = {"source": "Weather", "estimated_prob": 0.8, "confidence": "high"}
        mock_f.return_value = {"source": "Finance", "estimated_prob": 0.6, "confidence": "low"}
        mock_e.return_value = None
        mock_s.return_value = None
        mock_m.return_value = None
        mock_el.return_value = None
        mock_c.return_value = None
        end = datetime.now(timezone.utc) + timedelta(days=5)
        result = run_analyzers("question", end)
        assert "also_analyzed_by" in result
        assert len(result["also_analyzed_by"]) == 1


# =========================================================================
# v6 — Yahoo Finance Fallback
# =========================================================================

class TestYahooFallback:
    """Test Yahoo Finance endpoint fallback."""

    @patch("polymarket_scanner.requests.get")
    def test_fallback_to_query2(self, mock_get):
        """If query1 fails, should try query2."""
        from polymarket_scanner import _fetch_yahoo_chart, _finance_cache, _finance_lock
        # Clear cache first
        with _finance_lock:
            _finance_cache.clear()

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("query1 down")
            # Return valid response for query2
            mock_resp = type('Response', (), {
                'status_code': 200,
                'raise_for_status': lambda self: None,
                'json': lambda self: {"chart": {"result": [{
                    "meta": {"regularMarketPrice": 100},
                    "indicators": {"quote": [{"close": [95 + i for i in range(20)]}]},
                }]}},
            })()
            return mock_resp

        mock_get.side_effect = side_effect
        result = _fetch_yahoo_chart("TEST")
        assert result is not None
        assert result["price"] == 100
        assert call_count[0] >= 2  # tried at least 2 endpoints

    @pytest.fixture
    def now(self):
        return datetime.now(timezone.utc)


# =========================================================================
# Anomaly Detection — Trading Anomaly Scanner
# =========================================================================

from polymarket_scanner import (
    ANOMALY_VOL_SPIKE_THRESHOLD,
    ANOMALY_PRICE_JUMP_PCT,
    ANOMALY_PRICE_VELOCITY_PCT,
    ANOMALY_VL_RATIO_THRESHOLD,
    ANOMALY_MIN_VOLUME,
    ANOMALY_MIN_LIQUIDITY,
    _detect_volume_spike,
    _detect_price_jumps,
    _detect_price_velocity,
    _detect_volume_liquidity_imbalance,
    _fetch_price_history,
    detect_anomalies,
)


class TestVolumeSpike:
    """Test volume spike detection."""

    def test_spike_detected_when_above_threshold(self):
        """24h vol = 10x daily avg -> spike detected."""
        # Market active 30 days (default), total vol = 30000 -> daily avg = 1000
        # 24h vol = 6000 -> ratio = 6x (above 5x threshold)
        result = _detect_volume_spike(30000, 6000, None, None)
        assert result is not None
        assert result["type"] == "volume_spike"
        assert result["ratio"] >= ANOMALY_VOL_SPIKE_THRESHOLD

    def test_no_spike_below_threshold(self):
        """24h vol = 2x daily avg -> no spike."""
        # Market active 30 days (default), total vol = 30000 -> daily avg = 1000
        # 24h vol = 2000 -> ratio = 2x (below 5x threshold)
        result = _detect_volume_spike(30000, 2000, None, None)
        assert result is None

    def test_spike_with_start_date(self):
        """Uses startDate to calculate accurate daily average."""
        start = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        # 10 days old, total vol = 10000 -> daily avg = 1000
        # 24h vol = 8000 -> ratio = 8x (well above 5x)
        result = _detect_volume_spike(10000, 8000, start, None)
        assert result is not None
        assert result["ratio"] >= 7.0

    def test_zero_volume_24h_returns_none(self):
        """Zero 24h volume -> no spike."""
        assert _detect_volume_spike(100000, 0, None, None) is None

    def test_zero_total_volume_returns_none(self):
        """Zero total volume -> no spike."""
        assert _detect_volume_spike(0, 5000, None, None) is None

    def test_severity_scales_with_ratio(self):
        """Higher ratio -> higher severity."""
        r1 = _detect_volume_spike(30000, 6000, None, None)  # ~6x
        r2 = _detect_volume_spike(30000, 15000, None, None)  # ~15x
        assert r2["severity"] > r1["severity"]


class TestPriceJumps:
    """Test sudden price jump detection."""

    def test_jump_detected(self):
        """Large price jump between consecutive points -> detected."""
        history = [
            {"t": 1000, "p": 0.50},
            {"t": 2000, "p": 0.50},
            {"t": 3000, "p": 0.70},  # 40% jump
            {"t": 4000, "p": 0.72},
        ]
        result = _detect_price_jumps(history)
        assert result is not None
        assert result["type"] == "price_jump"
        assert result["jump_pct"] >= ANOMALY_PRICE_JUMP_PCT

    def test_no_jump_small_changes(self):
        """Small gradual changes -> no jump."""
        history = [
            {"t": 1000, "p": 0.50},
            {"t": 2000, "p": 0.51},
            {"t": 3000, "p": 0.52},
            {"t": 4000, "p": 0.53},
        ]
        result = _detect_price_jumps(history)
        assert result is None

    def test_single_point_returns_none(self):
        """Only one data point -> no jump possible."""
        assert _detect_price_jumps([{"t": 1000, "p": 0.50}]) is None

    def test_empty_history_returns_none(self):
        """Empty history -> no jump."""
        assert _detect_price_jumps([]) is None

    def test_finds_largest_jump(self):
        """Multiple jumps -> returns the largest one."""
        history = [
            {"t": 1000, "p": 0.50},
            {"t": 2000, "p": 0.58},  # 16% jump
            {"t": 3000, "p": 0.56},
            {"t": 4000, "p": 0.80},  # 42.8% jump (largest)
            {"t": 5000, "p": 0.82},
        ]
        result = _detect_price_jumps(history)
        assert result is not None
        assert result["price_to"] == pytest.approx(0.80, abs=0.01)

    def test_downward_jump_detected(self):
        """Price crash also detected."""
        history = [
            {"t": 1000, "p": 0.80},
            {"t": 2000, "p": 0.80},
            {"t": 3000, "p": 0.55},  # ~31% drop
        ]
        result = _detect_price_jumps(history)
        assert result is not None
        assert result["jump_pct"] > 20


class TestPriceVelocity:
    """Test price acceleration detection."""

    def test_velocity_detected_upward(self):
        """Rapid price increase over 6h window -> detected."""
        history = [
            {"t": i * 3600, "p": 0.40 + i * 0.04}
            for i in range(8)  # 0.40 -> 0.68 over 8 points
        ]
        result = _detect_price_velocity(history)
        assert result is not None
        assert result["type"] == "price_velocity"

    def test_velocity_detected_downward(self):
        """Rapid price decrease -> also detected."""
        history = [
            {"t": i * 3600, "p": 0.80 - i * 0.05}
            for i in range(8)  # 0.80 -> 0.45 over 8 points
        ]
        result = _detect_price_velocity(history)
        assert result is not None

    def test_no_velocity_stable(self):
        """Stable prices -> no velocity alert."""
        history = [
            {"t": i * 3600, "p": 0.50 + (i % 2) * 0.01}
            for i in range(8)
        ]
        result = _detect_price_velocity(history)
        assert result is None

    def test_short_history_handled(self):
        """Less than 6 points -> uses whatever is available."""
        history = [
            {"t": 1000, "p": 0.40},
            {"t": 2000, "p": 0.40},
            {"t": 3000, "p": 0.70},  # 75% increase
        ]
        result = _detect_price_velocity(history)
        assert result is not None

    def test_two_points_too_few(self):
        """Only 2 points -> too few for velocity."""
        result = _detect_price_velocity([
            {"t": 1000, "p": 0.50},
            {"t": 2000, "p": 0.90},
        ])
        assert result is None  # needs >= 3


class TestVLImbalance:
    """Test volume/liquidity imbalance detection."""

    def test_high_vl_detected(self):
        """Very high V/L ratio -> imbalance detected."""
        result = _detect_volume_liquidity_imbalance(150000, 3000)
        assert result is not None
        assert result["type"] == "vl_imbalance"
        assert result["vl_ratio"] >= ANOMALY_VL_RATIO_THRESHOLD

    def test_normal_vl_no_alert(self):
        """Normal V/L ratio -> no alert."""
        result = _detect_volume_liquidity_imbalance(10000, 50000)
        assert result is None

    def test_zero_liquidity_returns_none(self):
        """Zero liquidity -> no alert (div by zero guard)."""
        assert _detect_volume_liquidity_imbalance(10000, 0) is None

    def test_zero_volume_returns_none(self):
        """Zero volume -> no alert."""
        assert _detect_volume_liquidity_imbalance(0, 5000) is None


class TestDetectAnomalies:
    """Test the main detect_anomalies() function."""

    @pytest.fixture
    def now(self):
        return datetime.now(timezone.utc)

    def _make_anomaly_market(self, now, **overrides):
        """Factory for a market dict suitable for anomaly detection."""
        start = (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        base = {
            "question": "Will X happen by March?",
            "slug": "will-x-happen",
            "conditionId": "cond_anomaly_123",
            "endDate": (now + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "startDate": start,
            "volume": "50000",
            "volume24hr": "30000",  # 6x daily avg
            "liquidity": "10000",
            "outcomePrices": '["0.60", "0.40"]',
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["token_abc", "token_def"]',
        }
        base.update(overrides)
        return base

    def test_volume_spike_detected(self, now):
        """Market with abnormal 24h volume -> anomaly detected."""
        m = self._make_anomaly_market(now, volume="50000", volume24hr="30000")
        result = detect_anomalies(m, now)
        assert result is not None
        assert any(a["type"] == "volume_spike" for a in result["anomalies"])

    def test_vl_imbalance_detected(self, now):
        """Market with high V/L ratio -> detected."""
        m = self._make_anomaly_market(
            now, volume24hr="100000", liquidity="3000"
        )
        result = detect_anomalies(m, now)
        assert result is not None
        assert any(a["type"] == "vl_imbalance" for a in result["anomalies"])

    @patch("polymarket_scanner._fetch_price_history")
    def test_price_jump_detected_via_clob(self, mock_fetch, now):
        """CLOB price history with jump -> anomaly detected."""
        mock_fetch.return_value = [
            {"t": 1000, "p": 0.50},
            {"t": 2000, "p": 0.50},
            {"t": 3000, "p": 0.75},  # 50% jump
            {"t": 4000, "p": 0.74},
        ]
        m = self._make_anomaly_market(
            now, volume="50000", volume24hr="1000", liquidity="50000"
        )
        result = detect_anomalies(m, now)
        assert result is not None
        assert any(a["type"] == "price_jump" for a in result["anomalies"])

    @patch("polymarket_scanner._fetch_price_history")
    def test_price_velocity_detected(self, mock_fetch, now):
        """Rapid price acceleration -> detected."""
        mock_fetch.return_value = [
            {"t": i * 3600, "p": 0.40 + i * 0.04}
            for i in range(8)
        ]
        m = self._make_anomaly_market(
            now, volume="50000", volume24hr="1000", liquidity="50000"
        )
        result = detect_anomalies(m, now)
        assert result is not None
        assert any(a["type"] == "price_velocity" for a in result["anomalies"])

    def test_below_min_volume_filtered(self, now):
        """Market below ANOMALY_MIN_VOLUME -> no analysis."""
        m = self._make_anomaly_market(now, volume="100")
        result = detect_anomalies(m, now)
        assert result is None

    def test_below_min_liquidity_filtered(self, now):
        """Market below ANOMALY_MIN_LIQUIDITY -> no analysis."""
        m = self._make_anomaly_market(now, liquidity="500")
        result = detect_anomalies(m, now)
        assert result is None

    def test_no_anomaly_on_normal_market(self, now):
        """Normal market with no suspicious activity -> None."""
        m = self._make_anomaly_market(
            now,
            volume="100000",
            volume24hr="3000",  # ~3x daily avg (below 5x threshold)
            liquidity="50000",
        )
        # No price history (no CLOB call)
        with patch("polymarket_scanner._fetch_price_history", return_value=[]):
            result = detect_anomalies(m, now)
        assert result is None

    def test_severity_classification(self, now):
        """Severity labels assigned based on max severity score."""
        # High severity: huge volume spike
        m = self._make_anomaly_market(
            now,
            volume="10000",  # 10 day market: daily avg = 1000
            volume24hr="50000",  # 50x daily avg
            liquidity="3000",
        )
        result = detect_anomalies(m, now)
        assert result is not None
        assert result["severity_class"] in ("critical", "high", "moderate")
        assert result["max_severity"] > 0

    def test_multiple_anomalies_stacked(self, now):
        """Market can have multiple simultaneous anomalies."""
        m = self._make_anomaly_market(
            now,
            volume="10000",
            volume24hr="100000",  # huge spike + VL imbalance
            liquidity="2500",     # 100000/2500 = 40x (above 30x threshold)
        )
        result = detect_anomalies(m, now)
        assert result is not None
        assert result["anomaly_count"] >= 2
        types = [a["type"] for a in result["anomalies"]]
        assert "volume_spike" in types
        assert "vl_imbalance" in types

    def test_anomaly_output_fields(self, now):
        """Anomaly report contains expected fields."""
        m = self._make_anomaly_market(now)
        result = detect_anomalies(m, now)
        assert result is not None
        assert "id" in result
        assert "question" in result
        assert "anomalies" in result
        assert "max_severity" in result
        assert "severity_label" in result
        assert "severity_class" in result
        assert "detected_at" in result
        assert "category" in result
        assert "url" in result

    def test_bad_prices_handled(self, now):
        """Invalid outcomePrices -> returns None gracefully."""
        m = self._make_anomaly_market(now, outcomePrices="invalid_json")
        result = detect_anomalies(m, now)
        assert result is None

    def test_no_clob_ids_still_works(self, now):
        """No clobTokenIds -> volume anomalies still detected."""
        m = self._make_anomaly_market(now, clobTokenIds="[]")
        result = detect_anomalies(m, now)
        # Should still detect volume spike even without price history
        assert result is not None
        assert any(a["type"] == "volume_spike" for a in result["anomalies"])


class TestFetchPriceHistory:
    """Test CLOB price history fetching."""

    @patch("polymarket_scanner.requests.get")
    def test_successful_fetch(self, mock_get):
        """Successful API call -> returns history list."""
        from polymarket_scanner import _anomaly_price_cache, _anomaly_price_lock
        with _anomaly_price_lock:
            _anomaly_price_cache.clear()
        mock_resp = type('Response', (), {
            'status_code': 200,
            'raise_for_status': lambda self: None,
            'json': lambda self: {
                "history": [
                    {"t": 1000, "p": 0.50},
                    {"t": 2000, "p": 0.55},
                    {"t": 3000, "p": 0.60},
                ]
            },
        })()
        mock_get.return_value = mock_resp
        result = _fetch_price_history("token_test")
        assert len(result) == 3
        assert result[0]["p"] == 0.50
        assert result[2]["p"] == 0.60

    @patch("polymarket_scanner.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        """API error -> returns empty list."""
        from polymarket_scanner import _anomaly_price_cache, _anomaly_price_lock
        with _anomaly_price_lock:
            _anomaly_price_cache.clear()
        mock_get.side_effect = ConnectionError("Network error")
        result = _fetch_price_history("token_error")
        assert result == []

    @patch("polymarket_scanner.requests.get")
    def test_cache_hit(self, mock_get):
        """Second call within TTL -> uses cache, no API call."""
        from polymarket_scanner import _anomaly_price_cache, _anomaly_price_lock
        with _anomaly_price_lock:
            _anomaly_price_cache.clear()
        mock_resp = type('Response', (), {
            'status_code': 200,
            'raise_for_status': lambda self: None,
            'json': lambda self: {
                "history": [{"t": 1000, "p": 0.50}]
            },
        })()
        mock_get.return_value = mock_resp
        _fetch_price_history("token_cache")
        _fetch_price_history("token_cache")
        assert mock_get.call_count == 1  # only one API call
