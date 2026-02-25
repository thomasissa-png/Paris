"""
Polymarket Scanner — Test Suite v3
===================================
Run: python -m pytest tests/ -v

EV-first philosophy, no luck:
- Arb garanti (sum < 1.0, >2%) = super/interesting, guaranteed profit
- Near-certain (>90%) = watch only, EV ~$0, involves luck
- Overround (sum > 1.0) = excluded entirely
- Arb <2% = excluded (unprofitable after ~2% fees)
"""

import json
import math
import pytest
from datetime import datetime, timezone, timedelta

from polymarket_scanner import (
    EST_FEE_PCT,
    MAX_ANN_ROI,
    MIN_LIQUIDITY,
    classify,
    compute_score,
    extract_category,
    is_crypto,
    parse_date,
    parse_float,
    process_market,
    truncate,
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

    # --- Too certain ---

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
        tier, _, _ = classify(0.997, 0.003, 0.998)
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

    def test_filter_crypto_market(self, now):
        m = _make_market(now, question="Will Bitcoin hit $200k?")
        assert process_market(m, now) is None

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
        """All results must have fee, net, warning fields."""
        m = _make_market(now)
        result = process_market(m, now)
        assert result is not None
        assert "ev_per_100" in result
        assert "risk_per_100" in result
        assert "fee_per_100" in result
        assert "net_per_100" in result
        assert "arb_warning" in result
        assert "ann_roi_capped" in result
