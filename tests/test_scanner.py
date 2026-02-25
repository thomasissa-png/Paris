"""
Polymarket Scanner — Test Suite
================================
Run: python -m pytest tests/ -v
Run: bash run_tests.sh

Must pass before every commit.
"""

import json
import math
import pytest
from datetime import datetime, timezone, timedelta

from polymarket_scanner import (
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
# classify()
# =========================================================================

class TestClassify:
    """Test tier classification logic."""

    def test_super_near_certain(self):
        tier, label, rtype = classify(0.95, 0.01)
        assert tier == "super"
        assert rtype == "near_certain"

    def test_super_mispricing(self):
        tier, label, rtype = classify(0.50, 0.06)
        assert tier == "super"
        assert rtype == "mispricing"

    def test_interesting_high_confidence(self):
        tier, label, rtype = classify(0.85, 0.01)
        assert tier == "interesting"
        assert rtype == "near_certain"

    def test_interesting_moderate_mispricing(self):
        tier, label, rtype = classify(0.50, 0.03)
        assert tier == "interesting"
        assert rtype == "mispricing"

    def test_watch_imbalance(self):
        tier, label, rtype = classify(0.75, 0.005)
        assert tier == "watch"
        assert rtype == "near_certain"

    def test_watch_slight_mispricing(self):
        tier, label, rtype = classify(0.50, 0.015)
        assert tier == "watch"
        assert rtype == "mispricing"

    def test_too_certain_filtered(self):
        """Markets > 99.5% offer negligible profit — should be filtered out."""
        tier, label, rtype = classify(0.998, 0.005)
        assert tier is None

    def test_boundary_995_is_filtered(self):
        """0.996 > 0.995 — should be filtered."""
        tier, _, _ = classify(0.996, 0.005)
        assert tier is None

    def test_just_below_995_still_super(self):
        """0.994 ≤ 0.995 — should still be classified as super."""
        tier, _, _ = classify(0.994, 0.005)
        assert tier == "super"

    def test_not_interesting(self):
        tier, label, rtype = classify(0.60, 0.005)
        assert tier is None
        assert label == ""
        assert rtype == ""

    def test_boundary_90_is_not_super(self):
        """0.90 exactly should NOT trigger super (requires > 0.90)."""
        tier, _, _ = classify(0.90, 0.005)
        assert tier != "super"

    def test_boundary_80_is_not_interesting(self):
        """0.80 exactly should NOT trigger interesting (requires > 0.80)."""
        tier, _, _ = classify(0.80, 0.005)
        assert tier != "interesting"

    def test_near_certain_takes_priority_over_mispricing(self):
        """When max_price > 0.90, tier is super even if deviation is also high."""
        tier, _, rtype = classify(0.95, 0.10)
        assert tier == "super"
        assert rtype == "near_certain"


# =========================================================================
# compute_score()
# =========================================================================

class TestComputeScore:
    """Test composite attractiveness scoring."""

    def test_sooner_is_higher_score(self):
        s_soon = compute_score(0.95, 0.01, 2, 50000)
        s_late = compute_score(0.95, 0.01, 50, 50000)
        assert s_soon > s_late

    def test_more_liquid_is_higher_score(self):
        s_liq = compute_score(0.95, 0.01, 10, 100000)
        s_dry = compute_score(0.95, 0.01, 10, 1000)
        assert s_liq > s_dry

    def test_higher_certainty_is_higher_score(self):
        s_high = compute_score(0.98, 0.01, 10, 50000)
        s_low = compute_score(0.75, 0.01, 10, 50000)
        assert s_high > s_low

    def test_score_is_positive(self):
        s = compute_score(0.80, 0.01, 30, 5000)
        assert s > 0

    def test_mispricing_uses_deviation(self):
        """When max_price < 0.70, score uses deviation * 15."""
        s = compute_score(0.50, 0.06, 5, 10000)
        assert s > 0

    def test_very_short_time_does_not_explode(self):
        """days_left near zero should be capped, not cause division by zero."""
        s = compute_score(0.95, 0.01, 0.01, 50000)
        assert math.isfinite(s)


# =========================================================================
# is_crypto() — Regex correctness
# =========================================================================

class TestCryptoRegex:
    """Ensure no false positives on ambiguous short tokens."""

    # Should MATCH (true crypto)
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

    # Should NOT match (false positives from removed short tokens)
    @pytest.mark.parametrize("q", [
        "Will the resolution pass the Senate?",      # NOT "sol"
        "Whether the bill is approved",               # NOT "eth"
        "Canada election results",                     # NOT "ada"
        "The linked document shows evidence",          # NOT "link"
        "Dot plot from the Fed meeting",               # NOT "dot"
        "NBA season opener tonight",                    # unrelated
        "Will Trump win the election?",                 # politics
        "Solar energy investment growth",               # "sol" in "solar"
        "Ethical AI guidelines adopted",                # "eth" in "ethical"
        "Adaptation strategy for climate change",       # "ada" in "adaptation"
    ])
    def test_crypto_not_detected(self, q):
        assert is_crypto(q) is False, f"Should NOT detect crypto: {q}"


# =========================================================================
# extract_category()
# =========================================================================

class TestExtractCategory:
    """Test category extraction from market data."""

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
        """API tags should override heuristic detection."""
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
        assert dt.month == 3

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
# process_market() — Full pipeline
# =========================================================================

class TestProcessMarket:
    """Integration tests for the full market processing pipeline."""

    def test_valid_binary_market(self, now):
        m = _make_market(now)
        result = process_market(m, now)
        assert result is not None
        assert result["tier"] == "super"
        assert result["yes"] == 0.94
        assert result["no"] == 0.06
        assert result["is_binary"] is True
        assert result["category"] == "Politics"

    def test_question_not_truncated(self, now):
        long_q = "Will the very important Supreme Court ruling affect the upcoming presidential election results significantly?" * 2
        m = _make_market(now, question=long_q)
        result = process_market(m, now)
        assert result is not None
        assert result["question"] == long_q  # NO truncation

    def test_trade_recommendation_buy_yes(self, now):
        m = _make_market(now, outcomePrices='["0.94", "0.06"]')
        result = process_market(m, now)
        assert result is not None
        assert "BUY" in result["trade_label"]
        assert "YES" in result["trade_label"]
        assert result["trade_side_class"] == "yes"
        assert result["guaranteed"] is False

    def test_trade_recommendation_arb(self, now):
        """sum < 1.0 = guaranteed arbitrage."""
        m = _make_market(now, outcomePrices='["0.45", "0.48"]')
        result = process_market(m, now)
        assert result is not None
        assert "ARB" in result["trade_label"]
        assert result["trade_side_class"] == "arb"
        assert result["guaranteed"] is True
        assert result["profit_100"] > 0

    def test_roi_calculation(self, now):
        m = _make_market(now, outcomePrices='["0.94", "0.06"]')
        result = process_market(m, now)
        assert result is not None
        # Profit per $100 of YES at 0.94: 100*(1/0.94 - 1) ≈ $6.38
        assert abs(result["profit_100"] - 6.38) < 0.1
        # Annualized should be positive and > basic ROI
        assert result["ann_roi"] > 0

    def test_multi_outcome_market(self, now):
        m = _make_market(
            now,
            question="Who will win the GOP primary?",
            outcomePrices='["0.82", "0.10", "0.05", "0.03"]',
            outcomes='["Trump", "DeSantis", "Haley", "Other"]',
        )
        result = process_market(m, now)
        assert result is not None
        assert result["is_binary"] is False
        assert result["num_outcomes"] == 4
        assert "TRUMP" in result["trade_label"]
        assert result["yes_label"] == "Trump"
        assert "Others" in result["no_label"]

    def test_vol_liq_ratio_and_thin(self, now):
        # Thin market: low liquidity
        m = _make_market(now, liquidity="1500")
        result = process_market(m, now)
        assert result is not None
        assert result["thin"] is True

    def test_vol_liq_ratio_healthy(self, now):
        m = _make_market(now, volume="10000", liquidity="50000")
        result = process_market(m, now)
        assert result is not None
        assert result["thin"] is False

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
        """Markets resolving in < 1 day should show hours."""
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
        m = _make_market(now, liquidity="100")
        assert process_market(m, now) is None

    def test_filter_crypto_market(self, now):
        m = _make_market(now, question="Will Bitcoin hit $200k?")
        assert process_market(m, now) is None

    def test_filter_no_end_date(self, now):
        m = _make_market(now, endDate=None)
        assert process_market(m, now) is None

    def test_filter_too_certain(self, now):
        """Markets > 99.5% offer negligible profit — filtered out."""
        m = _make_market(now, outcomePrices='["0.998", "0.002"]')
        assert process_market(m, now) is None

    def test_filter_not_interesting(self, now):
        """Market at 50/50 with no mispricing should be filtered."""
        m = _make_market(now, outcomePrices='["0.50", "0.50"]')
        assert process_market(m, now) is None

    def test_filter_invalid_prices(self, now):
        m = _make_market(now, outcomePrices="invalid_json")
        assert process_market(m, now) is None

    def test_filter_single_price(self, now):
        m = _make_market(now, outcomePrices='["0.94"]')
        assert process_market(m, now) is None

    # --- Edge cases ---

    def test_zero_price_no_crash(self, now):
        m = _make_market(now, outcomePrices='["0.00", "1.00"]')
        result = process_market(m, now)
        # 1.00 > 0.995 → filtered as too certain
        assert result is None

    def test_outcomes_as_list_not_string(self, now):
        """Handle outcomes already parsed as list (not JSON string)."""
        m = _make_market(now, outcomes=["Yes", "No"])
        result = process_market(m, now)
        assert result is not None
        assert result["yes_label"] == "Yes"

    def test_prices_as_list_not_string(self, now):
        """Handle outcomePrices already parsed as list."""
        m = _make_market(now, outcomePrices=[0.94, 0.06])
        result = process_market(m, now)
        assert result is not None

    def test_missing_outcomes_field(self, now):
        """Should fall back gracefully if outcomes field is missing."""
        m = _make_market(now)
        del m["outcomes"]
        result = process_market(m, now)
        assert result is not None

    def test_volume_24h_passthrough(self, now):
        m = _make_market(now, volume24hr="12345")
        result = process_market(m, now)
        assert result is not None
        assert result["volume_24h"] == 12345.0

    def test_mispricing_overpriced(self, now):
        """sum > 1.0 mispricing — should buy cheapest side."""
        m = _make_market(now, outcomePrices='["0.55", "0.52"]')
        result = process_market(m, now)
        assert result is not None
        assert result["guaranteed"] is False
        assert result["reason_type"] == "mispricing"
