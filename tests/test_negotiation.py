"""
Tests for the Negotiation Engine.

Uses MockPlayer to test negotiation protocol without API calls.
Verifies:
- Turn structure and alternation
- Offer parsing and clamping
- Utility function evaluation
- Accept/reject/counter-offer flows
- Discount factor application
- BATNA fallback
"""

import json

import pytest

from src.core.llm_player import MockPlayer
from src.core.negotiation import (
    NegotiationAction,
    NegotiationConfig,
    NegotiationEngine,
    NegotiationIssue,
    NegotiationResult,
    UtilityFunction,
)


@pytest.fixture
def engine() -> NegotiationEngine:
    return NegotiationEngine()


@pytest.fixture
def issues() -> list[NegotiationIssue]:
    return [
        NegotiationIssue("price", 0.0, 100.0, "Item price"),
        NegotiationIssue("warranty", 0.0, 5.0, "Warranty years"),
    ]


@pytest.fixture
def utility_buyer() -> UtilityFunction:
    """Buyer wants low price, high warranty."""
    return UtilityFunction(
        weights={"price": 2.0, "warranty": 1.0},
        prefer_high={"price": False, "warranty": True},
        batna=0.2,
    )


@pytest.fixture
def utility_seller() -> UtilityFunction:
    """Seller wants high price, low warranty."""
    return UtilityFunction(
        weights={"price": 2.0, "warranty": 1.0},
        prefer_high={"price": True, "warranty": False},
        batna=0.2,
    )


def _make_config(
    issues: list[NegotiationIssue],
    utility_a: UtilityFunction,
    utility_b: UtilityFunction,
    max_rounds: int = 3,
    discount: float = 0.95,
) -> NegotiationConfig:
    return NegotiationConfig(
        issues=issues,
        utility_a=utility_a,
        utility_b=utility_b,
        max_rounds=max_rounds,
        discount_factor=discount,
        scenario_description="Test negotiation",
    )


class TestUtilityFunction:
    """Test utility function evaluation."""

    def test_buyer_prefers_low_price(self, issues, utility_buyer):
        low_price = {"price": 10.0, "warranty": 3.0}
        high_price = {"price": 90.0, "warranty": 3.0}
        assert utility_buyer.evaluate(low_price, issues) > utility_buyer.evaluate(
            high_price, issues
        )

    def test_seller_prefers_high_price(self, issues, utility_seller):
        low_price = {"price": 10.0, "warranty": 3.0}
        high_price = {"price": 90.0, "warranty": 3.0}
        assert utility_seller.evaluate(high_price, issues) > utility_seller.evaluate(
            low_price, issues
        )

    def test_max_utility_buyer(self, issues, utility_buyer):
        """Buyer's best deal: lowest price, highest warranty."""
        best = {"price": 0.0, "warranty": 5.0}
        u = utility_buyer.evaluate(best, issues)
        # weights: price=2.0, warranty=1.0; both normalized to 1.0
        assert u == pytest.approx(3.0)

    def test_min_utility_buyer(self, issues, utility_buyer):
        """Buyer's worst deal: highest price, lowest warranty."""
        worst = {"price": 100.0, "warranty": 0.0}
        u = utility_buyer.evaluate(worst, issues)
        assert u == pytest.approx(0.0)

    def test_weights_affect_utility(self, issues):
        """Higher weight issues should have more impact."""
        u_heavy_price = UtilityFunction(
            weights={"price": 10.0, "warranty": 1.0},
            prefer_high={"price": True, "warranty": True},
        )
        u_heavy_warranty = UtilityFunction(
            weights={"price": 1.0, "warranty": 10.0},
            prefer_high={"price": True, "warranty": True},
        )
        offer = {"price": 50.0, "warranty": 2.5}
        # Both should give same utility for midpoint since normalized = 0.5
        assert u_heavy_price.evaluate(offer, issues) == pytest.approx(
            10.0 * 0.5 + 1.0 * 0.5
        )
        assert u_heavy_warranty.evaluate(offer, issues) == pytest.approx(
            1.0 * 0.5 + 10.0 * 0.5
        )


class TestNegotiationProtocol:
    """Test the negotiation protocol."""

    def test_agreement_reached(self, engine, issues, utility_buyer, utility_seller):
        """Test that an accept action leads to agreement."""
        player_a = MockPlayer(
            name="Buyer",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 50.0, "warranty": 3.0},
                    "message": "I offer 50 for 3 years warranty.",
                })
            ],
        )
        player_b = MockPlayer(
            name="Seller",
            responses=[
                json.dumps({
                    "action": "accept",
                    "message": "I accept your offer.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller)
        result = engine.run(player_a, player_b, config)
        assert result.agreed is True
        assert result.final_offer is not None

    def test_no_agreement_all_rejects(self, engine, issues, utility_buyer, utility_seller):
        """If all rounds are offers with no accept, should fail to agree."""
        player_a = MockPlayer(
            name="Buyer",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 20.0, "warranty": 4.0},
                    "message": "Low price please.",
                })
            ],
        )
        player_b = MockPlayer(
            name="Seller",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 80.0, "warranty": 1.0},
                    "message": "I want more.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller, max_rounds=4)
        result = engine.run(player_a, player_b, config)
        assert result.agreed is False
        assert result.utility_a == utility_buyer.batna
        assert result.utility_b == utility_seller.batna

    def test_turn_alternation(self, engine, issues, utility_buyer, utility_seller):
        """Players should alternate making offers."""
        player_a = MockPlayer(
            name="A",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 50.0, "warranty": 3.0},
                    "message": "My offer.",
                })
            ],
        )
        player_b = MockPlayer(
            name="B",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 60.0, "warranty": 2.0},
                    "message": "Counter.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller, max_rounds=4)
        result = engine.run(player_a, player_b, config)

        # Check alternating names
        names = [t.player_name for t in result.turns]
        for i in range(len(names) - 1):
            assert names[i] != names[i + 1], f"Same player at turns {i} and {i + 1}"


class TestOfferClamping:
    """Test that offers are clamped to valid ranges."""

    def test_values_clamped_to_range(self, engine, issues, utility_buyer, utility_seller):
        """Values outside the valid range should be clamped."""
        player_a = MockPlayer(
            name="A",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 200.0, "warranty": -1.0},
                    "message": "Extreme offer.",
                })
            ],
        )
        player_b = MockPlayer(
            name="B",
            responses=[
                json.dumps({
                    "action": "accept",
                    "message": "OK.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller)
        result = engine.run(player_a, player_b, config)
        assert result.agreed is True
        # Price clamped to 100, warranty clamped to 0
        assert result.final_offer is not None
        assert result.final_offer["price"] == 100.0
        assert result.final_offer["warranty"] == 0.0


class TestDiscountFactor:
    """Test discount factor application."""

    def test_later_rounds_worth_less(self, engine, issues, utility_buyer, utility_seller):
        """Utilities should decrease in later rounds due to discounting."""
        # Create a negotiation that accepts in round 2 (turn 2)
        player_a = MockPlayer(
            name="A",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 50.0, "warranty": 2.5},
                    "message": "First offer.",
                })
            ],
        )
        player_b = MockPlayer(
            name="B",
            responses=[
                json.dumps({
                    "action": "accept",
                    "message": "Deal.",
                })
            ],
        )

        config_fast = _make_config(issues, utility_buyer, utility_seller, discount=0.5)
        result = engine.run(player_a, player_b, config_fast)

        # With discount=0.5 and agreement at round 2, discount = 0.5^1 = 0.5
        # Utilities should be about half of undiscounted values
        assert result.agreed
        # The utility values should be discounted
        undiscounted_a = utility_buyer.evaluate(
            {"price": 50.0, "warranty": 2.5}, issues
        )
        # Agreement at round 2, discount = 0.5^(2-1) = 0.5
        assert result.utility_a == pytest.approx(undiscounted_a * 0.5)


class TestEloConversion:
    """Test ELO score conversion from negotiation results."""

    def test_higher_utility_wins(self, engine, issues, utility_buyer, utility_seller):
        player_a = MockPlayer(
            name="A",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 20.0, "warranty": 5.0},
                    "message": "Great deal for me!",
                })
            ],
        )
        player_b = MockPlayer(
            name="B",
            responses=[
                json.dumps({
                    "action": "accept",
                    "message": "I accept.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller, discount=1.0)
        result = engine.run(player_a, player_b, config)

        if result.utility_a > result.utility_b:
            assert result.elo_score_a == 1.0
        elif result.utility_b > result.utility_a:
            assert result.elo_score_a == 0.0

    def test_no_agreement_is_draw(self, engine, issues, utility_buyer, utility_seller):
        player_a = MockPlayer(
            name="A",
            responses=[
                json.dumps({
                    "action": "reject",
                    "message": "No.",
                })
            ],
        )
        player_b = MockPlayer(
            name="B",
            responses=[
                json.dumps({
                    "action": "reject",
                    "message": "No.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller, max_rounds=2)
        result = engine.run(player_a, player_b, config)
        assert result.elo_score_a == 0.5


class TestTranscript:
    """Test transcript generation."""

    def test_transcript_contains_scenario(self, engine, issues, utility_buyer, utility_seller):
        player_a = MockPlayer(
            name="A",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 50.0, "warranty": 3.0},
                    "message": "My offer.",
                })
            ],
        )
        player_b = MockPlayer(
            name="B",
            responses=[
                json.dumps({
                    "action": "accept",
                    "message": "Agreed.",
                })
            ],
        )

        config = _make_config(issues, utility_buyer, utility_seller)
        result = engine.run(player_a, player_b, config)
        transcript = result.transcript()
        assert "Test negotiation" in transcript
