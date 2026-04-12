"""
Tests for the Arena orchestrator and integration tests.

Verifies:
- Player registration and retrieval
- Running games through the arena interface
- ELO updates after games
- Leaderboard generation
- Head-to-head statistics
- Game result tracking
- Kuhn Poker game protocol
"""

import json

import pytest

from src.core.arena import Arena, GameResult, GameType
from src.core.bluffing import KuhnPokerConfig, KuhnPokerEngine, Card, Action
from src.core.debate import DebateConfig
from src.core.elo import EloRatingSystem
from src.core.llm_player import MockPlayer
from src.core.negotiation import (
    NegotiationConfig,
    NegotiationIssue,
    UtilityFunction,
)
from src.core.strategies import StrategyLibrary


@pytest.fixture
def arena() -> Arena:
    return Arena()


@pytest.fixture
def mock_players() -> tuple[MockPlayer, MockPlayer, MockPlayer]:
    player_a = MockPlayer(
        name="AlphaBot",
        responses=[
            "AI will revolutionize healthcare.",
            "The evidence is overwhelming.",
            "In conclusion, AI is beneficial.",
        ],
    )
    player_b = MockPlayer(
        name="BetaBot",
        responses=[
            "AI poses existential risks.",
            "We must proceed with caution.",
            "The risks are too great.",
        ],
    )
    judge = MockPlayer(
        name="JudgeBot",
        responses=[
            json.dumps({
                "for": {
                    "logical_coherence": 8,
                    "evidence_quality": 7,
                    "rebuttal_effectiveness": 7,
                    "persuasiveness": 8,
                    "total": 30,
                    "reasoning": "Strong arguments.",
                },
                "against": {
                    "logical_coherence": 7,
                    "evidence_quality": 6,
                    "rebuttal_effectiveness": 6,
                    "persuasiveness": 7,
                    "total": 26,
                    "reasoning": "Decent but weaker.",
                },
            })
        ],
    )
    return player_a, player_b, judge


class TestPlayerRegistration:
    """Test player management."""

    def test_register_and_get(self, arena):
        player = MockPlayer(name="Test")
        arena.register_player("test", player)
        assert arena.get_player("test") is player

    def test_get_nonexistent_raises(self, arena):
        with pytest.raises(KeyError, match="not found"):
            arena.get_player("nonexistent")

    def test_list_players(self, arena):
        arena.register_player("a", MockPlayer(name="A"))
        arena.register_player("b", MockPlayer(name="B"))
        players = arena.list_players()
        assert "a" in players
        assert "b" in players


class TestDebateIntegration:
    """Test running debates through the arena."""

    def test_run_debate(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="AI is good", num_rounds=2)
        result = arena.run_debate("alpha", "beta", config, judge=judge)

        assert isinstance(result, GameResult)
        assert result.game_type == GameType.DEBATE
        assert result.winner in ("alpha", "beta", "draw")

    def test_elo_updated_after_debate(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        result = arena.run_debate("alpha", "beta", config, judge=judge)

        # Winner should have higher ELO than initial
        if result.winner == "alpha":
            assert result.rating_a_after > 1500
            assert result.rating_b_after < 1500
        elif result.winner == "beta":
            assert result.rating_b_after > 1500
            assert result.rating_a_after < 1500

    def test_result_stored_in_history(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        arena.run_debate("alpha", "beta", config, judge=judge)

        results = arena.results()
        assert len(results) == 1
        assert results[0].game_type == GameType.DEBATE


class TestNegotiationIntegration:
    """Test running negotiations through the arena."""

    def test_run_negotiation(self, arena):
        player_a = MockPlayer(
            name="Buyer",
            responses=[
                json.dumps({
                    "action": "offer",
                    "offer": {"price": 50.0},
                    "message": "My offer.",
                })
            ],
        )
        player_b = MockPlayer(
            name="Seller",
            responses=[
                json.dumps({
                    "action": "accept",
                    "message": "Deal.",
                })
            ],
        )

        arena.register_player("buyer", player_a)
        arena.register_player("seller", player_b)

        config = NegotiationConfig(
            issues=[NegotiationIssue("price", 0, 100)],
            utility_a=UtilityFunction(
                weights={"price": 1.0}, prefer_high={"price": False}, batna=0.1
            ),
            utility_b=UtilityFunction(
                weights={"price": 1.0}, prefer_high={"price": True}, batna=0.1
            ),
            max_rounds=3,
        )

        result = arena.run_negotiation("buyer", "seller", config)
        assert isinstance(result, GameResult)
        assert result.game_type == GameType.NEGOTIATION


class TestBluffingIntegration:
    """Test running Kuhn Poker through the arena."""

    def test_run_bluffing(self, arena):
        player_a = MockPlayer(
            name="PokerA",
            responses=[json.dumps({"action": "check"})],
        )
        player_b = MockPlayer(
            name="PokerB",
            responses=[json.dumps({"action": "check"})],
        )

        arena.register_player("poker_a", player_a)
        arena.register_player("poker_b", player_b)

        config = KuhnPokerConfig(num_hands=4, seed=42)
        result = arena.run_bluffing("poker_a", "poker_b", config)

        assert isinstance(result, GameResult)
        assert result.game_type == GameType.BLUFFING

    def test_bluffing_elo_updated(self, arena):
        player_a = MockPlayer(
            name="PokerA",
            responses=[json.dumps({"action": "bet"})],
        )
        player_b = MockPlayer(
            name="PokerB",
            responses=[json.dumps({"action": "fold"})],
        )

        arena.register_player("poker_a", player_a)
        arena.register_player("poker_b", player_b)

        config = KuhnPokerConfig(num_hands=10, seed=42)
        result = arena.run_bluffing("poker_a", "poker_b", config)

        # The betting/folding player should have definite ELO changes
        assert result.rating_a_after != 1500 or result.rating_b_after != 1500


class TestKuhnPokerEngine:
    """Direct tests of the Kuhn Poker engine."""

    def test_hand_count(self):
        engine = KuhnPokerEngine()
        player_a = MockPlayer(
            name="A",
            responses=[json.dumps({"action": "check"})],
        )
        player_b = MockPlayer(
            name="B",
            responses=[json.dumps({"action": "check"})],
        )

        config = KuhnPokerConfig(num_hands=6, seed=123)
        result = engine.run(player_a, player_b, config)
        assert len(result.hands) == 6

    def test_zero_sum_per_hand(self):
        """Each hand should be zero-sum (p1_profit + p2_profit = 0)."""
        engine = KuhnPokerEngine()
        player_a = MockPlayer(
            name="A",
            responses=[json.dumps({"action": "bet"})],
        )
        player_b = MockPlayer(
            name="B",
            responses=[json.dumps({"action": "call"})],
        )

        config = KuhnPokerConfig(num_hands=10, seed=42)
        result = engine.run(player_a, player_b, config)

        for hand in result.hands:
            assert hand.p1_profit + hand.p2_profit == 0, (
                f"Hand {hand.hand_number} not zero-sum: "
                f"p1={hand.p1_profit}, p2={hand.p2_profit}"
            )

    def test_fold_loses_ante(self):
        """Folding should lose exactly the ante."""
        engine = KuhnPokerEngine()
        # P1 bets, P2 folds
        player_a = MockPlayer(
            name="A",
            responses=[json.dumps({"action": "bet"})],
        )
        player_b = MockPlayer(
            name="B",
            responses=[json.dumps({"action": "fold"})],
        )

        config = KuhnPokerConfig(num_hands=2, seed=42)
        result = engine.run(player_a, player_b, config)

        for hand in result.hands:
            if not hand.went_to_showdown:
                # Folder loses ante (1), winner gains ante (1)
                if hand.winner_position == 1:
                    assert hand.p1_profit == 1
                    assert hand.p2_profit == -1
                else:
                    assert hand.p1_profit == -1
                    assert hand.p2_profit == 1

    def test_showdown_higher_card_wins(self):
        """At showdown, the higher card should always win."""
        engine = KuhnPokerEngine()
        player_a = MockPlayer(
            name="A",
            responses=[json.dumps({"action": "check"})],
        )
        player_b = MockPlayer(
            name="B",
            responses=[json.dumps({"action": "check"})],
        )

        config = KuhnPokerConfig(num_hands=20, seed=42)
        result = engine.run(player_a, player_b, config)

        for hand in result.hands:
            if hand.went_to_showdown:
                if hand.card_p1.value > hand.card_p2.value:
                    assert hand.winner_position == 1
                else:
                    assert hand.winner_position == 2

    def test_card_values(self):
        """Test card enum values."""
        assert Card.JACK.value == 1
        assert Card.QUEEN.value == 2
        assert Card.KING.value == 3
        assert Card.JACK.value < Card.QUEEN.value < Card.KING.value


class TestLeaderboard:
    """Test arena leaderboard functionality."""

    def test_leaderboard_after_games(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        arena.run_debate("alpha", "beta", config, judge=judge)

        board = arena.leaderboard()
        assert len(board) == 2
        # Should be sorted descending
        assert board[0][1] >= board[1][1]

    def test_game_type_leaderboard(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        arena.run_debate("alpha", "beta", config, judge=judge)

        debate_board = arena.leaderboard(game_type="debate")
        assert len(debate_board) == 2


class TestHeadToHead:
    """Test head-to-head statistics."""

    def test_head_to_head(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        arena.run_debate("alpha", "beta", config, judge=judge)

        h2h = arena.head_to_head("alpha", "beta")
        assert h2h["total_matches"] == 1
        assert h2h["alpha_wins"] + h2h["beta_wins"] + h2h["draws"] == 1


class TestResultFiltering:
    """Test result filtering."""

    def test_filter_by_game_type(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        arena.run_debate("alpha", "beta", config, judge=judge)

        debate_results = arena.results(game_type=GameType.DEBATE)
        assert len(debate_results) == 1

        nego_results = arena.results(game_type=GameType.NEGOTIATION)
        assert len(nego_results) == 0

    def test_filter_by_player(self, arena, mock_players):
        player_a, player_b, judge = mock_players
        arena.register_player("alpha", player_a)
        arena.register_player("beta", player_b)

        config = DebateConfig(proposition="Test", num_rounds=2)
        arena.run_debate("alpha", "beta", config, judge=judge)

        alpha_results = arena.results(player="alpha")
        assert len(alpha_results) == 1

        gamma_results = arena.results(player="gamma")
        assert len(gamma_results) == 0


class TestStrategyLibrary:
    """Test strategy library integration."""

    def test_strategies_loaded(self):
        lib = StrategyLibrary()
        names = lib.strategy_names()
        assert "analytical" in names
        assert "aggressive" in names
        assert "cooperative" in names
        assert "socratic" in names
        assert "deceptive" in names
        assert "adaptive" in names

    def test_strategy_get_prompt(self):
        lib = StrategyLibrary()
        prompt = lib.get_prompt("analytical", "debate")
        assert len(prompt) > 0
        assert "debat" in prompt.lower()

    def test_strategy_fallback_to_base(self):
        lib = StrategyLibrary()
        strategy = lib.get("analytical")
        base = strategy.get_prompt("nonexistent_game")
        assert base == strategy.base_prompt

    def test_unknown_strategy_raises(self):
        lib = StrategyLibrary()
        with pytest.raises(KeyError, match="not found"):
            lib.get("nonexistent_strategy")
