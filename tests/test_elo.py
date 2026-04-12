"""
Tests for the ELO rating system.

Verifies:
- Zero-sum property: total rating points conserved per match
- Expected score symmetry: E_A + E_B = 1.0
- Transitivity: if A >> B and B >> C, A's rating > C's rating
- Win/loss/draw update correctness
- History tracking
- Persistence (save/load)
"""

import math
import tempfile
from pathlib import Path

import pytest

from src.core.elo import EloRatingSystem, MatchRecord


class TestExpectedScore:
    """Tests for the expected score calculation."""

    def test_equal_ratings_give_half(self):
        elo = EloRatingSystem()
        assert elo.expected_score(1500, 1500) == pytest.approx(0.5)

    def test_symmetry(self):
        """E_A + E_B should always equal 1.0."""
        elo = EloRatingSystem()
        for ra, rb in [(1500, 1500), (1600, 1400), (1800, 1200), (2000, 1000)]:
            ea = elo.expected_score(ra, rb)
            eb = elo.expected_score(rb, ra)
            assert ea + eb == pytest.approx(1.0), f"Failed for {ra} vs {rb}"

    def test_higher_rated_favored(self):
        elo = EloRatingSystem()
        assert elo.expected_score(1600, 1400) > 0.5
        assert elo.expected_score(1400, 1600) < 0.5

    def test_400_point_gap(self):
        """A 400-point gap should give ~91% expected score."""
        elo = EloRatingSystem()
        expected = elo.expected_score(1900, 1500)
        assert expected == pytest.approx(1.0 / (1.0 + 10 ** (-1.0)), rel=1e-5)

    def test_extreme_gap(self):
        """Very large gaps should approach 1.0 or 0.0."""
        elo = EloRatingSystem()
        assert elo.expected_score(3000, 1000) > 0.99
        assert elo.expected_score(1000, 3000) < 0.01


class TestRatingUpdates:
    """Tests for rating update mechanics."""

    def test_zero_sum_after_win(self):
        """Total rating change should sum to zero."""
        elo = EloRatingSystem()
        new_a, new_b = elo.update_ratings("A", "B", 1.0)
        delta_a = new_a - 1500
        delta_b = new_b - 1500
        assert delta_a + delta_b == pytest.approx(0.0)

    def test_zero_sum_after_loss(self):
        elo = EloRatingSystem()
        new_a, new_b = elo.update_ratings("A", "B", 0.0)
        delta_a = new_a - 1500
        delta_b = new_b - 1500
        assert delta_a + delta_b == pytest.approx(0.0)

    def test_zero_sum_after_draw(self):
        elo = EloRatingSystem()
        new_a, new_b = elo.update_ratings("A", "B", 0.5)
        delta_a = new_a - 1500
        delta_b = new_b - 1500
        assert delta_a + delta_b == pytest.approx(0.0)

    def test_winner_gains_rating(self):
        elo = EloRatingSystem()
        new_a, new_b = elo.update_ratings("A", "B", 1.0)
        assert new_a > 1500
        assert new_b < 1500

    def test_loser_loses_rating(self):
        elo = EloRatingSystem()
        new_a, new_b = elo.update_ratings("A", "B", 0.0)
        assert new_a < 1500
        assert new_b > 1500

    def test_draw_no_change_equal_ratings(self):
        """Equal-rated players drawing should have no rating change."""
        elo = EloRatingSystem()
        new_a, new_b = elo.update_ratings("A", "B", 0.5)
        assert new_a == pytest.approx(1500.0)
        assert new_b == pytest.approx(1500.0)

    def test_upset_gives_larger_change(self):
        """An upset (lower-rated wins) should produce larger rating changes."""
        elo = EloRatingSystem()
        # Set up unequal ratings
        elo._ratings["Strong"] = 1700
        elo._ratings["Weak"] = 1300

        # Normal result (strong wins)
        elo2 = EloRatingSystem()
        elo2._ratings["Strong"] = 1700
        elo2._ratings["Weak"] = 1300

        # Upset: Weak beats Strong
        new_weak_upset, new_strong_upset = elo.update_ratings("Weak", "Strong", 1.0)
        # Expected: Strong beats Weak
        new_strong_expected, new_weak_expected = elo2.update_ratings("Strong", "Weak", 1.0)

        weak_upset_gain = new_weak_upset - 1300
        strong_expected_gain = new_strong_expected - 1700
        assert weak_upset_gain > strong_expected_gain

    def test_k_factor_scales_changes(self):
        """Higher K-factor should produce larger rating changes."""
        elo_low_k = EloRatingSystem(k_factor=16)
        elo_high_k = EloRatingSystem(k_factor=64)

        new_a_low, _ = elo_low_k.update_ratings("A", "B", 1.0)
        new_a_high, _ = elo_high_k.update_ratings("A", "B", 1.0)

        assert abs(new_a_high - 1500) > abs(new_a_low - 1500)


class TestTransitivity:
    """Test that ELO ratings are transitive."""

    def test_transitive_ordering(self):
        """If A consistently beats B and B consistently beats C, A > B > C."""
        elo = EloRatingSystem()
        # A beats B many times
        for _ in range(20):
            elo.record_win("A", "B")
        # B beats C many times
        for _ in range(20):
            elo.record_win("B", "C")

        assert elo.get_rating("A") > elo.get_rating("B")
        assert elo.get_rating("B") > elo.get_rating("C")
        # Transitive: A > C
        assert elo.get_rating("A") > elo.get_rating("C")


class TestConvenienceMethods:
    """Test convenience methods for recording results."""

    def test_record_win(self):
        elo = EloRatingSystem()
        new_winner, new_loser = elo.record_win("W", "L")
        assert new_winner > 1500
        assert new_loser < 1500

    def test_record_draw(self):
        elo = EloRatingSystem()
        new_a, new_b = elo.record_draw("A", "B")
        assert new_a == pytest.approx(1500.0)
        assert new_b == pytest.approx(1500.0)


class TestLeaderboard:
    """Test leaderboard functionality."""

    def test_leaderboard_sorted_descending(self):
        elo = EloRatingSystem()
        elo.record_win("Alpha", "Beta")
        elo.record_win("Alpha", "Gamma")
        elo.record_win("Beta", "Gamma")

        board = elo.leaderboard()
        ratings = [r for _, r in board]
        assert ratings == sorted(ratings, reverse=True)

    def test_leaderboard_top_n(self):
        elo = EloRatingSystem()
        for name in ["A", "B", "C", "D", "E"]:
            elo.record_win(name, "Baseline")

        board = elo.leaderboard(top_n=3)
        assert len(board) == 3

    def test_game_specific_leaderboard(self):
        elo = EloRatingSystem()
        elo.update_ratings("A", "B", 1.0, game_type="debate")
        elo.update_ratings("C", "D", 1.0, game_type="poker")

        debate_board = elo.leaderboard(game_type="debate")
        player_names = {name for name, _ in debate_board}
        assert "A" in player_names
        assert "B" in player_names
        # C and D only played poker, not in debate leaderboard
        assert "C" not in player_names


class TestHistory:
    """Test match history tracking."""

    def test_history_recorded(self):
        elo = EloRatingSystem()
        elo.record_win("A", "B")
        assert len(elo.get_history()) == 1

    def test_history_filter_by_player(self):
        elo = EloRatingSystem()
        elo.record_win("A", "B")
        elo.record_win("C", "D")

        history_a = elo.get_history(player="A")
        assert len(history_a) == 1
        assert history_a[0].player_a == "A"

    def test_history_filter_by_game_type(self):
        elo = EloRatingSystem()
        elo.update_ratings("A", "B", 1.0, game_type="debate")
        elo.update_ratings("A", "B", 0.0, game_type="poker")

        debate_history = elo.get_history(game_type="debate")
        assert len(debate_history) == 1
        assert debate_history[0].game_type == "debate"

    def test_rating_history_chronological(self):
        elo = EloRatingSystem()
        elo.record_win("A", "B")
        elo.record_win("A", "C")
        elo.record_win("A", "D")

        history = elo.rating_history("A")
        assert len(history) == 3
        # Ratings should be increasing (A keeps winning)
        ratings = [r for _, r in history]
        assert all(ratings[i] <= ratings[i + 1] for i in range(len(ratings) - 1))


class TestPersistence:
    """Test save/load functionality."""

    def test_save_and_load(self):
        elo = EloRatingSystem(k_factor=24)
        elo.record_win("A", "B", game_type="debate")
        elo.record_win("B", "C", game_type="poker")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        elo.save(path)
        loaded = EloRatingSystem.load(path)

        assert loaded.k_factor == 24
        assert loaded.get_rating("A") == pytest.approx(elo.get_rating("A"))
        assert loaded.get_rating("B") == pytest.approx(elo.get_rating("B"))
        assert loaded.get_rating("C") == pytest.approx(elo.get_rating("C"))
        assert len(loaded.get_history()) == 2

        # Clean up
        Path(path).unlink()

    def test_round_trip_preserves_game_ratings(self):
        elo = EloRatingSystem()
        elo.update_ratings("X", "Y", 1.0, game_type="debate")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        elo.save(path)
        loaded = EloRatingSystem.load(path)

        assert loaded.get_rating("X", game_type="debate") == pytest.approx(
            elo.get_rating("X", game_type="debate")
        )

        Path(path).unlink()


class TestAllPlayers:
    """Test player listing."""

    def test_all_players_empty(self):
        elo = EloRatingSystem()
        assert elo.all_players() == []

    def test_all_players_after_matches(self):
        elo = EloRatingSystem()
        elo.record_win("A", "B")
        elo.record_win("C", "D")
        players = set(elo.all_players())
        assert players == {"A", "B", "C", "D"}
