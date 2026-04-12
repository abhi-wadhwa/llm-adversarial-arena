"""
ELO Rating System for the LLM Adversarial Arena.

Implements the standard ELO rating system (Arpad Elo, 1960) adapted for
tracking LLM performance across different adversarial games. Supports
per-game-type ratings, rating history tracking, and leaderboard generation.

Key properties:
- Zero-sum: total rating points are conserved in every match.
- Transitive: if A >> B and B >> C, the system will eventually reflect A >> C.
- K-factor of 32 (standard for new/active players).
- Initial rating of 1500.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MatchRecord:
    """Record of a single match for audit/history."""

    player_a: str
    player_b: str
    score_a: float  # 1.0 = win, 0.5 = draw, 0.0 = loss
    score_b: float
    rating_a_before: float
    rating_b_before: float
    rating_a_after: float
    rating_b_after: float
    game_type: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class EloRatingSystem:
    """Standard ELO rating system with per-game-type support.

    Attributes:
        k_factor: Determines how much ratings change per match (default 32).
        initial_rating: Starting rating for new players (default 1500).
    """

    DEFAULT_K = 32
    DEFAULT_INITIAL = 1500.0

    def __init__(
        self,
        k_factor: float = DEFAULT_K,
        initial_rating: float = DEFAULT_INITIAL,
    ) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        # Global ratings: player_name -> rating
        self._ratings: dict[str, float] = {}
        # Per-game-type ratings: game_type -> player_name -> rating
        self._game_ratings: dict[str, dict[str, float]] = {}
        # Full match history
        self._history: list[MatchRecord] = []

    def get_rating(self, player: str, game_type: str | None = None) -> float:
        """Get a player's current rating.

        Args:
            player: Player name/identifier.
            game_type: If provided, return game-specific rating.

        Returns:
            Current ELO rating.
        """
        if game_type:
            return self._game_ratings.get(game_type, {}).get(
                player, self.initial_rating
            )
        return self._ratings.get(player, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate the expected score for player A against player B.

        Uses the standard logistic curve:
            E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        Args:
            rating_a: Player A's rating.
            rating_b: Player B's rating.

        Returns:
            Expected score in [0, 1].
        """
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        player_a: str,
        player_b: str,
        score_a: float,
        game_type: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            player_a: Name of player A.
            player_b: Name of player B.
            score_a: Player A's score (1.0 = win, 0.5 = draw, 0.0 = loss).
            game_type: Type of game played.
            metadata: Optional extra information to store with the record.

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        score_b = 1.0 - score_a

        # Get current ratings (both global and game-specific)
        global_a = self._ratings.get(player_a, self.initial_rating)
        global_b = self._ratings.get(player_b, self.initial_rating)

        if game_type not in self._game_ratings:
            self._game_ratings[game_type] = {}
        game_a = self._game_ratings[game_type].get(player_a, self.initial_rating)
        game_b = self._game_ratings[game_type].get(player_b, self.initial_rating)

        # Compute expected scores using global ratings
        expected_a = self.expected_score(global_a, global_b)
        expected_b = 1.0 - expected_a

        # Update global ratings
        new_global_a = global_a + self.k_factor * (score_a - expected_a)
        new_global_b = global_b + self.k_factor * (score_b - expected_b)
        self._ratings[player_a] = new_global_a
        self._ratings[player_b] = new_global_b

        # Update game-specific ratings
        game_expected_a = self.expected_score(game_a, game_b)
        game_expected_b = 1.0 - game_expected_a
        self._game_ratings[game_type][player_a] = game_a + self.k_factor * (
            score_a - game_expected_a
        )
        self._game_ratings[game_type][player_b] = game_b + self.k_factor * (
            score_b - game_expected_b
        )

        # Record history
        record = MatchRecord(
            player_a=player_a,
            player_b=player_b,
            score_a=score_a,
            score_b=score_b,
            rating_a_before=global_a,
            rating_b_before=global_b,
            rating_a_after=new_global_a,
            rating_b_after=new_global_b,
            game_type=game_type,
            metadata=metadata or {},
        )
        self._history.append(record)

        return new_global_a, new_global_b

    def record_win(
        self, winner: str, loser: str, game_type: str = "general"
    ) -> tuple[float, float]:
        """Convenience method to record a decisive win."""
        return self.update_ratings(winner, loser, 1.0, game_type)

    def record_draw(
        self, player_a: str, player_b: str, game_type: str = "general"
    ) -> tuple[float, float]:
        """Convenience method to record a draw."""
        return self.update_ratings(player_a, player_b, 0.5, game_type)

    def leaderboard(
        self, game_type: str | None = None, top_n: int | None = None
    ) -> list[tuple[str, float]]:
        """Get sorted leaderboard.

        Args:
            game_type: If specified, return game-specific leaderboard.
            top_n: Limit to top N players.

        Returns:
            List of (player_name, rating) tuples sorted by rating descending.
        """
        if game_type:
            ratings = self._game_ratings.get(game_type, {})
        else:
            ratings = self._ratings

        sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            sorted_ratings = sorted_ratings[:top_n]
        return sorted_ratings

    def get_history(
        self,
        player: str | None = None,
        game_type: str | None = None,
    ) -> list[MatchRecord]:
        """Get match history, optionally filtered.

        Args:
            player: Filter to matches involving this player.
            game_type: Filter to this game type.

        Returns:
            List of MatchRecord objects.
        """
        records = self._history
        if player:
            records = [
                r
                for r in records
                if r.player_a == player or r.player_b == player
            ]
        if game_type:
            records = [r for r in records if r.game_type == game_type]
        return records

    def rating_history(self, player: str) -> list[tuple[str, float]]:
        """Get a player's rating over time.

        Returns:
            List of (timestamp, rating) tuples in chronological order.
        """
        points: list[tuple[str, float]] = []
        for record in self._history:
            if record.player_a == player:
                points.append((record.timestamp, record.rating_a_after))
            elif record.player_b == player:
                points.append((record.timestamp, record.rating_b_after))
        return points

    def all_players(self) -> list[str]:
        """Return a list of all registered players."""
        return list(self._ratings.keys())

    def save(self, path: str | Path) -> None:
        """Persist ratings and history to a JSON file."""
        data = {
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating,
            "ratings": self._ratings,
            "game_ratings": self._game_ratings,
            "history": [
                {
                    "player_a": r.player_a,
                    "player_b": r.player_b,
                    "score_a": r.score_a,
                    "score_b": r.score_b,
                    "rating_a_before": r.rating_a_before,
                    "rating_b_before": r.rating_b_before,
                    "rating_a_after": r.rating_a_after,
                    "rating_b_after": r.rating_b_after,
                    "game_type": r.game_type,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata,
                }
                for r in self._history
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EloRatingSystem:
        """Load ratings and history from a JSON file."""
        data = json.loads(Path(path).read_text())
        system = cls(
            k_factor=data["k_factor"],
            initial_rating=data["initial_rating"],
        )
        system._ratings = data["ratings"]
        system._game_ratings = data["game_ratings"]
        system._history = [
            MatchRecord(**record) for record in data["history"]
        ]
        return system
