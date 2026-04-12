"""
Arena -- top-level game orchestration for the LLM Adversarial Arena.

The Arena class ties together all game engines, players, the ELO rating
system, and the strategy library. It provides a single entry point for
running matches and tracking results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.bluffing import KuhnPokerConfig, KuhnPokerEngine, KuhnPokerResult
from src.core.debate import DebateConfig, DebateEngine, DebateResult
from src.core.elo import EloRatingSystem
from src.core.llm_player import LLMPlayer
from src.core.negotiation import NegotiationConfig, NegotiationEngine, NegotiationResult
from src.core.strategies import GameContext, StrategyLibrary


class GameType(str, Enum):
    """Supported game types."""

    DEBATE = "debate"
    NEGOTIATION = "negotiation"
    BLUFFING = "bluffing"


@dataclass
class GameResult:
    """Unified result wrapper for any game type.

    Attributes:
        game_type: Which game was played.
        player_a: Name of player A.
        player_b: Name of player B.
        winner: Name of the winning player, or "draw".
        score_a: ELO-compatible score for player A (1.0/0.5/0.0).
        score_b: ELO-compatible score for player B.
        details: The game-specific result object.
        transcript: Human-readable transcript.
        rating_a_after: Player A's ELO after the match.
        rating_b_after: Player B's ELO after the match.
    """

    game_type: GameType
    player_a: str
    player_b: str
    winner: str
    score_a: float
    score_b: float
    details: DebateResult | NegotiationResult | KuhnPokerResult
    transcript: str
    rating_a_after: float = 0.0
    rating_b_after: float = 0.0


class Arena:
    """Main orchestrator for the LLM Adversarial Arena.

    Manages players, runs games, tracks ELO ratings, and stores results.

    Usage:
        arena = Arena()
        arena.register_player("gpt4", player_gpt4)
        arena.register_player("claude", player_claude)
        result = arena.run_debate("gpt4", "claude", config)
        print(arena.leaderboard())
    """

    def __init__(
        self,
        elo_system: EloRatingSystem | None = None,
        strategy_library: StrategyLibrary | None = None,
    ) -> None:
        self.elo = elo_system or EloRatingSystem()
        self.strategies = strategy_library or StrategyLibrary()
        self._players: dict[str, LLMPlayer] = {}
        self._results: list[GameResult] = []
        self._debate_engine = DebateEngine()
        self._negotiation_engine = NegotiationEngine()
        self._bluffing_engine = KuhnPokerEngine()

    def register_player(self, name: str, player: LLMPlayer) -> None:
        """Register a player in the arena.

        Args:
            name: Unique identifier for the player.
            player: The LLMPlayer instance.
        """
        self._players[name] = player

    def get_player(self, name: str) -> LLMPlayer:
        """Retrieve a registered player.

        Raises:
            KeyError: If the player is not registered.
        """
        if name not in self._players:
            available = ", ".join(sorted(self._players.keys()))
            raise KeyError(f"Player '{name}' not found. Registered: {available}")
        return self._players[name]

    def list_players(self) -> list[str]:
        """Return all registered player names."""
        return list(self._players.keys())

    def run_debate(
        self,
        player_for_name: str,
        player_against_name: str,
        config: DebateConfig,
        judge_name: str | None = None,
        judge: LLMPlayer | None = None,
    ) -> GameResult:
        """Run a debate between two registered players.

        Args:
            player_for_name: Name of the player arguing FOR.
            player_against_name: Name of the player arguing AGAINST.
            config: Debate configuration.
            judge_name: Name of a registered player to act as judge.
            judge: An LLMPlayer to use as judge (overrides judge_name).

        Returns:
            GameResult with debate details and updated ELO.
        """
        player_for = self.get_player(player_for_name)
        player_against = self.get_player(player_against_name)

        if judge is None:
            if judge_name:
                judge = self.get_player(judge_name)
            else:
                # Use the FOR player as a fallback judge (in practice, use a separate model)
                judge = player_for

        result = self._debate_engine.run(player_for, player_against, judge, config)

        # Determine ELO scores
        score_a = result.elo_score_for
        score_b = 1.0 - score_a

        # Determine winner name
        if result.winner == "for":
            winner = player_for_name
        elif result.winner == "against":
            winner = player_against_name
        else:
            winner = "draw"

        # Update ELO
        new_a, new_b = self.elo.update_ratings(
            player_for_name,
            player_against_name,
            score_a,
            game_type=GameType.DEBATE.value,
            metadata={"proposition": config.proposition},
        )

        game_result = GameResult(
            game_type=GameType.DEBATE,
            player_a=player_for_name,
            player_b=player_against_name,
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            details=result,
            transcript=result.transcript(),
            rating_a_after=new_a,
            rating_b_after=new_b,
        )
        self._results.append(game_result)
        return game_result

    def run_negotiation(
        self,
        player_a_name: str,
        player_b_name: str,
        config: NegotiationConfig,
    ) -> GameResult:
        """Run a negotiation between two registered players.

        Args:
            player_a_name: Name of the first-mover player.
            player_b_name: Name of the second-mover player.
            config: Negotiation configuration.

        Returns:
            GameResult with negotiation details and updated ELO.
        """
        player_a = self.get_player(player_a_name)
        player_b = self.get_player(player_b_name)

        result = self._negotiation_engine.run(player_a, player_b, config)

        score_a = result.elo_score_a
        score_b = 1.0 - score_a

        if score_a > score_b:
            winner = player_a_name
        elif score_b > score_a:
            winner = player_b_name
        else:
            winner = "draw"

        new_a, new_b = self.elo.update_ratings(
            player_a_name,
            player_b_name,
            score_a,
            game_type=GameType.NEGOTIATION.value,
            metadata={"scenario": config.scenario_description},
        )

        game_result = GameResult(
            game_type=GameType.NEGOTIATION,
            player_a=player_a_name,
            player_b=player_b_name,
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            details=result,
            transcript=result.transcript(),
            rating_a_after=new_a,
            rating_b_after=new_b,
        )
        self._results.append(game_result)
        return game_result

    def run_bluffing(
        self,
        player_a_name: str,
        player_b_name: str,
        config: KuhnPokerConfig,
    ) -> GameResult:
        """Run a Kuhn Poker match between two registered players.

        Args:
            player_a_name: Name of the first player.
            player_b_name: Name of the second player.
            config: Kuhn Poker configuration.

        Returns:
            GameResult with poker details and updated ELO.
        """
        player_a = self.get_player(player_a_name)
        player_b = self.get_player(player_b_name)

        result = self._bluffing_engine.run(player_a, player_b, config)

        score_a = result.elo_score_p1
        score_b = 1.0 - score_a

        if result.total_p1 > result.total_p2:
            winner = player_a_name
        elif result.total_p2 > result.total_p1:
            winner = player_b_name
        else:
            winner = "draw"

        new_a, new_b = self.elo.update_ratings(
            player_a_name,
            player_b_name,
            score_a,
            game_type=GameType.BLUFFING.value,
            metadata={
                "num_hands": config.num_hands,
                "p1_profit": result.total_p1,
                "p2_profit": result.total_p2,
            },
        )

        game_result = GameResult(
            game_type=GameType.BLUFFING,
            player_a=player_a_name,
            player_b=player_b_name,
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            details=result,
            transcript=result.transcript(),
            rating_a_after=new_a,
            rating_b_after=new_b,
        )
        self._results.append(game_result)
        return game_result

    def leaderboard(
        self, game_type: str | None = None, top_n: int | None = None
    ) -> list[tuple[str, float]]:
        """Get the current ELO leaderboard.

        Args:
            game_type: Filter to a specific game type.
            top_n: Limit to top N players.

        Returns:
            Sorted list of (player_name, rating) tuples.
        """
        return self.elo.leaderboard(game_type=game_type, top_n=top_n)

    def results(
        self, game_type: GameType | None = None, player: str | None = None
    ) -> list[GameResult]:
        """Get historical results, optionally filtered."""
        results = self._results
        if game_type:
            results = [r for r in results if r.game_type == game_type]
        if player:
            results = [
                r for r in results if r.player_a == player or r.player_b == player
            ]
        return results

    def head_to_head(self, player_a: str, player_b: str) -> dict[str, Any]:
        """Get head-to-head record between two players.

        Returns:
            Dictionary with win/loss/draw counts and per-game-type breakdown.
        """
        relevant = [
            r
            for r in self._results
            if {r.player_a, r.player_b} == {player_a, player_b}
        ]

        stats: dict[str, Any] = {
            "total_matches": len(relevant),
            f"{player_a}_wins": sum(1 for r in relevant if r.winner == player_a),
            f"{player_b}_wins": sum(1 for r in relevant if r.winner == player_b),
            "draws": sum(1 for r in relevant if r.winner == "draw"),
            "by_game_type": {},
        }

        for gt in GameType:
            gt_results = [r for r in relevant if r.game_type == gt]
            if gt_results:
                stats["by_game_type"][gt.value] = {
                    "matches": len(gt_results),
                    f"{player_a}_wins": sum(
                        1 for r in gt_results if r.winner == player_a
                    ),
                    f"{player_b}_wins": sum(
                        1 for r in gt_results if r.winner == player_b
                    ),
                    "draws": sum(1 for r in gt_results if r.winner == "draw"),
                }

        return stats

    def save_state(self, path: str | Path) -> None:
        """Save arena state (ELO ratings and history) to disk."""
        self.elo.save(path)

    def load_state(self, path: str | Path) -> None:
        """Load arena state from disk."""
        self.elo = EloRatingSystem.load(path)
