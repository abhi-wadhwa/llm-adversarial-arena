"""
Kuhn Poker Engine for the LLM Adversarial Arena.

Implements Kuhn Poker, a simplified poker variant used in game theory
research. It's the simplest interesting poker game, with a known
Nash equilibrium, making it ideal for evaluating LLM strategic reasoning.

Rules:
- 3-card deck: Jack (J=1), Queen (Q=2), King (K=3)
- Each player antes 1 chip and receives one card
- Player 1 acts first: check or bet (1 chip)
- If Player 1 checks:
    - Player 2 can check (showdown) or bet (1 chip)
    - If Player 2 bets, Player 1 can call (1 chip) or fold
- If Player 1 bets:
    - Player 2 can call (1 chip) or fold
- Showdown: higher card wins the pot
- Nash equilibrium involves mixed strategies with specific bluffing frequencies

References:
    Kuhn, H.W. (1950). "Simplified Two-Person Poker"
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.llm_player import LLMPlayer, Message


class Card(int, Enum):
    """Cards in Kuhn Poker, ordered by rank."""

    JACK = 1
    QUEEN = 2
    KING = 3

    def __str__(self) -> str:
        return self.name[0]  # J, Q, K

    @property
    def full_name(self) -> str:
        return self.name.capitalize()


class Action(str, Enum):
    """Possible actions in Kuhn Poker."""

    CHECK = "check"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"


@dataclass
class KuhnPokerConfig:
    """Configuration for a Kuhn Poker match.

    Attributes:
        num_hands: Number of hands to play (players alternate positions).
        ante: Initial ante per player.
        bet_size: Size of a bet.
        seed: Random seed for card dealing (None for random).
    """

    num_hands: int = 20
    ante: int = 1
    bet_size: int = 1
    seed: int | None = None


@dataclass
class HandAction:
    """A single action within a hand."""

    player_name: str
    position: int  # 1 or 2
    action: Action
    pot_after: int


@dataclass
class HandResult:
    """Result of a single hand of Kuhn Poker."""

    hand_number: int
    card_p1: Card
    card_p2: Card
    actions: list[HandAction]
    winner_position: int  # 1, 2, or 0 for fold
    pot: int
    p1_profit: int  # positive = won, negative = lost
    p2_profit: int
    went_to_showdown: bool


@dataclass
class KuhnPokerResult:
    """Complete result of a Kuhn Poker match.

    Attributes:
        config: Game configuration.
        hands: Results of each hand.
        total_p1: Player 1's total profit/loss.
        total_p2: Player 2's total profit/loss.
        player_1_name: Player 1's name.
        player_2_name: Player 2's name.
        bluff_stats: Statistics about bluffing behavior.
    """

    config: KuhnPokerConfig
    hands: list[HandResult]
    total_p1: int
    total_p2: int
    player_1_name: str
    player_2_name: str
    bluff_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def elo_score_p1(self) -> float:
        """Convert to ELO score for player 1."""
        if self.total_p1 > self.total_p2:
            return 1.0
        elif self.total_p2 > self.total_p1:
            return 0.0
        return 0.5

    def transcript(self) -> str:
        """Generate a human-readable transcript."""
        lines = [
            f"KUHN POKER MATCH",
            f"{'=' * 60}",
            f"Player 1: {self.player_1_name}  |  Player 2: {self.player_2_name}",
            f"Hands: {self.config.num_hands} | Ante: {self.config.ante} | Bet: {self.config.bet_size}",
            f"{'=' * 60}",
            "",
        ]

        for hand in self.hands:
            lines.append(f"--- Hand {hand.hand_number} ---")
            lines.append(f"Cards: P1={hand.card_p1} | P2={hand.card_p2}")
            for action in hand.actions:
                lines.append(
                    f"  P{action.position} ({action.player_name}): "
                    f"{action.action.value} (pot: {action.pot_after})"
                )
            if hand.went_to_showdown:
                lines.append(f"  SHOWDOWN: P{hand.winner_position} wins pot of {hand.pot}")
            else:
                folder = 1 if hand.winner_position == 2 else 2
                lines.append(f"  P{folder} FOLDS: P{hand.winner_position} wins pot of {hand.pot}")
            lines.append(f"  P1: {hand.p1_profit:+d} | P2: {hand.p2_profit:+d}")
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"FINAL RESULTS")
        lines.append(f"  {self.player_1_name}: {self.total_p1:+d}")
        lines.append(f"  {self.player_2_name}: {self.total_p2:+d}")

        if self.bluff_stats:
            lines.append("")
            lines.append("BLUFF STATISTICS:")
            for key, val in self.bluff_stats.items():
                lines.append(f"  {key}: {val}")

        return "\n".join(lines)


class KuhnPokerEngine:
    """Engine for running Kuhn Poker matches between LLM players."""

    def run(
        self,
        player_a: LLMPlayer,
        player_b: LLMPlayer,
        config: KuhnPokerConfig,
    ) -> KuhnPokerResult:
        """Run a complete Kuhn Poker match.

        Players alternate who acts first each hand.

        Args:
            player_a: First player.
            player_b: Second player.
            config: Game configuration.

        Returns:
            KuhnPokerResult with full hand history and statistics.
        """
        player_a.reset()
        player_b.reset()

        rng = random.Random(config.seed)
        hands: list[HandResult] = []
        total_p1 = 0
        total_p2 = 0

        # Track bluffing statistics
        bluff_attempts: dict[str, int] = {player_a.name: 0, player_b.name: 0}
        bluff_successes: dict[str, int] = {player_a.name: 0, player_b.name: 0}
        total_actions: dict[str, int] = {player_a.name: 0, player_b.name: 0}

        for hand_num in range(1, config.num_hands + 1):
            # Alternate positions
            if hand_num % 2 == 1:
                p1, p2 = player_a, player_b
            else:
                p1, p2 = player_b, player_a

            # Deal cards
            deck = list(Card)
            rng.shuffle(deck)
            card_p1 = deck[0]
            card_p2 = deck[1]

            # Play the hand
            hand_result = self._play_hand(
                hand_num, p1, p2, card_p1, card_p2, config
            )
            hands.append(hand_result)

            # Track totals (map back to original player_a/player_b)
            if hand_num % 2 == 1:
                total_p1 += hand_result.p1_profit
                total_p2 += hand_result.p2_profit
            else:
                total_p1 += hand_result.p2_profit
                total_p2 += hand_result.p1_profit

            # Track bluffing
            for action_record in hand_result.actions:
                total_actions[action_record.player_name] = (
                    total_actions.get(action_record.player_name, 0) + 1
                )
                # A bluff is betting/calling with a weaker hand
                if action_record.action in (Action.BET, Action.CALL):
                    acting_card = (
                        card_p1 if action_record.position == 1 else card_p2
                    )
                    other_card = (
                        card_p2 if action_record.position == 1 else card_p1
                    )
                    if acting_card.value < other_card.value:
                        bluff_attempts[action_record.player_name] = (
                            bluff_attempts.get(action_record.player_name, 0) + 1
                        )
                        # Bluff succeeded if opponent folded after this
                        if (
                            not hand_result.went_to_showdown
                            and hand_result.winner_position == action_record.position
                        ):
                            bluff_successes[action_record.player_name] = (
                                bluff_successes.get(action_record.player_name, 0) + 1
                            )

        bluff_stats = {}
        for name in [player_a.name, player_b.name]:
            attempts = bluff_attempts.get(name, 0)
            successes = bluff_successes.get(name, 0)
            total = total_actions.get(name, 0)
            bluff_stats[f"{name}_bluff_attempts"] = attempts
            bluff_stats[f"{name}_bluff_successes"] = successes
            bluff_stats[f"{name}_bluff_rate"] = (
                f"{attempts / total:.1%}" if total > 0 else "N/A"
            )

        return KuhnPokerResult(
            config=config,
            hands=hands,
            total_p1=total_p1,
            total_p2=total_p2,
            player_1_name=player_a.name,
            player_2_name=player_b.name,
            bluff_stats=bluff_stats,
        )

    def _play_hand(
        self,
        hand_num: int,
        p1: LLMPlayer,
        p2: LLMPlayer,
        card_p1: Card,
        card_p2: Card,
        config: KuhnPokerConfig,
    ) -> HandResult:
        """Play a single hand of Kuhn Poker."""
        pot = 2 * config.ante  # Both players ante
        actions: list[HandAction] = []

        # P1's first action: check or bet
        p1_action = self._get_action(
            p1, card_p1, 1, pot, actions, config, hand_num
        )
        if p1_action not in (Action.CHECK, Action.BET):
            p1_action = Action.CHECK  # Default to check if invalid

        pot_after = pot + (config.bet_size if p1_action == Action.BET else 0)
        actions.append(
            HandAction(
                player_name=p1.name,
                position=1,
                action=p1_action,
                pot_after=pot_after,
            )
        )
        pot = pot_after

        if p1_action == Action.CHECK:
            # P1 checked. P2 can check (showdown) or bet
            p2_action = self._get_action(
                p2, card_p2, 2, pot, actions, config, hand_num
            )
            if p2_action not in (Action.CHECK, Action.BET):
                p2_action = Action.CHECK

            pot_after = pot + (config.bet_size if p2_action == Action.BET else 0)
            actions.append(
                HandAction(
                    player_name=p2.name,
                    position=2,
                    action=p2_action,
                    pot_after=pot_after,
                )
            )
            pot = pot_after

            if p2_action == Action.CHECK:
                # Showdown: check-check
                winner = 1 if card_p1.value > card_p2.value else 2
                winnings = config.ante
                return HandResult(
                    hand_number=hand_num,
                    card_p1=card_p1,
                    card_p2=card_p2,
                    actions=actions,
                    winner_position=winner,
                    pot=pot,
                    p1_profit=winnings if winner == 1 else -winnings,
                    p2_profit=winnings if winner == 2 else -winnings,
                    went_to_showdown=True,
                )
            else:
                # P2 bet after P1 checked. P1 can call or fold
                p1_response = self._get_action(
                    p1, card_p1, 1, pot, actions, config, hand_num
                )
                if p1_response not in (Action.CALL, Action.FOLD):
                    p1_response = Action.FOLD

                pot_after = pot + (config.bet_size if p1_response == Action.CALL else 0)
                actions.append(
                    HandAction(
                        player_name=p1.name,
                        position=1,
                        action=p1_response,
                        pot_after=pot_after,
                    )
                )
                pot = pot_after

                if p1_response == Action.FOLD:
                    # P2 wins
                    return HandResult(
                        hand_number=hand_num,
                        card_p1=card_p1,
                        card_p2=card_p2,
                        actions=actions,
                        winner_position=2,
                        pot=pot,
                        p1_profit=-config.ante,
                        p2_profit=config.ante,
                        went_to_showdown=False,
                    )
                else:
                    # Showdown: check-bet-call
                    winner = 1 if card_p1.value > card_p2.value else 2
                    winnings = config.ante + config.bet_size
                    return HandResult(
                        hand_number=hand_num,
                        card_p1=card_p1,
                        card_p2=card_p2,
                        actions=actions,
                        winner_position=winner,
                        pot=pot,
                        p1_profit=winnings if winner == 1 else -winnings,
                        p2_profit=winnings if winner == 2 else -winnings,
                        went_to_showdown=True,
                    )
        else:
            # P1 bet. P2 can call or fold
            p2_action = self._get_action(
                p2, card_p2, 2, pot, actions, config, hand_num
            )
            if p2_action not in (Action.CALL, Action.FOLD):
                p2_action = Action.FOLD

            pot_after = pot + (config.bet_size if p2_action == Action.CALL else 0)
            actions.append(
                HandAction(
                    player_name=p2.name,
                    position=2,
                    action=p2_action,
                    pot_after=pot_after,
                )
            )
            pot = pot_after

            if p2_action == Action.FOLD:
                # P1 wins
                return HandResult(
                    hand_number=hand_num,
                    card_p1=card_p1,
                    card_p2=card_p2,
                    actions=actions,
                    winner_position=1,
                    pot=pot,
                    p1_profit=config.ante,
                    p2_profit=-config.ante,
                    went_to_showdown=False,
                )
            else:
                # Showdown: bet-call
                winner = 1 if card_p1.value > card_p2.value else 2
                winnings = config.ante + config.bet_size
                return HandResult(
                    hand_number=hand_num,
                    card_p1=card_p1,
                    card_p2=card_p2,
                    actions=actions,
                    winner_position=winner,
                    pot=pot,
                    p1_profit=winnings if winner == 1 else -winnings,
                    p2_profit=winnings if winner == 2 else -winnings,
                    went_to_showdown=True,
                )

    def _get_action(
        self,
        player: LLMPlayer,
        card: Card,
        position: int,
        pot: int,
        history: list[HandAction],
        config: KuhnPokerConfig,
        hand_num: int,
    ) -> Action:
        """Get an action from an LLM player."""
        # Determine valid actions based on game state
        if not history:
            valid_actions = [Action.CHECK, Action.BET]
        else:
            last_action = history[-1].action
            if last_action == Action.CHECK:
                valid_actions = [Action.CHECK, Action.BET]
            elif last_action == Action.BET:
                valid_actions = [Action.CALL, Action.FOLD]
            else:
                valid_actions = [Action.CHECK, Action.BET]

        # Build prompt
        history_str = ""
        if history:
            history_str = "Action history this hand:\n"
            for h in history:
                history_str += f"  P{h.position}: {h.action.value}\n"

        valid_str = " or ".join(a.value for a in valid_actions)

        prompt = (
            f"You are playing Kuhn Poker (Hand #{hand_num}).\n"
            f"Your card: {card.full_name} (rank: {card.value}/3)\n"
            f"Your position: Player {position} "
            f"({'first to act' if position == 1 else 'second to act'})\n"
            f"Current pot: {pot} chips\n"
            f"Ante: {config.ante} | Bet size: {config.bet_size}\n\n"
            f"{history_str}\n"
            f"Valid actions: {valid_str}\n\n"
            f"Rules reminder:\n"
            f"- 3-card deck: Jack (1) < Queen (2) < King (3)\n"
            f"- Higher card wins at showdown\n"
            f"- Folding forfeits your ante\n\n"
            f'Respond with ONLY a JSON object: {{"action": "{valid_actions[0].value}"}}\n'
            f"Choose strategically. Consider bluffing and opponent tendencies."
        )

        messages = [Message(role="user", content=prompt)]
        response = player.generate(messages)

        # Parse action
        return self._parse_action(response, valid_actions)

    def _parse_action(self, response: str, valid_actions: list[Action]) -> Action:
        """Parse an action from the LLM response."""
        # Try JSON parsing
        json_match = re.search(r"\{[\s\S]*?\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                action_str = data.get("action", "").lower()
                for action in valid_actions:
                    if action.value == action_str:
                        return action
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback: look for action keywords in response
        response_lower = response.lower()
        for action in valid_actions:
            if action.value in response_lower:
                return action

        # Default to first valid action
        return valid_actions[0]
