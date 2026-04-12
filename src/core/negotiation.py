"""
Negotiation Protocol Engine for the LLM Adversarial Arena.

Implements a Rubinstein-style alternating-offers bargaining protocol between
two LLM players. Each player has a utility function over a set of issues,
and they take turns proposing offers that the other can accept or reject.

Protocol:
1. Define a set of negotiable issues (e.g., price, delivery, warranty).
2. Each player receives their private utility function.
3. Player A makes the first offer (allocations for each issue).
4. Player B can accept, reject, or counter-offer.
5. Alternating offers continue until agreement or deadline.
6. A discount factor delta penalizes delay (models time pressure).
7. If no agreement by deadline, both get their BATNA (fallback).

Key concepts:
- ZOPA: Zone of Possible Agreement
- BATNA: Best Alternative to Negotiated Agreement
- Pareto Efficiency: No reallocation makes one better off without hurting the other
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.llm_player import LLMPlayer, Message


class NegotiationAction(str, Enum):
    """Possible actions in a negotiation turn."""

    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class NegotiationIssue:
    """A single negotiable issue.

    Attributes:
        name: Issue identifier (e.g., "price", "delivery_weeks").
        min_value: Minimum possible value.
        max_value: Maximum possible value.
        description: Human-readable description.
    """

    name: str
    min_value: float
    max_value: float
    description: str = ""


@dataclass
class UtilityFunction:
    """Utility function for a negotiation player.

    Maps issue allocations to a utility value. Each issue has a weight
    and a preferred direction (higher or lower is better).

    Attributes:
        weights: Importance weight for each issue (issue_name -> weight).
        prefer_high: Whether higher values are preferred (issue_name -> bool).
        batna: Utility of walking away (best alternative).
    """

    weights: dict[str, float]
    prefer_high: dict[str, bool]
    batna: float = 0.0

    def evaluate(self, offer: dict[str, float], issues: list[NegotiationIssue]) -> float:
        """Calculate total utility for a given offer.

        Normalizes each issue to [0, 1] based on preference direction,
        then takes a weighted sum.
        """
        total = 0.0
        for issue in issues:
            if issue.name not in offer:
                continue
            value = offer[issue.name]
            # Normalize to [0, 1]
            range_size = issue.max_value - issue.min_value
            if range_size == 0:
                normalized = 0.5
            elif self.prefer_high.get(issue.name, True):
                normalized = (value - issue.min_value) / range_size
            else:
                normalized = (issue.max_value - value) / range_size

            weight = self.weights.get(issue.name, 1.0)
            total += weight * normalized

        return total


@dataclass
class NegotiationConfig:
    """Configuration for a negotiation game.

    Attributes:
        issues: List of negotiable issues.
        utility_a: Player A's utility function.
        utility_b: Player B's utility function.
        max_rounds: Maximum number of offer rounds before BATNA.
        discount_factor: Per-round discount (0 < delta <= 1). Lower = more time pressure.
        scenario_description: Context for the negotiation.
    """

    issues: list[NegotiationIssue]
    utility_a: UtilityFunction
    utility_b: UtilityFunction
    max_rounds: int = 5
    discount_factor: float = 0.95
    scenario_description: str = "A negotiation between two parties."


@dataclass
class NegotiationTurn:
    """A single turn in the negotiation."""

    round_number: int
    player_name: str
    action: NegotiationAction
    offer: dict[str, float] | None  # None for accept/reject without counter
    message: str  # Natural language explanation
    utility_for_player: float = 0.0
    utility_for_opponent: float = 0.0


@dataclass
class NegotiationResult:
    """Complete result of a negotiation.

    Attributes:
        config: Negotiation configuration used.
        turns: Full turn history.
        agreed: Whether an agreement was reached.
        final_offer: The accepted offer (None if no agreement).
        utility_a: Player A's final utility (discounted).
        utility_b: Player B's final utility (discounted).
        player_a: Player A's name.
        player_b: Player B's name.
    """

    config: NegotiationConfig
    turns: list[NegotiationTurn]
    agreed: bool
    final_offer: dict[str, float] | None
    utility_a: float
    utility_b: float
    player_a: str = ""
    player_b: str = ""

    @property
    def elo_score_a(self) -> float:
        """Convert negotiation outcome to ELO score for player A."""
        if not self.agreed:
            return 0.5  # Both get BATNA, treat as draw
        if self.utility_a > self.utility_b:
            return 1.0
        elif self.utility_b > self.utility_a:
            return 0.0
        return 0.5

    def is_pareto_efficient(self) -> bool:
        """Check if the final agreement is Pareto efficient.

        A rough check: can either player's utility increase without
        decreasing the other's?
        """
        if not self.agreed or not self.final_offer:
            return False

        # Check boundary solutions
        for issue in self.config.issues:
            for test_val in [issue.min_value, issue.max_value]:
                test_offer = dict(self.final_offer)
                test_offer[issue.name] = test_val
                test_ua = self.config.utility_a.evaluate(test_offer, self.config.issues)
                test_ub = self.config.utility_b.evaluate(test_offer, self.config.issues)
                # If one improves and other doesn't worsen, not Pareto efficient
                if (test_ua > self.utility_a and test_ub >= self.utility_b) or (
                    test_ub > self.utility_b and test_ua >= self.utility_a
                ):
                    return False
        return True

    def transcript(self) -> str:
        """Generate a human-readable transcript."""
        lines = [
            f"NEGOTIATION: {self.config.scenario_description}",
            f"{'=' * 60}",
            f"Player A: {self.player_a}  |  Player B: {self.player_b}",
            f"Issues: {', '.join(i.name for i in self.config.issues)}",
            f"Max rounds: {self.config.max_rounds} | Discount: {self.config.discount_factor}",
            f"{'=' * 60}",
            "",
        ]
        for turn in self.turns:
            header = f"[Round {turn.round_number}] {turn.player_name} - {turn.action.value.upper()}"
            lines.append(header)
            if turn.offer:
                offer_str = ", ".join(f"{k}: {v:.2f}" for k, v in turn.offer.items())
                lines.append(f"  Offer: {offer_str}")
            lines.append(f"  Message: {turn.message}")
            lines.append(
                f"  Utility (self/opponent): {turn.utility_for_player:.3f} / "
                f"{turn.utility_for_opponent:.3f}"
            )
            lines.append("")

        lines.append("=" * 60)
        if self.agreed:
            lines.append(f"AGREEMENT REACHED")
            if self.final_offer:
                offer_str = ", ".join(
                    f"{k}: {v:.2f}" for k, v in self.final_offer.items()
                )
                lines.append(f"Final terms: {offer_str}")
            lines.append(f"Player A utility: {self.utility_a:.3f}")
            lines.append(f"Player B utility: {self.utility_b:.3f}")
            lines.append(f"Pareto efficient: {self.is_pareto_efficient()}")
        else:
            lines.append("NO AGREEMENT - Both parties receive BATNA")
            lines.append(f"Player A BATNA: {self.config.utility_a.batna:.3f}")
            lines.append(f"Player B BATNA: {self.config.utility_b.batna:.3f}")

        return "\n".join(lines)


class NegotiationEngine:
    """Orchestrates Rubinstein-style alternating-offers negotiations."""

    def run(
        self,
        player_a: LLMPlayer,
        player_b: LLMPlayer,
        config: NegotiationConfig,
    ) -> NegotiationResult:
        """Run a complete negotiation.

        Args:
            player_a: First mover.
            player_b: Second mover.
            config: Negotiation configuration.

        Returns:
            NegotiationResult with full history and outcomes.
        """
        player_a.reset()
        player_b.reset()

        turns: list[NegotiationTurn] = []
        conversation_a: list[Message] = []
        conversation_b: list[Message] = []

        issues_desc = self._format_issues(config.issues)
        utility_desc_a = self._format_utility(config.utility_a, "your")
        utility_desc_b = self._format_utility(config.utility_b, "your")

        agreed = False
        final_offer: dict[str, float] | None = None

        for round_num in range(1, config.max_rounds + 1):
            discount = config.discount_factor ** (round_num - 1)

            # Determine whose turn it is
            is_a_turn = round_num % 2 == 1
            current_player = player_a if is_a_turn else player_b
            other_player = player_b if is_a_turn else player_a
            current_utility = config.utility_a if is_a_turn else config.utility_b
            other_utility = config.utility_b if is_a_turn else config.utility_a
            current_conv = conversation_a if is_a_turn else conversation_b

            # Build prompt
            if round_num <= 2:
                # Initial prompt with full context
                utility_desc = utility_desc_a if is_a_turn else utility_desc_b
                prompt = (
                    f"You are negotiating in the following scenario:\n"
                    f"{config.scenario_description}\n\n"
                    f"Issues being negotiated:\n{issues_desc}\n\n"
                    f"Your utility function:\n{utility_desc}\n\n"
                    f"Discount factor per round: {config.discount_factor} "
                    f"(current discount: {discount:.3f})\n"
                    f"Rounds remaining: {config.max_rounds - round_num + 1}\n\n"
                )
                if round_num == 1:
                    prompt += "You make the first offer. "
                else:
                    last_turn = turns[-1]
                    prompt += (
                        f"Your opponent offered: {json.dumps(last_turn.offer)}\n"
                        f"Their message: \"{last_turn.message}\"\n\n"
                    )
                prompt += self._action_instructions(config.issues)
            else:
                last_turn = turns[-1]
                prompt = (
                    f"Round {round_num}/{config.max_rounds} "
                    f"(discount: {discount:.3f}).\n"
                    f"Your opponent's last action: {last_turn.action.value}\n"
                )
                if last_turn.offer:
                    prompt += f"Their offer: {json.dumps(last_turn.offer)}\n"
                prompt += (
                    f"Their message: \"{last_turn.message}\"\n\n"
                    f"{self._action_instructions(config.issues)}"
                )

            current_conv.append(Message(role="user", content=prompt))
            response = current_player.generate(current_conv)
            current_conv.append(Message(role="assistant", content=response))

            # Parse the response
            action, offer, message = self._parse_response(response, config.issues)

            # Calculate utilities
            if offer:
                u_current = current_utility.evaluate(offer, config.issues) * discount
                u_other = other_utility.evaluate(offer, config.issues) * discount
            else:
                u_current = 0.0
                u_other = 0.0

            turn = NegotiationTurn(
                round_number=round_num,
                player_name=current_player.name,
                action=action,
                offer=offer,
                message=message,
                utility_for_player=u_current,
                utility_for_opponent=u_other,
            )
            turns.append(turn)

            # Check for agreement
            if action == NegotiationAction.ACCEPT and round_num > 1:
                agreed = True
                # The accepted offer is the previous turn's offer
                final_offer = turns[-2].offer if turns[-2].offer else offer
                break

        # Calculate final utilities
        if agreed and final_offer:
            final_round = len(turns)
            final_discount = config.discount_factor ** (final_round - 1)
            utility_a = config.utility_a.evaluate(final_offer, config.issues) * final_discount
            utility_b = config.utility_b.evaluate(final_offer, config.issues) * final_discount
        else:
            utility_a = config.utility_a.batna
            utility_b = config.utility_b.batna

        return NegotiationResult(
            config=config,
            turns=turns,
            agreed=agreed,
            final_offer=final_offer,
            utility_a=utility_a,
            utility_b=utility_b,
            player_a=player_a.name,
            player_b=player_b.name,
        )

    def _format_issues(self, issues: list[NegotiationIssue]) -> str:
        """Format issues for display in prompts."""
        lines = []
        for issue in issues:
            desc = f" - {issue.description}" if issue.description else ""
            lines.append(
                f"- {issue.name}: range [{issue.min_value}, {issue.max_value}]{desc}"
            )
        return "\n".join(lines)

    def _format_utility(self, utility: UtilityFunction, possessive: str) -> str:
        """Format a utility function for display."""
        lines = [f"{possessive.capitalize()} priorities (weight / preference):"]
        for issue, weight in utility.weights.items():
            direction = "higher is better" if utility.prefer_high.get(issue, True) else "lower is better"
            lines.append(f"  - {issue}: weight={weight:.1f}, {direction}")
        lines.append(f"  - BATNA (walk-away value): {utility.batna:.2f}")
        return "\n".join(lines)

    def _action_instructions(self, issues: list[NegotiationIssue]) -> str:
        """Instructions for formatting a response."""
        issue_names = [i.name for i in issues]
        example = {name: (i.min_value + i.max_value) / 2 for name, i in zip(issue_names, issues)}
        return (
            "Respond in the following JSON format:\n"
            "{\n"
            '  "action": "offer" | "accept" | "reject",\n'
            f'  "offer": {json.dumps(example)},\n'
            '  "message": "Your explanation/reasoning"\n'
            "}\n\n"
            "If accepting, you may omit the offer field. "
            "If rejecting without a counter-offer, you may omit the offer field."
        )

    def _parse_response(
        self, response: str, issues: list[NegotiationIssue]
    ) -> tuple[NegotiationAction, dict[str, float] | None, str]:
        """Parse an LLM's negotiation response."""
        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                action_str = data.get("action", "offer").lower()
                if action_str == "accept":
                    action = NegotiationAction.ACCEPT
                elif action_str == "reject":
                    action = NegotiationAction.REJECT
                else:
                    action = NegotiationAction.OFFER

                offer = data.get("offer")
                if offer and isinstance(offer, dict):
                    # Clamp values to valid ranges
                    clamped: dict[str, float] = {}
                    for issue in issues:
                        if issue.name in offer:
                            val = float(offer[issue.name])
                            val = max(issue.min_value, min(issue.max_value, val))
                            clamped[issue.name] = val
                    offer = clamped if clamped else None

                message = data.get("message", response[:200])
                return action, offer, str(message)

            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to detect action from text
        response_lower = response.lower()
        if "accept" in response_lower:
            return NegotiationAction.ACCEPT, None, response[:200]
        elif "reject" in response_lower:
            return NegotiationAction.REJECT, None, response[:200]

        return NegotiationAction.OFFER, None, response[:200]
