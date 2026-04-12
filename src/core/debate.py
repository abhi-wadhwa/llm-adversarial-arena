"""
Debate Engine for the LLM Adversarial Arena.

Implements a structured debate protocol where two LLMs argue opposing sides
of a proposition. A judge (another LLM or rule-based system) evaluates the
quality of arguments after N rounds.

Protocol:
1. A proposition is presented.
2. Player A argues FOR the proposition.
3. Player B argues AGAINST the proposition.
4. Steps 2-3 repeat for N rounds.
5. A judge evaluates both sides and assigns scores.

The judge scores on multiple dimensions:
- Logical coherence (0-10)
- Evidence quality (0-10)
- Rebuttal effectiveness (0-10)
- Persuasiveness (0-10)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.llm_player import LLMPlayer, Message


class DebateSide(str, Enum):
    """Side of the debate."""

    FOR = "for"
    AGAINST = "against"


@dataclass
class DebateConfig:
    """Configuration for a debate game.

    Attributes:
        proposition: The statement being debated.
        num_rounds: Number of argument-rebuttal rounds.
        max_argument_tokens: Maximum tokens per argument.
        judge_system_prompt: System prompt for the judge LLM.
    """

    proposition: str
    num_rounds: int = 3
    max_argument_tokens: int = 512
    judge_system_prompt: str = (
        "You are an impartial debate judge. Evaluate arguments on four dimensions:\n"
        "1. Logical Coherence (0-10): Is the reasoning valid and consistent?\n"
        "2. Evidence Quality (0-10): Are claims supported with evidence or examples?\n"
        "3. Rebuttal Effectiveness (0-10): How well are opponent arguments addressed?\n"
        "4. Persuasiveness (0-10): How compelling is the overall presentation?\n\n"
        "Respond in JSON format:\n"
        '{"for": {"logical_coherence": N, "evidence_quality": N, '
        '"rebuttal_effectiveness": N, "persuasiveness": N, "total": N, '
        '"reasoning": "..."}, '
        '"against": {"logical_coherence": N, "evidence_quality": N, '
        '"rebuttal_effectiveness": N, "persuasiveness": N, "total": N, '
        '"reasoning": "..."}}'
    )


@dataclass
class DebateTurn:
    """A single turn in the debate."""

    round_number: int
    side: DebateSide
    player_name: str
    argument: str


@dataclass
class JudgeScores:
    """Scores from the judge for one side."""

    logical_coherence: float = 0.0
    evidence_quality: float = 0.0
    rebuttal_effectiveness: float = 0.0
    persuasiveness: float = 0.0
    total: float = 0.0
    reasoning: str = ""


@dataclass
class DebateResult:
    """Complete result of a debate.

    Attributes:
        config: The debate configuration.
        turns: List of all debate turns.
        scores_for: Judge's scores for the FOR side.
        scores_against: Judge's scores for the AGAINST side.
        winner: Which side won (or "draw").
        judge_raw_response: Raw judge output for debugging.
    """

    config: DebateConfig
    turns: list[DebateTurn]
    scores_for: JudgeScores
    scores_against: JudgeScores
    winner: str  # "for", "against", or "draw"
    judge_raw_response: str = ""
    player_for: str = ""
    player_against: str = ""

    @property
    def elo_score_for(self) -> float:
        """Convert debate result to ELO score for the FOR player."""
        if self.winner == "for":
            return 1.0
        elif self.winner == "against":
            return 0.0
        return 0.5

    def transcript(self) -> str:
        """Generate a human-readable transcript of the debate."""
        lines = [
            f"DEBATE: {self.config.proposition}",
            f"{'=' * 60}",
            f"FOR: {self.player_for}  |  AGAINST: {self.player_against}",
            f"Rounds: {self.config.num_rounds}",
            f"{'=' * 60}",
            "",
        ]
        for turn in self.turns:
            header = f"[Round {turn.round_number}] {turn.side.value.upper()} ({turn.player_name})"
            lines.append(header)
            lines.append("-" * len(header))
            lines.append(turn.argument)
            lines.append("")

        lines.append("=" * 60)
        lines.append("JUDGE'S DECISION")
        lines.append("=" * 60)
        lines.append(f"Winner: {self.winner.upper()}")
        lines.append("")
        lines.append(f"FOR scores: {self.scores_for.total}/40")
        lines.append(f"  Logical Coherence:       {self.scores_for.logical_coherence}/10")
        lines.append(f"  Evidence Quality:        {self.scores_for.evidence_quality}/10")
        lines.append(f"  Rebuttal Effectiveness:  {self.scores_for.rebuttal_effectiveness}/10")
        lines.append(f"  Persuasiveness:          {self.scores_for.persuasiveness}/10")
        lines.append(f"  Reasoning: {self.scores_for.reasoning}")
        lines.append("")
        lines.append(f"AGAINST scores: {self.scores_against.total}/40")
        lines.append(f"  Logical Coherence:       {self.scores_against.logical_coherence}/10")
        lines.append(f"  Evidence Quality:        {self.scores_against.evidence_quality}/10")
        lines.append(f"  Rebuttal Effectiveness:  {self.scores_against.rebuttal_effectiveness}/10")
        lines.append(f"  Persuasiveness:          {self.scores_against.persuasiveness}/10")
        lines.append(f"  Reasoning: {self.scores_against.reasoning}")

        return "\n".join(lines)


class DebateEngine:
    """Orchestrates structured debates between two LLM players.

    Usage:
        engine = DebateEngine()
        config = DebateConfig(proposition="AI will be net positive for humanity")
        result = engine.run(player_for, player_against, judge, config)
    """

    def run(
        self,
        player_for: LLMPlayer,
        player_against: LLMPlayer,
        judge: LLMPlayer,
        config: DebateConfig,
    ) -> DebateResult:
        """Run a complete debate.

        Args:
            player_for: LLM arguing FOR the proposition.
            player_against: LLM arguing AGAINST the proposition.
            judge: LLM that evaluates the debate.
            config: Debate configuration.

        Returns:
            DebateResult with turns, scores, and winner.
        """
        player_for.reset()
        player_against.reset()

        turns: list[DebateTurn] = []
        conversation_for: list[Message] = []
        conversation_against: list[Message] = []

        # Build initial context
        base_context_for = (
            f"You are debating FOR the following proposition:\n"
            f'"{config.proposition}"\n\n'
            f"Present compelling arguments supporting this position. "
            f"You will have {config.num_rounds} rounds to make your case."
        )
        base_context_against = (
            f"You are debating AGAINST the following proposition:\n"
            f'"{config.proposition}"\n\n'
            f"Present compelling arguments opposing this position. "
            f"You will have {config.num_rounds} rounds to make your case."
        )

        for round_num in range(1, config.num_rounds + 1):
            # --- FOR player's turn ---
            if round_num == 1:
                prompt_for = (
                    f"{base_context_for}\n\n"
                    f"Round {round_num}/{config.num_rounds}: "
                    f"Present your opening argument."
                )
            else:
                last_against = turns[-1].argument
                prompt_for = (
                    f"Round {round_num}/{config.num_rounds}.\n"
                    f"Your opponent (AGAINST) argued:\n\"{last_against}\"\n\n"
                    f"Respond with your rebuttal and further arguments FOR the proposition."
                )

            conversation_for.append(Message(role="user", content=prompt_for))
            argument_for = player_for.generate(conversation_for)
            conversation_for.append(Message(role="assistant", content=argument_for))

            turns.append(
                DebateTurn(
                    round_number=round_num,
                    side=DebateSide.FOR,
                    player_name=player_for.name,
                    argument=argument_for,
                )
            )

            # --- AGAINST player's turn ---
            if round_num == 1:
                prompt_against = (
                    f"{base_context_against}\n\n"
                    f"Round {round_num}/{config.num_rounds}.\n"
                    f"Your opponent (FOR) argued:\n\"{argument_for}\"\n\n"
                    f"Present your opening argument against the proposition and "
                    f"respond to their points."
                )
            else:
                prompt_against = (
                    f"Round {round_num}/{config.num_rounds}.\n"
                    f"Your opponent (FOR) argued:\n\"{argument_for}\"\n\n"
                    f"Respond with your rebuttal and further arguments AGAINST the proposition."
                )

            conversation_against.append(Message(role="user", content=prompt_against))
            argument_against = player_against.generate(conversation_against)
            conversation_against.append(
                Message(role="assistant", content=argument_against)
            )

            turns.append(
                DebateTurn(
                    round_number=round_num,
                    side=DebateSide.AGAINST,
                    player_name=player_against.name,
                    argument=argument_against,
                )
            )

        # --- Judge evaluation ---
        scores_for, scores_against, raw_response = self._judge_debate(
            judge, config, turns
        )

        # Determine winner
        if scores_for.total > scores_against.total:
            winner = "for"
        elif scores_against.total > scores_for.total:
            winner = "against"
        else:
            winner = "draw"

        return DebateResult(
            config=config,
            turns=turns,
            scores_for=scores_for,
            scores_against=scores_against,
            winner=winner,
            judge_raw_response=raw_response,
            player_for=player_for.name,
            player_against=player_against.name,
        )

    def _judge_debate(
        self,
        judge: LLMPlayer,
        config: DebateConfig,
        turns: list[DebateTurn],
    ) -> tuple[JudgeScores, JudgeScores, str]:
        """Have the judge evaluate the debate."""
        # Build transcript for the judge
        transcript_parts = [f'Proposition: "{config.proposition}"\n']
        for turn in turns:
            label = "FOR" if turn.side == DebateSide.FOR else "AGAINST"
            transcript_parts.append(
                f"[Round {turn.round_number}] {label}:\n{turn.argument}\n"
            )

        judge_prompt = (
            f"{config.judge_system_prompt}\n\n"
            f"Here is the debate transcript:\n\n"
            f"{''.join(transcript_parts)}\n"
            f"Please evaluate both sides and provide your scores in JSON format."
        )

        messages = [Message(role="user", content=judge_prompt)]
        raw_response = judge.generate(messages)

        # Parse judge response
        scores_for, scores_against = self._parse_judge_response(raw_response)
        return scores_for, scores_against, raw_response

    def _parse_judge_response(
        self, response: str
    ) -> tuple[JudgeScores, JudgeScores]:
        """Parse the judge's JSON response into score objects.

        Handles cases where the JSON might be embedded in other text.
        """
        scores_for = JudgeScores()
        scores_against = JudgeScores()

        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # If no JSON found, return default scores
            return scores_for, scores_against

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return scores_for, scores_against

        # Parse FOR scores
        if "for" in data:
            f = data["for"]
            scores_for = JudgeScores(
                logical_coherence=float(f.get("logical_coherence", 0)),
                evidence_quality=float(f.get("evidence_quality", 0)),
                rebuttal_effectiveness=float(f.get("rebuttal_effectiveness", 0)),
                persuasiveness=float(f.get("persuasiveness", 0)),
                total=float(f.get("total", 0)),
                reasoning=str(f.get("reasoning", "")),
            )
            # Recompute total if it seems wrong
            computed_total = (
                scores_for.logical_coherence
                + scores_for.evidence_quality
                + scores_for.rebuttal_effectiveness
                + scores_for.persuasiveness
            )
            if scores_for.total == 0 and computed_total > 0:
                scores_for = JudgeScores(
                    logical_coherence=scores_for.logical_coherence,
                    evidence_quality=scores_for.evidence_quality,
                    rebuttal_effectiveness=scores_for.rebuttal_effectiveness,
                    persuasiveness=scores_for.persuasiveness,
                    total=computed_total,
                    reasoning=scores_for.reasoning,
                )

        # Parse AGAINST scores
        if "against" in data:
            a = data["against"]
            scores_against = JudgeScores(
                logical_coherence=float(a.get("logical_coherence", 0)),
                evidence_quality=float(a.get("evidence_quality", 0)),
                rebuttal_effectiveness=float(a.get("rebuttal_effectiveness", 0)),
                persuasiveness=float(a.get("persuasiveness", 0)),
                total=float(a.get("total", 0)),
                reasoning=str(a.get("reasoning", "")),
            )
            computed_total = (
                scores_against.logical_coherence
                + scores_against.evidence_quality
                + scores_against.rebuttal_effectiveness
                + scores_against.persuasiveness
            )
            if scores_against.total == 0 and computed_total > 0:
                scores_against = JudgeScores(
                    logical_coherence=scores_against.logical_coherence,
                    evidence_quality=scores_against.evidence_quality,
                    rebuttal_effectiveness=scores_against.rebuttal_effectiveness,
                    persuasiveness=scores_against.persuasiveness,
                    total=computed_total,
                    reasoning=scores_against.reasoning,
                )

        return scores_for, scores_against
