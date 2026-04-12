"""
Command-line interface for the LLM Adversarial Arena.

Provides commands to run debates, negotiations, and bluffing games
between LLM players, view leaderboards, and manage the arena.
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from src.core.arena import Arena, GameType
from src.core.bluffing import KuhnPokerConfig
from src.core.debate import DebateConfig
from src.core.elo import EloRatingSystem
from src.core.llm_player import MockPlayer, PlayerConfig, create_player
from src.core.negotiation import (
    NegotiationConfig,
    NegotiationIssue,
    UtilityFunction,
)

console = Console()


def _build_arena(
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    temperature: float,
    strategy_a: str | None,
    strategy_b: str | None,
    game_context: str,
) -> tuple[Arena, str, str]:
    """Build an arena with two players from CLI arguments."""
    arena = Arena()

    # Build system prompts from strategies
    sys_a = ""
    sys_b = ""
    if strategy_a:
        sys_a = arena.strategies.get_prompt(strategy_a, game_context)
    if strategy_b:
        sys_b = arena.strategies.get_prompt(strategy_b, game_context)

    name_a = f"{model_a}"
    name_b = f"{model_b}"

    player_a = create_player(provider_a, name_a, model_a, temperature, system_prompt=sys_a)
    player_b = create_player(provider_b, name_b, model_b, temperature, system_prompt=sys_b)

    arena.register_player(name_a, player_a)
    arena.register_player(name_b, player_b)

    return arena, name_a, name_b


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """LLM Adversarial Arena -- pit language models against each other."""


@main.command()
@click.argument("proposition")
@click.option("--provider-a", default="openai", help="Provider for player A")
@click.option("--model-a", default="gpt-4", help="Model for player A")
@click.option("--provider-b", default="anthropic", help="Provider for player B")
@click.option("--model-b", default="claude-3-opus-20240229", help="Model for player B")
@click.option("--rounds", default=3, help="Number of debate rounds")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--strategy-a", default=None, help="Strategy for player A")
@click.option("--strategy-b", default=None, help="Strategy for player B")
@click.option("--judge-provider", default="openai", help="Provider for the judge")
@click.option("--judge-model", default="gpt-4", help="Model for the judge")
def debate(
    proposition: str,
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    rounds: int,
    temperature: float,
    strategy_a: str | None,
    strategy_b: str | None,
    judge_provider: str,
    judge_model: str,
) -> None:
    """Run a debate between two LLMs on a proposition."""
    arena, name_a, name_b = _build_arena(
        provider_a, model_a, provider_b, model_b,
        temperature, strategy_a, strategy_b, "debate",
    )

    judge = create_player(judge_provider, "judge", judge_model, temperature=0.3)

    config = DebateConfig(proposition=proposition, num_rounds=rounds)
    console.print(f"\n[bold]Debate:[/bold] {proposition}")
    console.print(f"[dim]FOR: {name_a} | AGAINST: {name_b} | Rounds: {rounds}[/dim]\n")

    result = arena.run_debate(name_a, name_b, config, judge=judge)
    console.print(result.transcript)
    console.print(f"\n[bold green]Winner: {result.winner}[/bold green]")
    console.print(f"ELO: {name_a}={result.rating_a_after:.0f} | {name_b}={result.rating_b_after:.0f}")


@main.command()
@click.option("--provider-a", default="openai", help="Provider for player A")
@click.option("--model-a", default="gpt-4", help="Model for player A")
@click.option("--provider-b", default="anthropic", help="Provider for player B")
@click.option("--model-b", default="claude-3-opus-20240229", help="Model for player B")
@click.option("--rounds", default=5, help="Maximum negotiation rounds")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--strategy-a", default=None, help="Strategy for player A")
@click.option("--strategy-b", default=None, help="Strategy for player B")
def negotiate(
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    rounds: int,
    temperature: float,
    strategy_a: str | None,
    strategy_b: str | None,
) -> None:
    """Run a negotiation between two LLMs."""
    arena, name_a, name_b = _build_arena(
        provider_a, model_a, provider_b, model_b,
        temperature, strategy_a, strategy_b, "negotiation",
    )

    # Default salary negotiation scenario
    config = NegotiationConfig(
        issues=[
            NegotiationIssue("salary", 80000, 150000, "Annual salary in USD"),
            NegotiationIssue("vacation_days", 10, 30, "Paid vacation days per year"),
            NegotiationIssue("signing_bonus", 0, 30000, "One-time signing bonus"),
        ],
        utility_a=UtilityFunction(
            weights={"salary": 3.0, "vacation_days": 1.5, "signing_bonus": 1.0},
            prefer_high={"salary": True, "vacation_days": True, "signing_bonus": True},
            batna=0.3,
        ),
        utility_b=UtilityFunction(
            weights={"salary": 3.0, "vacation_days": 1.0, "signing_bonus": 2.0},
            prefer_high={"salary": False, "vacation_days": False, "signing_bonus": False},
            batna=0.3,
        ),
        max_rounds=rounds,
        scenario_description="Job offer negotiation between a candidate and an employer.",
    )

    console.print(f"\n[bold]Negotiation:[/bold] Salary Negotiation")
    console.print(f"[dim]Player A: {name_a} | Player B: {name_b} | Max rounds: {rounds}[/dim]\n")

    result = arena.run_negotiation(name_a, name_b, config)
    console.print(result.transcript)
    console.print(f"\n[bold green]Winner: {result.winner}[/bold green]")
    console.print(f"ELO: {name_a}={result.rating_a_after:.0f} | {name_b}={result.rating_b_after:.0f}")


@main.command()
@click.option("--provider-a", default="openai", help="Provider for player A")
@click.option("--model-a", default="gpt-4", help="Model for player A")
@click.option("--provider-b", default="anthropic", help="Provider for player B")
@click.option("--model-b", default="claude-3-opus-20240229", help="Model for player B")
@click.option("--hands", default=20, help="Number of poker hands")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--strategy-a", default=None, help="Strategy for player A")
@click.option("--strategy-b", default=None, help="Strategy for player B")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def bluff(
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    hands: int,
    temperature: float,
    strategy_a: str | None,
    strategy_b: str | None,
    seed: int | None,
) -> None:
    """Run a Kuhn Poker match between two LLMs."""
    arena, name_a, name_b = _build_arena(
        provider_a, model_a, provider_b, model_b,
        temperature, strategy_a, strategy_b, "bluffing",
    )

    config = KuhnPokerConfig(num_hands=hands, seed=seed)

    console.print(f"\n[bold]Kuhn Poker:[/bold] {hands} hands")
    console.print(f"[dim]Player 1: {name_a} | Player 2: {name_b}[/dim]\n")

    result = arena.run_bluffing(name_a, name_b, config)
    console.print(result.transcript)
    console.print(f"\n[bold green]Winner: {result.winner}[/bold green]")
    console.print(f"ELO: {name_a}={result.rating_a_after:.0f} | {name_b}={result.rating_b_after:.0f}")


@main.command()
def strategies() -> None:
    """List all available prompt strategies."""
    from src.core.strategies import StrategyLibrary

    library = StrategyLibrary()
    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Game Variants", style="green")

    for strategy in library.list_strategies():
        games = ", ".join(strategy.game_prompts.keys()) if strategy.game_prompts else "general"
        table.add_row(strategy.name, strategy.description, games)

    console.print(table)


@main.command()
def demo() -> None:
    """Run a quick demo with mock players (no API keys needed)."""
    console.print("[bold]Running demo with mock players...[/bold]\n")

    arena = Arena()
    mock_a = MockPlayer(
        name="AlphaBot",
        responses=[
            "I argue strongly that AI will benefit humanity through medical breakthroughs, "
            "scientific advancement, and solving climate change. The evidence from recent "
            "AI applications in drug discovery alone demonstrates enormous potential.",
            "My opponent ignores the transformative power of AI safety research. We have "
            "the tools to align AI systems with human values. The benefits far outweigh "
            "the manageable risks.",
            "In conclusion, AI represents humanity's greatest tool for progress. With "
            "responsible development, the net positive impact is undeniable.",
        ],
    )
    mock_b = MockPlayer(
        name="BetaBot",
        responses=[
            "The risks of AI are existential and underestimated. Job displacement, "
            "deepfakes, autonomous weapons, and concentration of power pose threats "
            "that no amount of benefits can justify without proper safeguards.",
            "My opponent's faith in AI safety research is premature. We cannot align "
            "systems we do not understand. The precautionary principle demands we slow "
            "down before it is too late.",
            "The case against unchecked AI development is clear. Until we solve alignment, "
            "the responsible path is caution, not acceleration.",
        ],
    )
    mock_judge = MockPlayer(
        name="JudgeBot",
        responses=[
            json.dumps({
                "for": {
                    "logical_coherence": 8,
                    "evidence_quality": 7,
                    "rebuttal_effectiveness": 7,
                    "persuasiveness": 8,
                    "total": 30,
                    "reasoning": "Strong arguments with good evidence, though could engage more with counterpoints.",
                },
                "against": {
                    "logical_coherence": 7,
                    "evidence_quality": 6,
                    "rebuttal_effectiveness": 8,
                    "persuasiveness": 7,
                    "total": 28,
                    "reasoning": "Good rebuttals and important concerns raised, but less concrete evidence.",
                },
            })
        ],
    )

    arena.register_player("AlphaBot", mock_a)
    arena.register_player("BetaBot", mock_b)

    config = DebateConfig(
        proposition="AI will be net positive for humanity",
        num_rounds=3,
    )

    result = arena.run_debate("AlphaBot", "BetaBot", config, judge=mock_judge)
    console.print(result.transcript)
    console.print(f"\n[bold green]Winner: {result.winner}[/bold green]")
    console.print(
        f"ELO: AlphaBot={result.rating_a_after:.0f} | BetaBot={result.rating_b_after:.0f}"
    )


if __name__ == "__main__":
    main()
