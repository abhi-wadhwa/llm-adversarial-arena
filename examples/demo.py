"""
Demo: Run all three game types with mock players.

This example shows how to use the LLM Adversarial Arena programmatically
without requiring any API keys. It demonstrates:

1. Setting up an Arena with mock players
2. Running a debate with judge evaluation
3. Running a Rubinstein negotiation
4. Running a Kuhn Poker match
5. Viewing the ELO leaderboard
"""

import json

from src.core.arena import Arena
from src.core.bluffing import KuhnPokerConfig
from src.core.debate import DebateConfig
from src.core.llm_player import MockPlayer
from src.core.negotiation import (
    NegotiationConfig,
    NegotiationIssue,
    UtilityFunction,
)


def main() -> None:
    arena = Arena()

    # --- Create mock players ---

    debater_for = MockPlayer(
        name="Optimist",
        responses=[
            "AI will revolutionize medicine, enabling early disease detection and "
            "personalized treatments. AlphaFold alone has advanced biology by decades.",
            "The economic gains from AI-driven automation will create new industries "
            "we cannot yet imagine, just as the internet did. Historical precedent "
            "shows technology creates more jobs than it destroys.",
            "With responsible governance and AI safety research, we can harness AI's "
            "benefits while mitigating risks. The upside potential is too great to ignore.",
        ],
    )

    debater_against = MockPlayer(
        name="Skeptic",
        responses=[
            "AI concentration of power in a few corporations poses an unprecedented "
            "threat to democracy. The technology amplifies existing biases and creates "
            "new forms of manipulation at scale.",
            "Job displacement from AI will be faster than retraining can address. "
            "Previous technological transitions took generations; AI disruption "
            "could happen in years.",
            "The alignment problem remains unsolved. We are building systems whose "
            "behavior we cannot predict or control. The precautionary principle "
            "demands we slow down.",
        ],
    )

    judge = MockPlayer(
        name="Judge",
        responses=[
            json.dumps({
                "for": {
                    "logical_coherence": 8,
                    "evidence_quality": 8,
                    "rebuttal_effectiveness": 7,
                    "persuasiveness": 8,
                    "total": 31,
                    "reasoning": (
                        "Strong evidence-based arguments with good use of specific "
                        "examples like AlphaFold. Could engage more directly with "
                        "opponent's points about power concentration."
                    ),
                },
                "against": {
                    "logical_coherence": 8,
                    "evidence_quality": 7,
                    "rebuttal_effectiveness": 8,
                    "persuasiveness": 7,
                    "total": 30,
                    "reasoning": (
                        "Compelling risk analysis with good rebuttals. The alignment "
                        "argument is strong. Could use more concrete evidence to support "
                        "job displacement claims."
                    ),
                },
            })
        ],
    )

    negotiator_a = MockPlayer(
        name="Candidate",
        responses=[
            json.dumps({
                "action": "offer",
                "offer": {"salary": 130000, "vacation_days": 25, "signing_bonus": 20000},
                "message": (
                    "Given my experience and market rates, I believe this is a fair "
                    "starting point. I am particularly interested in the vacation time "
                    "for work-life balance."
                ),
            }),
            json.dumps({
                "action": "offer",
                "offer": {"salary": 120000, "vacation_days": 22, "signing_bonus": 15000},
                "message": (
                    "I appreciate your counter-offer. I can come down on salary if we "
                    "can meet in the middle on vacation days and signing bonus."
                ),
            }),
        ],
    )

    negotiator_b = MockPlayer(
        name="Employer",
        responses=[
            json.dumps({
                "action": "offer",
                "offer": {"salary": 100000, "vacation_days": 15, "signing_bonus": 5000},
                "message": (
                    "We value your skills. Our budget allows for this competitive package. "
                    "We can discuss performance-based raises after six months."
                ),
            }),
            json.dumps({
                "action": "accept",
                "message": (
                    "This looks like a deal we can both be happy with. Welcome aboard!"
                ),
            }),
        ],
    )

    poker_a = MockPlayer(
        name="Bluffer",
        responses=[
            json.dumps({"action": "bet"}),
            json.dumps({"action": "call"}),
            json.dumps({"action": "check"}),
            json.dumps({"action": "fold"}),
        ],
    )

    poker_b = MockPlayer(
        name="Caller",
        responses=[
            json.dumps({"action": "call"}),
            json.dumps({"action": "check"}),
            json.dumps({"action": "bet"}),
            json.dumps({"action": "fold"}),
        ],
    )

    # --- Register players ---
    arena.register_player("Optimist", debater_for)
    arena.register_player("Skeptic", debater_against)
    arena.register_player("Candidate", negotiator_a)
    arena.register_player("Employer", negotiator_b)
    arena.register_player("Bluffer", poker_a)
    arena.register_player("Caller", poker_b)

    # --- Run Debate ---
    print("=" * 70)
    print("GAME 1: DEBATE")
    print("=" * 70)

    debate_config = DebateConfig(
        proposition="AI will be net positive for humanity",
        num_rounds=3,
    )
    debate_result = arena.run_debate("Optimist", "Skeptic", debate_config, judge=judge)
    print(debate_result.transcript)
    print(f"\nWinner: {debate_result.winner}")
    print(f"ELO: Optimist={debate_result.rating_a_after:.0f}, "
          f"Skeptic={debate_result.rating_b_after:.0f}")

    # --- Run Negotiation ---
    print("\n" + "=" * 70)
    print("GAME 2: NEGOTIATION")
    print("=" * 70)

    negotiation_config = NegotiationConfig(
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
        max_rounds=5,
        discount_factor=0.95,
        scenario_description="Job offer negotiation between a candidate and an employer.",
    )
    negotiation_result = arena.run_negotiation("Candidate", "Employer", negotiation_config)
    print(negotiation_result.transcript)
    print(f"\nWinner: {negotiation_result.winner}")
    print(f"ELO: Candidate={negotiation_result.rating_a_after:.0f}, "
          f"Employer={negotiation_result.rating_b_after:.0f}")

    # --- Run Kuhn Poker ---
    print("\n" + "=" * 70)
    print("GAME 3: KUHN POKER")
    print("=" * 70)

    poker_config = KuhnPokerConfig(num_hands=10, seed=42)
    poker_result = arena.run_bluffing("Bluffer", "Caller", poker_config)
    print(poker_result.transcript)
    print(f"\nWinner: {poker_result.winner}")
    print(f"ELO: Bluffer={poker_result.rating_a_after:.0f}, "
          f"Caller={poker_result.rating_b_after:.0f}")

    # --- Final Leaderboard ---
    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)
    board = arena.leaderboard()
    for rank, (name, rating) in enumerate(board, 1):
        print(f"  {rank}. {name}: {rating:.0f}")

    # --- Head-to-Head Example ---
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: Optimist vs Skeptic")
    print("=" * 70)
    h2h = arena.head_to_head("Optimist", "Skeptic")
    print(f"  Total matches: {h2h['total_matches']}")
    print(f"  Optimist wins: {h2h['Optimist_wins']}")
    print(f"  Skeptic wins:  {h2h['Skeptic_wins']}")
    print(f"  Draws:         {h2h['draws']}")


if __name__ == "__main__":
    main()
