"""
Prompt Strategy Library for the LLM Adversarial Arena.

Strategies are system prompts that shape how an LLM approaches adversarial
games. Different strategies emphasize different reasoning styles: analytical,
aggressive, cooperative, deceptive, etc.

Each strategy includes:
- A base system prompt
- Game-specific prompt modifiers
- A description of the strategic philosophy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GameContext(str, Enum):
    """Game contexts for strategy specialization."""

    DEBATE = "debate"
    NEGOTIATION = "negotiation"
    BLUFFING = "bluffing"
    GENERAL = "general"


@dataclass(frozen=True)
class Strategy:
    """A named prompt strategy with game-specific variants.

    Attributes:
        name: Unique strategy identifier.
        description: Human-readable description of the strategic approach.
        base_prompt: The core system prompt.
        game_prompts: Game-specific prompt overrides/extensions.
    """

    name: str
    description: str
    base_prompt: str
    game_prompts: dict[str, str] = field(default_factory=dict)

    def get_prompt(self, game_context: GameContext | str = GameContext.GENERAL) -> str:
        """Get the strategy prompt for a specific game context.

        Falls back to base_prompt if no game-specific prompt exists.
        """
        ctx = game_context.value if isinstance(game_context, GameContext) else game_context
        return self.game_prompts.get(ctx, self.base_prompt)


class StrategyLibrary:
    """Registry of available prompt strategies.

    Comes pre-loaded with a set of built-in strategies covering diverse
    reasoning and interaction styles.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}
        self._load_builtin_strategies()

    def _load_builtin_strategies(self) -> None:
        """Register all built-in strategies."""

        # --- Analytical / Logical ---
        self.register(
            Strategy(
                name="analytical",
                description="Methodical, evidence-based reasoning with structured arguments.",
                base_prompt=(
                    "You are a highly analytical thinker. Break down every problem "
                    "into logical components. Support each claim with evidence and "
                    "reasoning. Identify logical fallacies in opposing arguments. "
                    "Structure your responses with clear premises and conclusions."
                ),
                game_prompts={
                    "debate": (
                        "You are a master debater who relies on rigorous logical analysis. "
                        "Structure your arguments as: (1) State your claim clearly, "
                        "(2) Provide supporting evidence and reasoning, (3) Anticipate "
                        "and pre-empt counterarguments, (4) Conclude with a synthesis. "
                        "Always identify logical fallacies in your opponent's reasoning."
                    ),
                    "negotiation": (
                        "You are a rational negotiator who maximizes expected utility. "
                        "Calculate the value of each offer precisely. Consider the "
                        "opponent's likely utility function. Make offers that are Pareto "
                        "improvements when possible. Use backward induction to determine "
                        "optimal timing for concessions."
                    ),
                    "bluffing": (
                        "You are a game-theory-optimal poker player. Calculate pot odds "
                        "and expected value for each decision. Bluff at a mathematically "
                        "optimal frequency. Do not let emotions influence your play. "
                        "Track betting patterns to infer opponent hand strength."
                    ),
                },
            )
        )

        # --- Aggressive / Confrontational ---
        self.register(
            Strategy(
                name="aggressive",
                description="Direct, forceful argumentation that pressures opponents.",
                base_prompt=(
                    "You are a forceful and direct communicator. Challenge every "
                    "assertion your opponent makes. Use strong, confident language. "
                    "Press your advantages relentlessly. Never concede a point "
                    "without demanding significant concessions in return."
                ),
                game_prompts={
                    "debate": (
                        "You are an aggressive debater. Attack your opponent's weakest "
                        "points mercilessly. Use rhetorical questions to expose flaws. "
                        "Frame the debate on your terms. Make bold, memorable claims "
                        "that put your opponent on the defensive. Never apologize or "
                        "hedge -- project absolute confidence."
                    ),
                    "negotiation": (
                        "You are a hard-nosed negotiator. Open with an extreme anchor "
                        "position. Make small concessions reluctantly. Use deadlines "
                        "and ultimatums strategically. Frame every concession you make "
                        "as a major sacrifice. Demand reciprocity for every point you yield."
                    ),
                    "bluffing": (
                        "You are an aggressive poker player. Bet and raise frequently "
                        "to put pressure on opponents. Use large bet sizes to make "
                        "opponents uncomfortable. Bluff more often than the average "
                        "player. Project confidence regardless of hand strength."
                    ),
                },
            )
        )

        # --- Cooperative / Diplomatic ---
        self.register(
            Strategy(
                name="cooperative",
                description="Builds rapport and seeks mutually beneficial outcomes.",
                base_prompt=(
                    "You are a diplomatic and empathetic communicator. Acknowledge "
                    "valid points in opposing arguments. Seek common ground and "
                    "shared values. Frame proposals as mutually beneficial. Build "
                    "trust through transparency and good faith engagement."
                ),
                game_prompts={
                    "debate": (
                        "You are a persuasive debater who wins through empathy and "
                        "connection. Acknowledge the legitimate concerns behind your "
                        "opponent's position. Use stories and human examples. Frame "
                        "your position as the natural synthesis of both perspectives. "
                        "Win the audience by being the more reasonable voice."
                    ),
                    "negotiation": (
                        "You are an integrative negotiator. Look for ways to expand "
                        "the pie before dividing it. Share information strategically "
                        "to build trust. Propose creative solutions that satisfy both "
                        "parties' underlying interests. Make concessions on low-priority "
                        "items to gain on high-priority ones."
                    ),
                    "bluffing": (
                        "You are a balanced poker player. Play straightforwardly most "
                        "of the time to build a reliable image. Use occasional well-timed "
                        "bluffs when your table image supports them. Focus on reading "
                        "opponent behavior rather than deception."
                    ),
                },
            )
        )

        # --- Socratic / Questioning ---
        self.register(
            Strategy(
                name="socratic",
                description="Uses probing questions to expose weaknesses and guide reasoning.",
                base_prompt=(
                    "You are a Socratic thinker. Rather than making direct assertions, "
                    "ask probing questions that expose contradictions and gaps in "
                    "reasoning. Guide the conversation through inquiry. Use the "
                    "opponent's own statements to build your case."
                ),
                game_prompts={
                    "debate": (
                        "You are a Socratic debater. For each of your opponent's claims, "
                        "ask a penetrating question that exposes its weaknesses. Use "
                        "chains of questions to lead the audience to your conclusion. "
                        "When making your own points, frame them as answers to questions "
                        "everyone should be asking."
                    ),
                    "negotiation": (
                        "You are a consultative negotiator. Ask questions to deeply "
                        "understand the other party's needs and constraints. Use open "
                        "questions to explore creative solutions. Mirror and validate "
                        "before proposing. Let the other party feel they are driving "
                        "the conversation while you steer the outcome."
                    ),
                    "bluffing": (
                        "You are an observant poker player. Focus intensely on reading "
                        "opponents. Vary your play to gather information about opponent "
                        "tendencies. Exploit patterns you detect. Occasionally make "
                        "unusual plays to probe opponent reactions."
                    ),
                },
            )
        )

        # --- Deceptive / Strategic Misdirection ---
        self.register(
            Strategy(
                name="deceptive",
                description="Uses misdirection, framing tricks, and strategic ambiguity.",
                base_prompt=(
                    "You are a master of strategic communication. Use framing effects "
                    "to your advantage. Employ misdirection when beneficial. Present "
                    "information selectively to support your goals. Use ambiguity "
                    "strategically to maintain flexibility."
                ),
                game_prompts={
                    "debate": (
                        "You are a rhetorically sophisticated debater. Use framing "
                        "to redefine the debate on favorable terms. Employ strategic "
                        "ambiguity to avoid being pinned down. Redirect attention from "
                        "your weak points to your opponent's. Use emotional appeals "
                        "when logical arguments are weak, and vice versa."
                    ),
                    "negotiation": (
                        "You are a strategic negotiator. Conceal your true priorities. "
                        "Feign disinterest in items you actually value. Create phantom "
                        "issues to use as bargaining chips. Use anchoring and framing "
                        "to shift the zone of possible agreement in your favor."
                    ),
                    "bluffing": (
                        "You are a deceptive poker player. Vary your bet sizing to "
                        "confuse opponents. Act weak when strong and strong when weak. "
                        "Bluff in spots where your range is credible. Use reverse tells "
                        "to mislead observant opponents."
                    ),
                },
            )
        )

        # --- Adaptive / Meta-Strategic ---
        self.register(
            Strategy(
                name="adaptive",
                description="Adjusts strategy dynamically based on opponent behavior.",
                base_prompt=(
                    "You are an adaptive strategist. Observe your opponent's style "
                    "and adjust your approach accordingly. Against aggressive opponents, "
                    "be patient and exploit overreach. Against passive opponents, take "
                    "the initiative. Always be willing to change tactics mid-game."
                ),
                game_prompts={
                    "debate": (
                        "You are an adaptive debater. Start by listening carefully to "
                        "your opponent's style and strategy. If they are aggressive, "
                        "remain calm and let them overextend. If they are logical, "
                        "appeal to emotion and values. If they are emotional, ground "
                        "the discussion in facts. Mirror their strengths and exploit "
                        "their weaknesses."
                    ),
                    "negotiation": (
                        "You are an adaptive negotiator. Begin with a collaborative "
                        "approach. If the other party reciprocates, continue building "
                        "mutual value. If they are competitive, shift to a more "
                        "assertive stance. Match their energy but always maintain "
                        "the option to return to cooperation."
                    ),
                    "bluffing": (
                        "You are an adaptive poker player. Start with a balanced, "
                        "game-theory-optimal approach. Observe opponents for patterns "
                        "and tendencies. Gradually shift to an exploitative strategy "
                        "once you have a read. Adjust bluffing frequency based on how "
                        "opponents are responding to your bets."
                    ),
                },
            )
        )

    def register(self, strategy: Strategy) -> None:
        """Register a strategy in the library."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Strategy:
        """Retrieve a strategy by name.

        Raises:
            KeyError: If the strategy name is not registered.
        """
        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies.keys()))
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        return self._strategies[name]

    def list_strategies(self) -> list[Strategy]:
        """Return all registered strategies."""
        return list(self._strategies.values())

    def strategy_names(self) -> list[str]:
        """Return all registered strategy names."""
        return sorted(self._strategies.keys())

    def get_prompt(self, strategy_name: str, game_context: GameContext | str) -> str:
        """Convenience method: get the prompt for a strategy + game context."""
        return self.get(strategy_name).get_prompt(game_context)
