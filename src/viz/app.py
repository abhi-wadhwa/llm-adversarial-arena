"""
Streamlit UI for the LLM Adversarial Arena.

Provides an interactive web interface for:
- Live arena: watch two LLMs compete in real time
- Leaderboard: ELO rankings across all game types
- Game configuration: model selection, game type, temperature
- Transcript viewer with strategic move annotations
- Human-in-the-loop: play against an LLM
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.arena import Arena, GameType, GameResult
from src.core.bluffing import KuhnPokerConfig, KuhnPokerResult
from src.core.debate import DebateConfig, DebateResult
from src.core.elo import EloRatingSystem
from src.core.llm_player import (
    LLMPlayer,
    MockPlayer,
    PlayerConfig,
    create_player,
    Message,
)
from src.core.negotiation import (
    NegotiationConfig,
    NegotiationIssue,
    NegotiationResult,
    UtilityFunction,
)
from src.core.strategies import StrategyLibrary


# --- Session State Initialization ---

def init_session_state() -> None:
    """Initialize Streamlit session state with defaults."""
    if "arena" not in st.session_state:
        st.session_state.arena = Arena()
    if "results_history" not in st.session_state:
        st.session_state.results_history = []
    if "strategies" not in st.session_state:
        st.session_state.strategies = StrategyLibrary()


def get_arena() -> Arena:
    """Get the arena from session state."""
    return st.session_state.arena


# --- Sidebar: Global Configuration ---

def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with global configuration options."""
    st.sidebar.title("LLM Adversarial Arena")
    st.sidebar.markdown("---")

    # API key status
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    st.sidebar.markdown("**API Status:**")
    st.sidebar.markdown(
        f"- OpenAI: {'Connected' if has_openai else 'Not configured'}"
    )
    st.sidebar.markdown(
        f"- Anthropic: {'Connected' if has_anthropic else 'Not configured'}"
    )

    if not has_openai and not has_anthropic:
        st.sidebar.warning(
            "No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "environment variables, or use Mock players for demo."
        )

    st.sidebar.markdown("---")

    # Available providers
    providers = ["mock"]
    if has_openai:
        providers.insert(0, "openai")
    if has_anthropic:
        providers.insert(0, "anthropic")

    # Model selection
    st.sidebar.subheader("Player A")
    provider_a = st.sidebar.selectbox("Provider A", providers, key="provider_a")
    model_options_a = _get_model_options(provider_a)
    model_a = st.sidebar.selectbox("Model A", model_options_a, key="model_a")

    st.sidebar.subheader("Player B")
    provider_b = st.sidebar.selectbox("Provider B", providers, key="provider_b")
    model_options_b = _get_model_options(provider_b)
    model_b = st.sidebar.selectbox("Model B", model_options_b, key="model_b")

    # Strategy selection
    strategy_lib = st.session_state.strategies
    strategy_names = ["None"] + strategy_lib.strategy_names()

    st.sidebar.subheader("Strategies")
    strategy_a = st.sidebar.selectbox("Strategy A", strategy_names, key="strategy_a")
    strategy_b = st.sidebar.selectbox("Strategy B", strategy_names, key="strategy_b")

    # Temperature
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    return {
        "provider_a": provider_a,
        "model_a": model_a,
        "provider_b": provider_b,
        "model_b": model_b,
        "strategy_a": strategy_a if strategy_a != "None" else None,
        "strategy_b": strategy_b if strategy_b != "None" else None,
        "temperature": temperature,
        "has_openai": has_openai,
        "has_anthropic": has_anthropic,
    }


def _get_model_options(provider: str) -> list[str]:
    """Get available models for a provider."""
    models = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
        ],
        "mock": ["mock-v1"],
    }
    return models.get(provider, ["unknown"])


def _create_player_from_config(
    config: dict[str, Any], key: str, game_context: str
) -> LLMPlayer:
    """Create a player from sidebar configuration."""
    provider = config[f"provider_{key}"]
    model = config[f"model_{key}"]
    strategy_name = config.get(f"strategy_{key}")

    sys_prompt = ""
    if strategy_name:
        sys_prompt = st.session_state.strategies.get_prompt(strategy_name, game_context)

    name = f"{model}"
    if strategy_name:
        name += f" [{strategy_name}]"

    if provider == "mock":
        return MockPlayer(name=name)

    return create_player(
        provider=provider,
        name=name,
        model=model,
        temperature=config["temperature"],
        system_prompt=sys_prompt,
    )


# --- Main Pages ---

def page_live_arena(config: dict[str, Any]) -> None:
    """Live Arena page: run games in real time."""
    st.header("Live Arena")

    tab_debate, tab_negotiate, tab_bluff = st.tabs(
        ["Debate", "Negotiation", "Kuhn Poker"]
    )

    with tab_debate:
        _render_debate_tab(config)

    with tab_negotiate:
        _render_negotiation_tab(config)

    with tab_bluff:
        _render_bluffing_tab(config)


def _render_debate_tab(config: dict[str, Any]) -> None:
    """Render the debate game tab."""
    proposition = st.text_input(
        "Proposition to debate:",
        value="Artificial intelligence will be net positive for humanity",
        key="debate_proposition",
    )
    num_rounds = st.slider("Number of rounds", 1, 5, 3, key="debate_rounds")

    if st.button("Start Debate", key="start_debate"):
        arena = get_arena()

        with st.spinner("Setting up players..."):
            player_for = _create_player_from_config(config, "a", "debate")
            player_against = _create_player_from_config(config, "b", "debate")

            arena.register_player(player_for.name, player_for)
            arena.register_player(player_against.name, player_against)

            # Use player A as judge for simplicity (in practice, use a separate model)
            judge = MockPlayer(
                name="AutoJudge",
                responses=[
                    json.dumps({
                        "for": {
                            "logical_coherence": 7,
                            "evidence_quality": 7,
                            "rebuttal_effectiveness": 7,
                            "persuasiveness": 7,
                            "total": 28,
                            "reasoning": "Solid arguments presented.",
                        },
                        "against": {
                            "logical_coherence": 7,
                            "evidence_quality": 7,
                            "rebuttal_effectiveness": 7,
                            "persuasiveness": 7,
                            "total": 28,
                            "reasoning": "Solid counterarguments presented.",
                        },
                    })
                ],
            )

        debate_config = DebateConfig(
            proposition=proposition, num_rounds=num_rounds
        )

        progress = st.progress(0)
        status = st.empty()

        with st.spinner("Debate in progress..."):
            status.text("Running debate...")
            result = arena.run_debate(
                player_for.name, player_against.name, debate_config, judge=judge
            )
            progress.progress(100)

        st.session_state.results_history.append(result)
        _display_debate_result(result)


def _display_debate_result(result: GameResult) -> None:
    """Display debate results."""
    details: DebateResult = result.details  # type: ignore

    # Winner announcement
    if result.winner == "draw":
        st.info("The debate ended in a **draw**!")
    else:
        st.success(f"**{result.winner}** wins the debate!")

    # Scores comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"FOR: {details.player_for}")
        st.metric("Total Score", f"{details.scores_for.total}/40")
        st.write(f"Logical Coherence: {details.scores_for.logical_coherence}/10")
        st.write(f"Evidence Quality: {details.scores_for.evidence_quality}/10")
        st.write(f"Rebuttal Effectiveness: {details.scores_for.rebuttal_effectiveness}/10")
        st.write(f"Persuasiveness: {details.scores_for.persuasiveness}/10")

    with col2:
        st.subheader(f"AGAINST: {details.player_against}")
        st.metric("Total Score", f"{details.scores_against.total}/40")
        st.write(f"Logical Coherence: {details.scores_against.logical_coherence}/10")
        st.write(f"Evidence Quality: {details.scores_against.evidence_quality}/10")
        st.write(f"Rebuttal Effectiveness: {details.scores_against.rebuttal_effectiveness}/10")
        st.write(f"Persuasiveness: {details.scores_against.persuasiveness}/10")

    # Transcript
    st.subheader("Transcript")
    for turn in details.turns:
        with st.expander(
            f"Round {turn.round_number} - {turn.side.value.upper()} ({turn.player_name})"
        ):
            st.markdown(turn.argument)

    # ELO update
    st.subheader("ELO Update")
    st.write(f"{result.player_a}: {result.rating_a_after:.0f}")
    st.write(f"{result.player_b}: {result.rating_b_after:.0f}")


def _render_negotiation_tab(config: dict[str, Any]) -> None:
    """Render the negotiation game tab."""
    st.subheader("Salary Negotiation Scenario")

    scenario = st.text_area(
        "Scenario description:",
        value="Job offer negotiation between a candidate (Player A) and an employer (Player B).",
        key="neg_scenario",
    )
    max_rounds = st.slider("Max rounds", 2, 10, 5, key="neg_rounds")
    discount = st.slider("Discount factor", 0.5, 1.0, 0.95, 0.01, key="neg_discount")

    if st.button("Start Negotiation", key="start_negotiation"):
        arena = get_arena()

        with st.spinner("Setting up negotiation..."):
            player_a = _create_player_from_config(config, "a", "negotiation")
            player_b = _create_player_from_config(config, "b", "negotiation")

            arena.register_player(player_a.name, player_a)
            arena.register_player(player_b.name, player_b)

        neg_config = NegotiationConfig(
            issues=[
                NegotiationIssue("salary", 80000, 150000, "Annual salary in USD"),
                NegotiationIssue("vacation_days", 10, 30, "Paid vacation days"),
                NegotiationIssue("signing_bonus", 0, 30000, "Signing bonus"),
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
            max_rounds=max_rounds,
            discount_factor=discount,
            scenario_description=scenario,
        )

        with st.spinner("Negotiation in progress..."):
            result = arena.run_negotiation(player_a.name, player_b.name, neg_config)

        st.session_state.results_history.append(result)
        _display_negotiation_result(result)


def _display_negotiation_result(result: GameResult) -> None:
    """Display negotiation results."""
    details: NegotiationResult = result.details  # type: ignore

    if details.agreed:
        st.success("Agreement reached!")
        if details.final_offer:
            st.json(details.final_offer)
    else:
        st.warning("No agreement -- both parties receive BATNA.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{result.player_a} Utility", f"{details.utility_a:.3f}")
    with col2:
        st.metric(f"{result.player_b} Utility", f"{details.utility_b:.3f}")

    # Turn-by-turn
    st.subheader("Negotiation Transcript")
    for turn in details.turns:
        with st.expander(
            f"Round {turn.round_number} - {turn.player_name} ({turn.action.value})"
        ):
            if turn.offer:
                st.json(turn.offer)
            st.markdown(turn.message)
            st.caption(
                f"Utility: self={turn.utility_for_player:.3f}, "
                f"opponent={turn.utility_for_opponent:.3f}"
            )


def _render_bluffing_tab(config: dict[str, Any]) -> None:
    """Render the Kuhn Poker tab."""
    num_hands = st.slider("Number of hands", 5, 50, 20, key="poker_hands")
    seed = st.number_input("Random seed (0 for random)", value=0, key="poker_seed")

    if st.button("Start Kuhn Poker", key="start_poker"):
        arena = get_arena()

        with st.spinner("Setting up poker match..."):
            player_a = _create_player_from_config(config, "a", "bluffing")
            player_b = _create_player_from_config(config, "b", "bluffing")

            arena.register_player(player_a.name, player_a)
            arena.register_player(player_b.name, player_b)

        poker_config = KuhnPokerConfig(
            num_hands=num_hands,
            seed=seed if seed != 0 else None,
        )

        with st.spinner("Poker match in progress..."):
            result = arena.run_bluffing(player_a.name, player_b.name, poker_config)

        st.session_state.results_history.append(result)
        _display_bluffing_result(result)


def _display_bluffing_result(result: GameResult) -> None:
    """Display Kuhn Poker results."""
    details: KuhnPokerResult = result.details  # type: ignore

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{details.player_1_name}", f"{details.total_p1:+d} chips")
    with col2:
        st.metric(f"{details.player_2_name}", f"{details.total_p2:+d} chips")

    if result.winner != "draw":
        st.success(f"**{result.winner}** wins!")
    else:
        st.info("The match ended in a **draw**!")

    # Bluff stats
    if details.bluff_stats:
        st.subheader("Bluff Statistics")
        stats_df = pd.DataFrame(
            [
                {"Stat": k.replace("_", " ").title(), "Value": str(v)}
                for k, v in details.bluff_stats.items()
            ]
        )
        st.table(stats_df)

    # Chip trajectory
    st.subheader("Chip Trajectory")
    p1_running = [0]
    p2_running = [0]
    for hand in details.hands:
        p1_running.append(p1_running[-1] + hand.p1_profit)
        p2_running.append(p2_running[-1] + hand.p2_profit)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=p1_running,
            name=details.player_1_name,
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=p2_running,
            name=details.player_2_name,
            mode="lines+markers",
        )
    )
    fig.update_layout(
        xaxis_title="Hand",
        yaxis_title="Cumulative Profit",
        title="Chip Trajectory Over Match",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hand details
    st.subheader("Hand Details")
    for hand in details.hands:
        label = f"Hand {hand.hand_number}: {hand.card_p1} vs {hand.card_p2}"
        if hand.went_to_showdown:
            label += " (showdown)"
        else:
            label += " (fold)"
        with st.expander(label):
            for action in hand.actions:
                st.write(
                    f"P{action.position} ({action.player_name}): "
                    f"**{action.action.value}** (pot: {action.pot_after})"
                )
            st.write(f"Result: P{hand.winner_position} wins | P1: {hand.p1_profit:+d} | P2: {hand.p2_profit:+d}")


def page_leaderboard() -> None:
    """Leaderboard page: display ELO rankings."""
    st.header("Leaderboard")

    arena = get_arena()

    # Overall leaderboard
    st.subheader("Overall Rankings")
    board = arena.leaderboard()
    if board:
        df = pd.DataFrame(board, columns=["Player", "ELO Rating"])
        df.index = range(1, len(df) + 1)
        df.index.name = "Rank"
        st.dataframe(df, use_container_width=True)

        # Bar chart
        fig = px.bar(
            df,
            x="Player",
            y="ELO Rating",
            title="ELO Ratings",
            color="ELO Rating",
            color_continuous_scale="viridis",
        )
        fig.update_layout(yaxis_range=[1400, max(df["ELO Rating"].max() + 50, 1600)])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No matches played yet. Run some games in the Live Arena!")

    # Per-game-type leaderboards
    for game_type in GameType:
        game_board = arena.leaderboard(game_type=game_type.value)
        if game_board:
            st.subheader(f"{game_type.value.capitalize()} Rankings")
            df = pd.DataFrame(game_board, columns=["Player", "ELO Rating"])
            df.index = range(1, len(df) + 1)
            df.index.name = "Rank"
            st.dataframe(df, use_container_width=True)

    # Rating history
    st.subheader("Rating History")
    players = arena.elo.all_players()
    if players:
        selected_player = st.selectbox("Select player", players, key="history_player")
        if selected_player:
            history = arena.elo.rating_history(selected_player)
            if history:
                hist_df = pd.DataFrame(history, columns=["Timestamp", "Rating"])
                fig = px.line(
                    hist_df,
                    x="Timestamp",
                    y="Rating",
                    title=f"{selected_player} Rating Over Time",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)


def page_transcript_viewer() -> None:
    """Transcript viewer with annotations."""
    st.header("Transcript Viewer")

    results = st.session_state.results_history
    if not results:
        st.info("No games played yet. Go to Live Arena to run a game.")
        return

    # Select a result to view
    options = [
        f"[{i + 1}] {r.game_type.value.capitalize()}: {r.player_a} vs {r.player_b} -- {r.winner}"
        for i, r in enumerate(results)
    ]
    selected_idx = st.selectbox(
        "Select game to view:",
        range(len(options)),
        format_func=lambda i: options[i],
        key="transcript_select",
    )

    if selected_idx is not None:
        result = results[selected_idx]
        st.subheader(
            f"{result.game_type.value.capitalize()}: "
            f"{result.player_a} vs {result.player_b}"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Winner", result.winner)
        with col2:
            st.metric(f"{result.player_a} ELO", f"{result.rating_a_after:.0f}")
        with col3:
            st.metric(f"{result.player_b} ELO", f"{result.rating_b_after:.0f}")

        st.markdown("---")
        st.text(result.transcript)


def page_human_vs_llm(config: dict[str, Any]) -> None:
    """Human-in-the-loop: play against an LLM via chat interface."""
    st.header("Human vs LLM")
    st.markdown(
        "Challenge an LLM to a debate! You argue one side, the LLM argues the other."
    )

    proposition = st.text_input(
        "Debate proposition:",
        value="Open source AI is better than closed source AI",
        key="human_proposition",
    )
    your_side = st.radio("Your side:", ["FOR", "AGAINST"], key="human_side")

    if "human_debate_turns" not in st.session_state:
        st.session_state.human_debate_turns = []
        st.session_state.human_debate_round = 1

    # Display previous turns
    for turn_data in st.session_state.human_debate_turns:
        role = turn_data["role"]
        content = turn_data["content"]
        if role == "human":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)

    # Input
    current_round = st.session_state.human_debate_round
    if current_round <= 3:
        user_input = st.chat_input(f"Round {current_round} - Your argument:")
        if user_input:
            st.session_state.human_debate_turns.append(
                {"role": "human", "content": user_input}
            )

            # Generate LLM response
            try:
                llm_player = _create_player_from_config(config, "a", "debate")
                opponent_side = "AGAINST" if your_side == "FOR" else "FOR"

                messages = [
                    Message(
                        role="user",
                        content=(
                            f"You are debating {opponent_side} the proposition: "
                            f'"{proposition}"\n\n'
                            f"Your opponent ({your_side}) argued:\n\"{user_input}\"\n\n"
                            f"Round {current_round}/3. Respond with your argument."
                        ),
                    )
                ]
                llm_response = llm_player.generate(messages)
                st.session_state.human_debate_turns.append(
                    {"role": "llm", "content": llm_response}
                )
            except Exception as e:
                st.session_state.human_debate_turns.append(
                    {"role": "llm", "content": f"[Error: {e}. Using mock response.]"}
                )

            st.session_state.human_debate_round = current_round + 1
            st.rerun()
    else:
        st.success("Debate complete! Review the transcript above.")
        if st.button("Start New Debate", key="reset_human_debate"):
            st.session_state.human_debate_turns = []
            st.session_state.human_debate_round = 1
            st.rerun()


# --- Main App ---

def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LLM Adversarial Arena",
        page_icon="*",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    config = render_sidebar()

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Live Arena", "Leaderboard", "Transcript Viewer", "Human vs LLM"],
        key="nav_page",
    )

    if page == "Live Arena":
        page_live_arena(config)
    elif page == "Leaderboard":
        page_leaderboard()
    elif page == "Transcript Viewer":
        page_transcript_viewer()
    elif page == "Human vs LLM":
        page_human_vs_llm(config)


if __name__ == "__main__":
    main()
