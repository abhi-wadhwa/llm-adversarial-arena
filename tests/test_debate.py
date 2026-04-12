"""
Tests for the Debate Engine.

Uses MockPlayer to test debate protocol without API calls.
Verifies:
- Turn structure (alternating FOR/AGAINST)
- Correct number of rounds
- Judge scoring and winner determination
- Transcript generation
- Edge cases (draw, parsing failures)
"""

import json

import pytest

from src.core.debate import (
    DebateConfig,
    DebateEngine,
    DebateResult,
    DebateSide,
    DebateTurn,
    JudgeScores,
)
from src.core.llm_player import MockPlayer


@pytest.fixture
def engine() -> DebateEngine:
    return DebateEngine()


@pytest.fixture
def config() -> DebateConfig:
    return DebateConfig(
        proposition="AI will be net positive for humanity",
        num_rounds=2,
    )


@pytest.fixture
def player_for() -> MockPlayer:
    return MockPlayer(
        name="ForBot",
        responses=[
            "AI will revolutionize healthcare and save millions of lives.",
            "The benefits clearly outweigh the risks when we invest in safety.",
        ],
    )


@pytest.fixture
def player_against() -> MockPlayer:
    return MockPlayer(
        name="AgainstBot",
        responses=[
            "AI poses existential risks that we are not prepared to handle.",
            "Without proper alignment, AI could lead to catastrophic outcomes.",
        ],
    )


def _make_judge(for_total: int = 30, against_total: int = 25) -> MockPlayer:
    """Create a mock judge with specific scores."""
    scores = json.dumps(
        {
            "for": {
                "logical_coherence": 8,
                "evidence_quality": 7,
                "rebuttal_effectiveness": 8,
                "persuasiveness": 7,
                "total": for_total,
                "reasoning": "Strong logical arguments.",
            },
            "against": {
                "logical_coherence": 6,
                "evidence_quality": 6,
                "rebuttal_effectiveness": 7,
                "persuasiveness": 6,
                "total": against_total,
                "reasoning": "Good points but less evidence.",
            },
        }
    )
    return MockPlayer(name="JudgeBot", responses=[scores])


class TestDebateProtocol:
    """Test the debate protocol structure."""

    def test_correct_number_of_turns(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        # 2 rounds * 2 players = 4 turns
        assert len(result.turns) == 4

    def test_alternating_sides(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        sides = [t.side for t in result.turns]
        expected = [
            DebateSide.FOR,
            DebateSide.AGAINST,
            DebateSide.FOR,
            DebateSide.AGAINST,
        ]
        assert sides == expected

    def test_correct_player_names(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        for_turns = [t for t in result.turns if t.side == DebateSide.FOR]
        against_turns = [t for t in result.turns if t.side == DebateSide.AGAINST]
        assert all(t.player_name == "ForBot" for t in for_turns)
        assert all(t.player_name == "AgainstBot" for t in against_turns)

    def test_round_numbers_sequential(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        round_nums = [t.round_number for t in result.turns]
        assert round_nums == [1, 1, 2, 2]

    def test_single_round(self, engine, player_for, player_against):
        config = DebateConfig(
            proposition="Test proposition", num_rounds=1
        )
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        assert len(result.turns) == 2

    def test_arguments_are_non_empty(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        for turn in result.turns:
            assert len(turn.argument) > 0


class TestJudging:
    """Test judge scoring and winner determination."""

    def test_for_wins(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=35, against_total=25)
        result = engine.run(player_for, player_against, judge, config)
        assert result.winner == "for"

    def test_against_wins(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=20, against_total=30)
        result = engine.run(player_for, player_against, judge, config)
        assert result.winner == "against"

    def test_draw(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=28, against_total=28)
        result = engine.run(player_for, player_against, judge, config)
        assert result.winner == "draw"

    def test_scores_parsed_correctly(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=30, against_total=25)
        result = engine.run(player_for, player_against, judge, config)
        assert result.scores_for.logical_coherence == 8
        assert result.scores_for.evidence_quality == 7
        assert result.scores_against.logical_coherence == 6

    def test_judge_raw_response_stored(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        assert len(result.judge_raw_response) > 0

    def test_malformed_judge_response(self, engine, config, player_for, player_against):
        """If the judge returns non-JSON, default scores should be used."""
        judge = MockPlayer(
            name="BadJudge",
            responses=["This is not JSON at all."],
        )
        result = engine.run(player_for, player_against, judge, config)
        # Should not crash; should use default scores
        assert result.scores_for.total == 0
        assert result.scores_against.total == 0
        assert result.winner == "draw"


class TestEloScoreConversion:
    """Test conversion from debate result to ELO score."""

    def test_for_win_gives_1(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=35, against_total=25)
        result = engine.run(player_for, player_against, judge, config)
        assert result.elo_score_for == 1.0

    def test_against_win_gives_0(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=20, against_total=30)
        result = engine.run(player_for, player_against, judge, config)
        assert result.elo_score_for == 0.0

    def test_draw_gives_half(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=28, against_total=28)
        result = engine.run(player_for, player_against, judge, config)
        assert result.elo_score_for == 0.5


class TestTranscript:
    """Test transcript generation."""

    def test_transcript_contains_proposition(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        transcript = result.transcript()
        assert config.proposition in transcript

    def test_transcript_contains_arguments(self, engine, config, player_for, player_against):
        judge = _make_judge()
        result = engine.run(player_for, player_against, judge, config)
        transcript = result.transcript()
        assert "revolutionize healthcare" in transcript
        assert "existential risks" in transcript

    def test_transcript_contains_winner(self, engine, config, player_for, player_against):
        judge = _make_judge(for_total=35, against_total=25)
        result = engine.run(player_for, player_against, judge, config)
        transcript = result.transcript()
        assert "FOR" in transcript


class TestPlayerReset:
    """Test that players are reset between games."""

    def test_players_reset(self, engine, config, player_for, player_against):
        judge = _make_judge()
        engine.run(player_for, player_against, judge, config)
        # After the game, message count should reflect game activity
        assert player_for._message_count > 0
        assert player_against._message_count > 0

        # Running again should work (players are reset internally)
        player_for._call_index = 0
        player_against._call_index = 0
        result2 = engine.run(player_for, player_against, _make_judge(), config)
        assert len(result2.turns) == 4
