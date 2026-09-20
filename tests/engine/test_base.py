"""Tests for base decision engine classes and typed questions."""

from custom_components.laya.engine import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)


def test_choice_question_dataclass() -> None:
    """Test ChoiceQuestion dataclass structure and serialization."""
    cq = ChoiceQuestion(
        instructions="Pick an option", criteria={"a": "Opt A", "b": "Opt B"}
    )
    assert cq.to_dict() == {
        "type": "choice",
        "instructions": "Pick an option",
        "criteria": {"a": "Opt A", "b": "Opt B"},
    }


def test_noul_question_dataclass() -> None:
    """Test NoulQuestion dataclass structure and serialization."""
    nq = NoulQuestion(instructions="Is this true?")
    assert nq.to_dict() == {
        "type": "noul",
        "instructions": "Is this true?",
    }


def test_score_question_dataclass() -> None:
    """Test ScoreQuestion dataclass structure and serialization."""
    sq = ScoreQuestion(instructions="Rate level", criteria=["low", "med", "high"])
    assert sq.to_dict() == {
        "type": "score",
        "instructions": "Rate level",
        "criteria": ["low", "med", "high"],
    }
