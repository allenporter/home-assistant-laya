"""Tests for FakeDecisionEngine."""

from custom_components.laya.engine import (
    ChoiceAnswer,
    ChoiceQuestion,
    FakeDecisionEngine,
    NoulAnswer,
    NoulQuestion,
    PredictionResult,
    ScoreAnswer,
    ScoreQuestion,
)


async def test_fake_decision_engine_default_generation() -> None:
    """Test FakeDecisionEngine fallback synthesis from typed questions."""
    engine = FakeDecisionEngine()

    questions = {
        "intent": ChoiceQuestion(
            instructions="Choose action",
            criteria={"HassTurnOn": "Turn on", "HassTurnOff": "Turn off"},
        ),
        "is_compound": NoulQuestion(instructions="Multiple actions?"),
        "target_temperature": ScoreQuestion(
            instructions="Target temp",
            criteria=["low", "med", "high"],
        ),
    }

    result = await engine.async_predict({"user_query": "Turn on the light"}, questions)

    assert len(engine.calls) == 1
    assert engine.calls[0]["state"] == {"user_query": "Turn on the light"}
    assert "intent" in result.answers
    assert isinstance(result.answers["intent"], ChoiceAnswer)
    assert result.answers["intent"].choice == "HassTurnOn"

    assert isinstance(result.answers["is_compound"], NoulAnswer)
    assert result.answers["is_compound"].noul == 0.1

    assert isinstance(result.answers["target_temperature"], ScoreAnswer)
    assert result.answers["target_temperature"].score == 1.0


async def test_fake_decision_engine_queued_results() -> None:
    """Test FakeDecisionEngine queue_result functionality."""
    engine = FakeDecisionEngine()

    custom_res = PredictionResult(
        answers={"intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.99)},
        model="custom-mock",
    )
    engine.queue_result(custom_res)

    result = await engine.async_predict("test", {})
    assert result.model == "custom-mock"
    intent_answer = result.answers["intent"]
    assert isinstance(intent_answer, ChoiceAnswer)
    assert intent_answer.choice == "HassTurnOff"
