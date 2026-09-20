"""Tests for decision engine implementations."""

import asyncio
from unittest.mock import MagicMock, patch

from custom_components.laya.engine import (
    ChoiceAnswer,
    ChoiceQuestion,
    FakeDecisionEngine,
    LocalLayaEngine,
    NoulAnswer,
    NoulQuestion,
    PredictionResult,
    ScoreAnswer,
    ScoreQuestion,
)


def test_typed_question_dataclasses() -> None:
    """Test Question dataclass structure and serialization."""
    cq = ChoiceQuestion(
        instructions="Pick an option", criteria={"a": "Opt A", "b": "Opt B"}
    )
    assert cq.to_dict() == {
        "type": "choice",
        "instructions": "Pick an option",
        "criteria": {"a": "Opt A", "b": "Opt B"},
    }

    nq = NoulQuestion(instructions="Is this true?")
    assert nq.to_dict() == {
        "type": "noul",
        "instructions": "Is this true?",
    }

    sq = ScoreQuestion(instructions="Rate level", criteria=["low", "med", "high"])
    assert sq.to_dict() == {
        "type": "score",
        "instructions": "Rate level",
        "criteria": ["low", "med", "high"],
    }


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


async def test_local_engine_eager_load_and_idle_unload() -> None:
    """Test LocalLayaEngine eager load and automatic idle unload."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "model": "laya-test",
        "answers": {"q": {"type": "choice", "choice": "opt1", "confidence": 0.9}},
    }

    with patch("laya.load", return_value=mock_model) as mock_load:
        engine = LocalLayaEngine(device="cpu", idle_timeout=0.05)
        try:
            assert not engine.loaded

            # Eager load
            await engine.async_load()
            assert engine.loaded
            assert mock_load.call_count == 1

            # Wait for idle timeout to trigger unload
            await asyncio.sleep(0.08)
            assert not engine.loaded
        finally:
            await engine.async_unload()


async def test_local_engine_predict_keepalive() -> None:
    """Test LocalLayaEngine predict keeps model alive and reloads if unloaded."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "model": "laya-test",
        "answers": {"q": {"type": "choice", "choice": "opt1", "confidence": 0.9}},
    }

    with patch("laya.load", return_value=mock_model) as mock_load:
        engine = LocalLayaEngine(device="cpu", idle_timeout=0.08)
        try:
            # First predict triggers load
            res = await engine.async_predict(
                "hello", {"q": ChoiceQuestion("test", {"opt1": "1"})}
            )
            assert res.answers["q"].choice == "opt1"
            assert engine.loaded
            assert mock_load.call_count == 1

            # Utterance before idle timeout resets timer
            await asyncio.sleep(0.04)
            assert engine.loaded
            res2 = await engine.async_predict(
                "hello again", {"q": ChoiceQuestion("test", {"opt1": "1"})}
            )
            assert res2.answers["q"].choice == "opt1"
            assert mock_load.call_count == 1  # No reload needed

            # Idle timeout expires and unloads model
            await asyncio.sleep(0.12)
            assert not engine.loaded

            # Subsequent prediction transparently reloads model
            res3 = await engine.async_predict(
                "waking up", {"q": ChoiceQuestion("test", {"opt1": "1"})}
            )
            assert res3.answers["q"].choice == "opt1"
            assert engine.loaded
            assert mock_load.call_count == 2
        finally:
            await engine.async_unload()


async def test_local_engine_zero_idle_timeout_unloads_immediately() -> None:
    """Test LocalLayaEngine with idle_timeout=0 unloads immediately after prediction."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "model": "laya-test",
        "answers": {"q": {"type": "choice", "choice": "opt1", "confidence": 0.9}},
    }

    with patch("laya.load", return_value=mock_model) as mock_load:
        engine = LocalLayaEngine(device="cpu", idle_timeout=0.0)
        try:
            res = await engine.async_predict(
                "test", {"q": ChoiceQuestion("test", {"opt1": "1"})}
            )
            assert res.answers["q"].choice == "opt1"
            assert mock_load.call_count == 1
            # When idle_timeout is 0, model unloads immediately once prediction finishes
            assert not engine.loaded
        finally:
            await engine.async_unload()
