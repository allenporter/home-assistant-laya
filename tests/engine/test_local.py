"""Tests for LocalLayaEngine implementation."""

import asyncio
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from custom_components.laya.engine import ChoiceQuestion, LocalLayaEngine

TEST_QUESTIONS = {"q": ChoiceQuestion("test", {"opt1": "1"})}


async def test_local_engine_eager_load(
    create_local_engine: Callable[..., LocalLayaEngine],
    mock_load: MagicMock,
) -> None:
    """Test LocalLayaEngine eager load loads model into memory."""
    engine = create_local_engine(idle_timeout=0.05)
    assert not engine.loaded

    await engine.async_load()
    assert engine.loaded
    assert mock_load.call_count == 1


async def test_local_engine_idle_unload_after_eager_load(
    create_local_engine: Callable[..., LocalLayaEngine],
) -> None:
    """Test LocalLayaEngine automatically unloads after idle timeout post eager load."""
    engine = create_local_engine(idle_timeout=0.05)
    await engine.async_load()
    assert engine.loaded

    await asyncio.sleep(0.08)
    assert not engine.loaded


async def test_local_engine_keepalive_resets_idle_timer(
    create_local_engine: Callable[..., LocalLayaEngine],
    mock_load: MagicMock,
) -> None:
    """Test LocalLayaEngine subsequent predictions reset the idle timer and keep model alive."""
    engine = create_local_engine(idle_timeout=0.08)

    res = await engine.async_predict("hello", TEST_QUESTIONS)
    assert res.answers["q"].choice == "opt1"
    assert engine.loaded
    assert mock_load.call_count == 1

    # Utterance before idle timeout resets timer
    await asyncio.sleep(0.04)
    assert engine.loaded
    res2 = await engine.async_predict("hello again", TEST_QUESTIONS)
    assert res2.answers["q"].choice == "opt1"
    assert engine.loaded
    assert mock_load.call_count == 1


async def test_local_engine_idle_timeout_unloads(
    create_local_engine: Callable[..., LocalLayaEngine],
) -> None:
    """Test LocalLayaEngine unloads after prediction when idle timeout expires."""
    engine = create_local_engine(idle_timeout=0.05)

    res = await engine.async_predict("hello", TEST_QUESTIONS)
    assert res.answers["q"].choice == "opt1"
    assert engine.loaded

    await asyncio.sleep(0.08)
    assert not engine.loaded


async def test_local_engine_predict_reloads_after_unload(
    create_local_engine: Callable[..., LocalLayaEngine],
    mock_load: MagicMock,
) -> None:
    """Test LocalLayaEngine reloads model transparently on prediction after idle unload."""
    engine = create_local_engine(idle_timeout=0.05)

    await engine.async_predict("first", TEST_QUESTIONS)
    assert mock_load.call_count == 1

    await asyncio.sleep(0.08)
    assert not engine.loaded

    res = await engine.async_predict("second", TEST_QUESTIONS)
    assert res.answers["q"].choice == "opt1"
    assert engine.loaded
    assert mock_load.call_count == 2


async def test_local_engine_zero_idle_timeout_unloads_immediately(
    create_local_engine: Callable[..., LocalLayaEngine],
    mock_load: MagicMock,
) -> None:
    """Test LocalLayaEngine with idle_timeout=0 unloads immediately after prediction."""
    engine = create_local_engine(idle_timeout=0.0)
    res = await engine.async_predict("test", TEST_QUESTIONS)
    assert res.answers["q"].choice == "opt1"
    assert mock_load.call_count == 1
    assert not engine.loaded


@pytest.mark.slow
async def test_live_local_engine_turn_on_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live inference test verifying real Laya weights predict turn on light."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = LocalLayaEngine(device="cpu", idle_timeout=10.0)
    questions = {
        "intent": ChoiceQuestion(
            "Determine the primary action",
            {
                "HassTurnOn": "Turn on or activate device or light",
                "HassTurnOff": "Turn off or stop device or light",
            },
        ),
        "target_domain": ChoiceQuestion(
            "What device domain is targeted?",
            {
                "light": "Lighting devices and lamps",
                "switch": "Switches and power outlets",
            },
        ),
    }
    try:
        res = await engine.async_predict("Turn on the light", questions)
        assert res.answers["intent"].choice == "HassTurnOn"
        assert res.answers["target_domain"].choice == "light"
    finally:
        await engine.async_unload()
