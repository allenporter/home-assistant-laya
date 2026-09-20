"""Tests for candidate ranking, discovery, and retrieval recall using synthetic home fixtures."""

from __future__ import annotations

from custom_components.laya.speculative import (
    ChoiceAnswer,
    NoulAnswer,
    SpeculativeFanOutStrategy,
)
from custom_components.laya.speculative.inmemory.engine import FakeDecisionEngine
from tests.common.fixture_loader import load_device_action_cases


async def test_farmhouse_context_loaded(farmhouse_context) -> None:
    """Verify that the farmhouse context has all 12 areas and 29 unique entities loaded."""
    assert len(farmhouse_context.area_registry.areas) == 12
    assert len(farmhouse_context.states) == 29
    assert farmhouse_context.home_name == "Family Farmhouse"


async def test_farmhouse_candidate_entity_ranking(
    farmhouse_context,
) -> None:
    """Test that candidate entity ranking surfaces the correct entity out of 28 entities."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine()

    # Query targeting kitchen light
    await strategy.async_decide(engine, "Turn on the Kitchen Light", farmhouse_context)
    assert len(engine.calls) == 1
    questions = engine.calls[0]["questions"]

    # Kitchen light should be in candidate entity choices
    assert "target_entity" in questions
    entity_crit = questions["target_entity"].criteria
    assert "light.kitchen_light" in entity_crit

    # And kitchen should be in target_area
    assert "target_area" in questions
    area_crit = questions["target_area"].criteria
    assert "kitchen" in area_crit


async def test_farmhouse_porch_light_ranking(
    farmhouse_context,
) -> None:
    """Test candidate ranking for wrap-around porch devices."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine()

    await strategy.async_decide(engine, "Turn off the Porch Light", farmhouse_context)
    questions = engine.calls[0]["questions"]

    assert "light.porch_light" in questions["target_entity"].criteria
    assert "wrap_around_porch" in questions["target_area"].criteria


async def test_farmhouse_thermostat_intent_discovery(
    farmhouse_context,
) -> None:
    """Test that supported climate actions are dynamically discovered when thermostat exists."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine()

    await strategy.async_decide(engine, "Turn off the thermostat", farmhouse_context)
    questions = engine.calls[0]["questions"]

    intent_crit = questions["intent"].criteria
    assert "HassTurnOff" in intent_crit
    assert "climate.thermostat" in questions["target_entity"].criteria


async def test_farmhouse_decision_routing(
    farmhouse_context,
) -> None:
    """Test end-to-end decision routing with FakeDecisionEngine on a farmhouse utterance."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.96),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.98),
            "target_entity": ChoiceAnswer(
                choice="light.kitchen_light", confidence=0.98
            ),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = await strategy.async_decide(
        engine, "Turn on the Kitchen Light", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.96
    assert decision.slots == {"entity_id": "light.kitchen_light"}


async def test_farmhouse_labeled_cases_loading() -> None:
    """Test that all supported labeled action cases from family-farmhouse-us are parsed."""
    cases = load_device_action_cases()
    assert len(cases) == 173

    sentences = {c.sentence: c for c in cases}
    assert "Please turn on the Kitchen Light" in sentences
    assert sentences["Please turn on the Kitchen Light"].expected_intent == "HassTurnOn"

    assert "Set the Kitchen Light to 50% brightness" in sentences
    assert (
        sentences["Set the Kitchen Light to 50% brightness"].action == "Set brightness"
    )
    assert (
        sentences["Set the Kitchen Light to 50% brightness"].expected_intent
        == "HassLightSet"
    )

    assert "Open the garage door" in sentences
    assert sentences["Open the garage door"].expected_intent == "HassOpenCover"
    assert "Stop moving the garage door" in sentences
    assert sentences["Stop moving the garage door"].expected_intent == "HassStopMoving"


async def test_farmhouse_hard_disambiguation_recall(farmhouse_context) -> None:
    """Test disambiguation across identical device names using area tokens."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine()

    # Family room speaker
    await strategy.async_decide(
        engine, "Pause the music in the family room", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "target_area" in questions
    assert "family_room" in questions["target_area"].criteria
    assert "target_entity" in questions
    assert "media_player.smart_speaker" in questions["target_entity"].criteria

    # Master bedroom speaker
    await strategy.async_decide(
        engine, "Turn up the master bedroom speaker", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "target_area" in questions
    assert "master_bedroom" in questions["target_area"].criteria

    # Porch speaker
    await strategy.async_decide(engine, "Resume music on the porch", farmhouse_context)
    questions = engine.calls[-1]["questions"]
    assert "target_area" in questions
    assert "wrap_around_porch" in questions["target_area"].criteria


async def test_farmhouse_valve_and_cover_candidate_recall(farmhouse_context) -> None:
    """Test candidate retrieval for valve and cover domains."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine()

    # Valve: Backyard smart sprinkler system
    await strategy.async_decide(
        engine, "Turn on the backyard sprinklers", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "valve.smart_sprinkler_system" in questions["target_entity"].criteria
    assert "HassTurnOn" in questions["intent"].criteria

    # Cover: Open barn garage door
    await strategy.async_decide(engine, "Open the barn garage door", farmhouse_context)
    questions = engine.calls[-1]["questions"]
    assert "cover.barn_garage_door" in questions["target_entity"].criteria
    assert "HassOpenCover" in questions["intent"].criteria

    # Cover: Close barn garage door
    await strategy.async_decide(engine, "Shut the barn garage door", farmhouse_context)
    questions = engine.calls[-1]["questions"]
    assert "cover.barn_garage_door" in questions["target_entity"].criteria
    assert "HassCloseCover" in questions["intent"].criteria

    # Cover: Stop moving
    await strategy.async_decide(
        engine, "Stop moving the garage door", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "cover.barn_garage_door" in questions["target_entity"].criteria
    assert "HassStopMoving" in questions["intent"].criteria


async def test_farmhouse_batch_candidate_recall(
    farmhouse_context,
) -> None:
    """Test candidate intent and entity retrieval recall across all labeled utterances."""
    strategy = SpeculativeFanOutStrategy()
    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.05),
        }
    )
    cases = load_device_action_cases()

    intent_hits = 0
    entity_hits = 0

    for c in cases:
        await strategy.async_decide(engine, c.sentence, farmhouse_context)
        questions = engine.calls[-1]["questions"]
        if c.expected_intent in questions["intent"].criteria:
            intent_hits += 1

        if "target_entity" in questions:
            matched_entity = any(
                c.device_name.lower() in desc.lower()
                for desc in questions["target_entity"].criteria.values()
            )
            if matched_entity:
                entity_hits += 1

    intent_recall = intent_hits / len(cases)
    entity_recall = entity_hits / len(cases)

    # Over 90% of utterances should correctly retrieve expected intent in candidate choices
    assert intent_recall > 0.90
    # Over 85% of utterances should correctly retrieve targeted device entity in candidate choices
    assert entity_recall > 0.85
