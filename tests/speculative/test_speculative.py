"""Unit tests for speculative decision strategy, models, and candidate discovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.laya.speculative.engine import PredictionResult
from custom_components.laya.speculative.inmemory.engine import FakeDecisionEngine
from custom_components.laya.speculative.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreQuestion,
)
from custom_components.laya.speculative.strategy.base import StrategyContext
from custom_components.laya.speculative.strategy.discovery import (
    ONOFF_DOMAINS,
    can_fulfill_intent,
    get_allowed_domains_for_intents,
    get_handler_slot_info,
    lexical_score,
    rank_areas,
    rank_entities,
    tokenize,
)
from custom_components.laya.speculative.strategy.speculative import (
    SpeculativeFanOutStrategy,
)


@pytest.fixture(name="strategy")
def strategy_fixture() -> SpeculativeFanOutStrategy:
    """Fixture providing a default SpeculativeFanOutStrategy."""
    return SpeculativeFanOutStrategy(confidence_threshold=0.7)


@pytest.fixture(name="empty_context")
def empty_context_fixture() -> MagicMock:
    """Fixture providing an empty StrategyContext mock."""
    context = MagicMock(spec=StrategyContext)
    context.home_name = "My Home"
    context.area_registry.async_list_areas.return_value = []
    context.states = []
    return context


def test_question_models_dataclasses() -> None:
    """Test Question primitives dataclass initialization."""
    cq = ChoiceQuestion(
        instructions="Pick an option", criteria={"a": "Opt A", "b": "Opt B"}
    )
    assert cq.instructions == "Pick an option"
    assert cq.criteria == {"a": "Opt A", "b": "Opt B"}

    nq = NoulQuestion(instructions="Is this true?")
    assert nq.instructions == "Is this true?"

    sq = ScoreQuestion(instructions="Rate level", criteria=["low", "med", "high"])
    assert sq.instructions == "Rate level"
    assert sq.criteria == ["low", "med", "high"]


def test_fake_decision_engine() -> None:
    """Test FakeDecisionEngine fallback synthesis and queue_result."""
    engine = FakeDecisionEngine()
    custom_res = PredictionResult(
        answers={"intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.99)},
        model="custom-mock",
    )
    engine.queue_result(custom_res)

    import asyncio

    res = asyncio.run(engine.async_predict("test", {}))
    assert res.model == "custom-mock"
    assert res.answers["intent"].choice == "HassTurnOff"


def test_tokenize_and_lexical_score() -> None:
    """Test text tokenization and lexical scoring."""
    tokens = tokenize("Turn on the Kitchen Light!")
    assert "turn" in tokens
    assert "kitchen" in tokens
    assert "light" in tokens

    score_match = lexical_score(
        tokens, "Kitchen Ceiling Light", "turn on the kitchen light"
    )
    assert score_match > 0.0

    score_mismatch = lexical_score(tokens, "Basement Fan", "turn on the kitchen light")
    assert score_mismatch == 0.0


def test_intent_handler_slot_info_and_can_fulfill() -> None:
    """Test extracting slot info from Home Assistant intent handlers."""
    handler = MagicMock()
    handler.required_slots = {"name": None}
    handler.optional_slots = {"area": None}
    handler.slot_schema = None

    supported, required = get_handler_slot_info(handler)
    assert "name" in required
    assert "name" in supported
    assert "area" in supported
    assert can_fulfill_intent(handler)

    unsupported_handler = MagicMock()
    unsupported_handler.required_slots = {"unknown_slot": None}
    unsupported_handler.optional_slots = {}
    unsupported_handler.slot_schema = None
    assert not can_fulfill_intent(unsupported_handler)


def test_discovery_and_ranking_with_area_boosting() -> None:
    """Test intent discovery, area ranking, and entity ranking with area boosting."""
    context = MagicMock(spec=StrategyContext)
    context.hass = MagicMock()

    # Mock area registry
    area_kitchen = MagicMock()
    area_kitchen.name = "Kitchen"
    area_bedroom = MagicMock()
    area_bedroom.name = "Bedroom"
    context.area_registry.async_list_areas.return_value = [area_kitchen, area_bedroom]

    # Mock entity states
    light_kitchen = MagicMock()
    light_kitchen.entity_id = "light.kitchen_lights"
    light_kitchen.domain = "light"
    light_kitchen.attributes = {"friendly_name": "Kitchen Lights"}

    light_bedroom = MagicMock()
    light_bedroom.entity_id = "light.bedroom_lights"
    light_bedroom.domain = "light"
    light_bedroom.attributes = {"friendly_name": "Bedroom Lights"}

    context.states = [light_kitchen, light_bedroom]

    # Mock entity registry entries linking entities to areas
    entry_k = MagicMock()
    entry_k.area_id = "kitchen_id"
    context.area_registry.async_get_area.side_effect = (
        lambda aid: area_kitchen if aid == "kitchen_id" else None
    )
    context.entity_registry.async_get.side_effect = (
        lambda eid: entry_k if eid == "light.kitchen_lights" else None
    )

    # Area ranking
    areas = rank_areas(context, "Turn off kitchen light", max_options=5)
    assert "Kitchen" in areas
    assert "none" in areas

    # Entity ranking with area boosting
    entities = rank_entities(
        context, "Turn off kitchen light", active_areas={"Kitchen"}, max_options=5
    )
    assert "light.kitchen_lights" in entities
    # Kitchen light gets boosted over bedroom light
    crit_keys = list(entities.keys())
    assert crit_keys[0] == "light.kitchen_lights"


async def test_speculative_fan_out_strategy_decision(
    strategy: SpeculativeFanOutStrategy,
    empty_context: MagicMock,
) -> None:
    """Test end-to-end decision evaluation via SpeculativeFanOutStrategy with FakeDecisionEngine."""
    fake_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_entity": ChoiceAnswer(choice="light.kitchen", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = await strategy.async_decide(
        fake_engine, "Turn on the kitchen light to 50%", empty_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.95
    assert decision.slots.get("entity_id") == "light.kitchen"
    assert decision.slots.get("brightness") == 50


async def test_speculative_fan_out_compound_and_low_confidence(
    empty_context: MagicMock,
) -> None:
    """Test compound command escalation and low confidence handling."""
    strategy = SpeculativeFanOutStrategy(
        confidence_threshold=0.8, compound_threshold=0.5
    )

    # Compound escalation
    compound_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.8),
        }
    )

    compound_dec = await strategy.async_decide(
        compound_engine, "Turn on light and play music", empty_context
    )
    assert compound_dec.should_escalate
    assert compound_dec.is_compound
    assert compound_dec.escalation_reason == "Compound command detected"

    # Low confidence escalation
    low_conf_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.6),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )
    low_conf_dec = await strategy.async_decide(
        low_conf_engine, "Turn on light", empty_context
    )
    assert low_conf_dec.should_escalate
    assert low_conf_dec.escalation_reason == "Unhandled intent or low confidence"


async def test_intent_driven_domain_filtering(empty_context: MagicMock) -> None:
    """Test that candidate entity domains are derived directly from candidate intents."""
    # Dummy handlers mapping
    mock_media_handler = MagicMock()
    mock_media_handler.platforms = {"media_player"}

    mock_light_handler = MagicMock()
    mock_light_handler.platforms = {"light"}

    handlers = {
        "HassMediaPause": mock_media_handler,
        "HassLightSet": mock_light_handler,
    }

    # 1. Specialized intent narrows to its platform
    media_domains = get_allowed_domains_for_intents(["HassMediaPause"], handlers)
    assert media_domains == {"media_player"}

    light_domains = get_allowed_domains_for_intents(["HassLightSet"], handlers)
    assert light_domains == {"light"}

    # 2. Generic intent falls back to ONOFF_DOMAINS
    turn_on_domains = get_allowed_domains_for_intents(["HassTurnOn"], handlers)
    assert turn_on_domains == set(ONOFF_DOMAINS)
    assert "light" in turn_on_domains
    assert "cover" in turn_on_domains
    assert "valve" in turn_on_domains
    assert "sensor" not in turn_on_domains
    assert "binary_sensor" not in turn_on_domains

    # 3. Informational intent is ignored
    info_domains = get_allowed_domains_for_intents(["HassGetState"], handlers)
    assert info_domains == set(ONOFF_DOMAINS)

    # 4. Entity ranking respects allowed_domains
    empty_context.entity_registry.async_get.return_value = None
    mock_light = MagicMock(
        domain="light", entity_id="light.kitchen", name="Kitchen Light"
    )
    mock_speaker = MagicMock(
        domain="media_player",
        entity_id="media_player.kitchen_speaker",
        name="Kitchen Speaker",
    )
    empty_context.states = [mock_light, mock_speaker]

    ranked_media = rank_entities(
        empty_context,
        "pause kitchen",
        active_areas=set(),
        allowed_domains={"media_player"},
    )
    assert "media_player.kitchen_speaker" in ranked_media
    assert "light.kitchen" not in ranked_media

    ranked_light = rank_entities(
        empty_context, "turn on kitchen", active_areas=set(), allowed_domains={"light"}
    )
    assert "light.kitchen" in ranked_light
    assert "media_player.kitchen_speaker" not in ranked_light
