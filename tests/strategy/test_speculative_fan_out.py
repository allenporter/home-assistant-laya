"""Tests for speculative fan-out decision strategy."""

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, entity_registry as er, intent

from custom_components.laya.const import MAX_OPTIONS_PER_QUESTION
from custom_components.laya.engine import (
    ChoiceAnswer,
    FakeDecisionEngine,
    NoulAnswer,
)
from custom_components.laya.strategy import SpeculativeFanOutStrategy, StrategyContext
from tests.common.fixture_loader import DummyIntentHandler


async def test_speculative_fan_out_area_turn_on(hass: HomeAssistant) -> None:
    """Test routing when targeting an area."""
    area_reg = ar.async_get(hass)
    area_reg.async_get_or_create("living_room")

    entity_reg = er.async_get(hass)
    context = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[
            State(
                "light.living_room_light", "on", {"friendly_name": "Living Room Light"}
            )
        ],
    )

    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "target_type": ChoiceAnswer(choice="area", confidence=0.9),
            "target_domain": ChoiceAnswer(choice="light", confidence=0.95),
            "target_area": ChoiceAnswer(choice="living_room", confidence=0.92),
            "light_action": ChoiceAnswer(choice="turn_on", confidence=0.98),
            "is_compound": NoulAnswer(noul=0.05, confidence=0.95),
        }
    )

    strategy = SpeculativeFanOutStrategy()
    decision = await strategy.async_decide(
        engine, "Turn on the living room lights", context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.slots == {"area": "living_room", "domain": "light"}
    assert "target_area" in decision.active_keys
    assert "light_action" in decision.active_keys


async def test_speculative_fan_out_entity_turn_off(hass: HomeAssistant) -> None:
    """Test routing when targeting a specific device entity."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    context = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[State("light.desk_lamp", "on", {"friendly_name": "Desk Lamp"})],
    )

    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.91),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_domain": ChoiceAnswer(choice="light", confidence=0.95),
            "target_entity": ChoiceAnswer(choice="light.desk_lamp", confidence=0.96),
            "is_compound": NoulAnswer(noul=0.01, confidence=0.99),
        }
    )

    strategy = SpeculativeFanOutStrategy()
    decision = await strategy.async_decide(engine, "Turn off desk lamp", context)

    assert not decision.should_escalate
    assert decision.intent_name == "HassTurnOff"
    assert decision.slots == {"entity_id": "light.desk_lamp"}
    assert "target_entity" in decision.active_keys


async def test_speculative_fan_out_prunes_unsupported_area_slot(
    hass: HomeAssistant,
) -> None:
    """Test that target_area is omitted when candidate intents do not support area targeting."""
    intent.async_register(
        hass,
        DummyIntentHandler(
            "CustomDeviceCalibrate",
            description="Performs an entity-specific calibrate on a device",
            platforms={"light"},
            supported_slots={"name"},
        ),
    )

    area_reg = ar.async_get(hass)
    area_reg.async_get_or_create("living_room")

    entity_reg = er.async_get(hass)
    context = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[State("light.desk_lamp", "on", {"friendly_name": "Desk Lamp"})],
    )

    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="CustomDeviceCalibrate", confidence=0.95),
            "target_entity": ChoiceAnswer(choice="light.desk_lamp", confidence=0.96),
            "is_compound": NoulAnswer(noul=0.01, confidence=0.99),
        }
    )

    strategy = SpeculativeFanOutStrategy()
    decision = await strategy.async_decide(engine, "Calibrate the desk lamp", context)

    assert len(engine.calls) == 1
    questions = engine.calls[0]["questions"]

    # Since CustomDeviceCalibrate only supports 'name', 'target_area' and 'target_domain' must be pruned!
    assert "target_area" not in questions
    assert "target_domain" not in questions
    # And target_type is omitted since only 1 target scope ('entity') is possible
    assert "target_type" not in questions
    assert "target_entity" in questions

    assert not decision.should_escalate
    assert decision.intent_name == "CustomDeviceCalibrate"
    assert decision.slots == {"entity_id": "light.desk_lamp"}
    assert "target_entity" in decision.active_keys


async def test_speculative_fan_out_filters_unfulfillable_required_slots(
    hass: HomeAssistant,
) -> None:
    """Test that intents requiring slots we cannot fulfill are filtered out."""
    intent.async_register(
        hass,
        DummyIntentHandler(
            "HassClimateSetTemperature",
            description="Sets thermostat target temperature in degrees",
            platforms={"climate"},
            supported_slots={"name", "area", "temperature"},
            required_slots={"temperature"},
        ),
    )

    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    context = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[
            State(
                "climate.thermostat",
                "heat",
                {"friendly_name": "Thermostat"},
            )
        ],
    )

    engine = FakeDecisionEngine()
    strategy = SpeculativeFanOutStrategy()

    await strategy.async_decide(engine, "Set thermostat to 72 degrees", context)

    assert len(engine.calls) == 1
    questions = engine.calls[0]["questions"]
    # The intent requiring temperature must NOT be in the intent criteria
    assert "HassClimateSetTemperature" not in questions["intent"].criteria


async def test_speculative_fan_out_brightness_numeric_extraction(
    hass: HomeAssistant,
) -> None:
    """Test routing and deterministic brightness extraction for lights."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    context = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[
            State("light.kitchen_ceiling", "on", {"friendly_name": "Kitchen Light"})
        ],
    )

    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.96),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_domain": ChoiceAnswer(choice="light", confidence=0.95),
            "target_entity": ChoiceAnswer(
                choice="light.kitchen_ceiling", confidence=0.95
            ),
            "light_action": ChoiceAnswer(choice="dim", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01, confidence=0.99),
        }
    )

    strategy = SpeculativeFanOutStrategy()
    decision = await strategy.async_decide(
        engine, "Set kitchen ceiling light to 50%", context
    )

    assert not decision.should_escalate
    assert decision.intent_name == "HassTurnOn"
    assert decision.slots["entity_id"] == "light.kitchen_ceiling"
    assert decision.slots["brightness"] == 50


async def test_speculative_fan_out_compound_escalation(hass: HomeAssistant) -> None:
    """Test compound utterance triggers escalation."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    context = StrategyContext(
        hass=hass, area_registry=area_reg, entity_registry=entity_reg
    )

    engine = FakeDecisionEngine(
        default_answers={
            "is_compound": NoulAnswer(noul=0.88, confidence=0.88),
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.8),
        }
    )

    strategy = SpeculativeFanOutStrategy(compound_threshold=0.65)
    decision = await strategy.async_decide(
        engine, "Turn on the lights and lock the door", context
    )

    assert decision.is_compound
    assert decision.should_escalate
    assert decision.intent_name is None
    assert decision.escalation_reason == "Compound command detected"


async def test_speculative_fan_out_dynamic_domain_discovery(
    hass: HomeAssistant,
) -> None:
    """Test dynamic domain detection discovers all entity domains in state."""
    states = [
        State("fan.bedroom_fan", "off", {"friendly_name": "Bedroom Fan"}),
        State("vacuum.robot_cleaner", "docked", {"friendly_name": "Robot Cleaner"}),
        State("cover.garage_door", "closed", {"friendly_name": "Garage Door"}),
        State("light.ceiling", "off", {"friendly_name": "Ceiling Light"}),
    ]

    context = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=states,
    )

    engine = FakeDecisionEngine()
    strategy = SpeculativeFanOutStrategy()
    await strategy.async_decide(engine, "Clean the kitchen", context)

    assert len(engine.calls) == 1
    questions = engine.calls[0]["questions"]
    target_domain_q = questions["target_domain"]
    criteria = target_domain_q.criteria
    assert "fan" in criteria
    assert "vacuum" in criteria
    assert "cover" in criteria
    assert "light" in criteria


async def test_speculative_fan_out_cardinality_limit_enforcement(
    hass: HomeAssistant,
) -> None:
    """Test cardinality limit enforcement ensures options do not exceed budget."""
    area_reg = ar.async_get(hass)
    for i in range(35):
        area_reg.async_get_or_create(f"room_{i}")

    states = [
        State(f"light.light_{i}", "off", {"friendly_name": f"Light {i}"})
        for i in range(35)
    ]

    context = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=er.async_get(hass),
        states=states,
    )

    engine = FakeDecisionEngine()
    strategy = SpeculativeFanOutStrategy()
    await strategy.async_decide(engine, "Turn on lights", context)

    assert len(engine.calls) == 1
    questions = engine.calls[0]["questions"]

    area_q = questions["target_area"]
    assert len(area_q.criteria) <= MAX_OPTIONS_PER_QUESTION

    ent_q = questions["target_entity"]
    assert len(ent_q.criteria) <= MAX_OPTIONS_PER_QUESTION
