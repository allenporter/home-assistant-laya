"""Tests for the Laya conversation agent."""

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar, intent
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.laya.const import (
    CONF_DEVICE,
    CONF_FALLBACK_AGENT,
    DOMAIN,
)
from custom_components.laya.engine import (
    ChoiceAnswer,
    FakeDecisionEngine,
    NoulAnswer,
    ScoreAnswer,
)


async def test_turn_on_area(hass: HomeAssistant) -> None:
    """Test processing a turn on command for an area through Laya conversation agent."""
    area_reg = ar.async_get(hass)
    area_reg.async_get_or_create("kitchen")

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    engine: FakeDecisionEngine = entry.runtime_data.engine
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.98),
            "target_type": ChoiceAnswer(choice="area", confidence=0.95),
            "target_domain": ChoiceAnswer(choice="light", confidence=0.95),
            "target_area": ChoiceAnswer(choice="kitchen", confidence=0.95),
            "light_action": ChoiceAnswer(choice="turn_on", confidence=0.98),
            "is_compound": NoulAnswer(noul=0.01, confidence=0.99),
        }
    )

    handled_intents: list[intent.Intent] = []

    class MockTurnOnHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            handled_intents.append(intent_obj)
            res = intent.IntentResponse(language=intent_obj.language)
            res.async_set_speech("Turned on kitchen lights")
            return res

    intent.async_register(hass, MockTurnOnHandler())

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id="conv_1",
        context=Context(),
        agent_id=entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert len(handled_intents) == 1
    assert handled_intents[0].slots["area"]["value"] == "kitchen"
    assert handled_intents[0].slots["domain"]["value"] == "light"


async def test_turn_off_entity(hass: HomeAssistant) -> None:
    """Test processing a turn off command for a specific device entity."""
    hass.states.async_set("light.desk_lamp", "on", {"friendly_name": "Desk Lamp"})

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    engine: FakeDecisionEngine = entry.runtime_data.engine
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.95),
            "target_domain": ChoiceAnswer(choice="light", confidence=0.95),
            "target_entity": ChoiceAnswer(choice="light.desk_lamp", confidence=0.95),
            "light_action": ChoiceAnswer(choice="turn_off", confidence=0.98),
            "is_compound": NoulAnswer(noul=0.01, confidence=0.99),
        }
    )

    handled_intents: list[intent.Intent] = []

    class MockTurnOffHandler(intent.IntentHandler):
        intent_type = "HassTurnOff"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            handled_intents.append(intent_obj)
            res = intent.IntentResponse(language=intent_obj.language)
            res.async_set_speech("Turned off desk lamp")
            return res

    intent.async_register(hass, MockTurnOffHandler())

    result = await conversation.async_converse(
        hass=hass,
        text="Turn off desk lamp",
        conversation_id="conv_desk",
        context=Context(),
        agent_id=entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert len(handled_intents) == 1
    assert handled_intents[0].slots["name"]["value"] == "Desk Lamp"


async def test_climate_temperature(hass: HomeAssistant) -> None:
    """Test processing a climate temperature adjustment command."""
    hass.states.async_set(
        "climate.living_room", "heat", {"friendly_name": "Living Room AC"}
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    engine: FakeDecisionEngine = entry.runtime_data.engine
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassClimateSetTemperature", confidence=0.93),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.90),
            "target_domain": ChoiceAnswer(choice="climate", confidence=0.95),
            "target_entity": ChoiceAnswer(
                choice="climate.living_room", confidence=0.95
            ),
            "target_temperature": ScoreAnswer(score=2.0, confidence=0.92),
            "is_compound": NoulAnswer(noul=0.02, confidence=0.98),
        }
    )

    handled_intents: list[intent.Intent] = []

    class MockClimateHandler(intent.IntentHandler):
        intent_type = "HassClimateSetTemperature"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            handled_intents.append(intent_obj)
            res = intent.IntentResponse(language=intent_obj.language)
            res.async_set_speech("Set temperature to 65")
            return res

    intent.async_register(hass, MockClimateHandler())

    result = await conversation.async_converse(
        hass=hass,
        text="Set the living room thermostat to 65 degrees",
        conversation_id="conv_clim",
        context=Context(),
        agent_id=entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert len(handled_intents) == 1
    assert handled_intents[0].slots["name"]["value"] == "Living Room AC"
    assert handled_intents[0].slots["temperature"]["value"] == 65.0


async def test_compound_escalation_no_fallback(hass: HomeAssistant) -> None:
    """Test compound utterance returns informative error when no fallback agent is configured."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    engine: FakeDecisionEngine = entry.runtime_data.engine
    engine.set_default_answers(
        {
            "is_compound": NoulAnswer(noul=0.92, confidence=0.92),
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.8),
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn off the music and turn on living room lights",
        conversation_id="conv_2",
        context=Context(),
        agent_id=entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert result.response.error_code == intent.IntentResponseErrorCode.NO_INTENT_MATCH
    assert "multiple requests" in result.response.speech["plain"]["speech"].lower()


async def test_compound_escalation_with_fallback(hass: HomeAssistant) -> None:
    """Test compound utterance delegates to fallback System 2 agent when configured."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_DEVICE: "cpu",
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    engine: FakeDecisionEngine = entry.runtime_data.engine
    engine.set_default_answers(
        {
            "is_compound": NoulAnswer(noul=0.95, confidence=0.95),
            "intent": ChoiceAnswer(choice="none", confidence=0.1),
        }
    )

    fallback_called = []

    class MockFallbackAgent(conversation.AbstractConversationAgent):
        @property
        def supported_languages(self) -> list[str]:
            return ["en"]

        async def async_process(
            self, user_input: conversation.ConversationInput
        ) -> conversation.ConversationResult:
            fallback_called.append(user_input.text)
            res = intent.IntentResponse(language=user_input.language)
            res.async_set_speech("Fallback agent handled compound command")
            return conversation.ConversationResult(
                response=res,
                conversation_id=user_input.conversation_id,
            )

    manager = conversation.get_agent_manager(hass)
    manager.async_set_agent("mock_fallback_agent", MockFallbackAgent())

    result = await conversation.async_converse(
        hass=hass,
        text="Turn off the patio lights and start the dishwasher",
        conversation_id="conv_3",
        context=Context(),
        agent_id=entry.entry_id,
    )

    assert len(fallback_called) == 1
    assert fallback_called[0] == "Turn off the patio lights and start the dishwasher"
    assert (
        result.response.speech["plain"]["speech"]
        == "Fallback agent handled compound command"
    )


async def test_unhandled_intent_or_low_confidence(hass: HomeAssistant) -> None:
    """Test unhandled intent or low confidence returns NO_INTENT_MATCH."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    engine: FakeDecisionEngine = entry.runtime_data.engine
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="none", confidence=0.2),
            "is_compound": NoulAnswer(noul=0.05, confidence=0.95),
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="What is the weather outside?",
        conversation_id="conv_unhandled",
        context=Context(),
        agent_id=entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert result.response.error_code == intent.IntentResponseErrorCode.NO_INTENT_MATCH
    assert "could not understand" in result.response.speech["plain"]["speech"].lower()
