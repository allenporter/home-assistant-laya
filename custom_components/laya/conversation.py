"""Conversation entity for Laya."""

from __future__ import annotations

import logging
from typing import Any, Literal
from typing_extensions import override

from homeassistant.components import conversation
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    intent,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_FALLBACK_AGENT, DOMAIN
from .engine.base import DecisionEngine
from .models import LayaConfigEntry
from .strategy.base import DecisionStrategy, StrategyContext

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LayaConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    data = config_entry.runtime_data
    entity = LayaConversationEntity(config_entry, data.engine, data.strategy)
    async_add_entities([entity])


class LayaConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Laya System 1 conversation entity."""

    _attr_has_entity_name = True

    def __init__(
        self,
        entry: LayaConfigEntry,
        engine: DecisionEngine,
        strategy: DecisionStrategy,
    ) -> None:
        """Initialize LayaConversationEntity."""
        self._entry = entry
        self._engine = engine
        self._strategy = strategy
        self._attr_unique_id = entry.entry_id
        # Determine the entity name from the config entry title
        self._attr_name = entry.title

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @override
    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    @override
    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process incoming user utterance through decision engine."""
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)

        # Filter entities exposed to Home Assistant Assist
        all_states = self.hass.states.async_all()
        exposed_states = [
            state
            for state in all_states
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        active_states = exposed_states if exposed_states else all_states

        context = StrategyContext(
            hass=self.hass,
            area_registry=area_reg,
            entity_registry=entity_reg,
            states=active_states,
            home_name=self.hass.config.location_name,
            language=user_input.language,
            device_id=user_input.device_id,
        )

        decision = await self._strategy.async_decide(
            self._engine, user_input.text, context
        )

        # Handle escalation (e.g. compound command or low confidence)
        if decision.should_escalate or not decision.intent_name:
            fallback_agent = self._entry.options.get(
                CONF_FALLBACK_AGENT
            ) or self._entry.data.get(CONF_FALLBACK_AGENT)
            if fallback_agent:
                _LOGGER.debug(
                    "Escalating utterance %r to System 2 fallback agent: %s (reason: %s)",
                    user_input.text,
                    fallback_agent,
                    decision.escalation_reason,
                )
                return await conversation.async_converse(
                    hass=self.hass,
                    text=user_input.text,
                    conversation_id=user_input.conversation_id,
                    context=user_input.context,
                    language=user_input.language,
                    agent_id=fallback_agent,
                    device_id=user_input.device_id,
                )

            intent_response = intent.IntentResponse(language=user_input.language)
            if decision.is_compound:
                msg = (
                    "I heard multiple requests. Please give one command at a time or "
                    "configure a System 2 fallback agent."
                )
            else:
                msg = "Sorry, I could not understand that request."
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.NO_INTENT_MATCH, msg
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        # Format slots for Home Assistant intent handling
        formatted_slots: dict[str, Any] = {}
        for slot_key, slot_val in decision.slots.items():
            if slot_key == "entity_id":
                state = self.hass.states.get(slot_val)
                name = state.attributes.get("friendly_name") if state else slot_val
                formatted_slots["name"] = {"value": name}
            else:
                formatted_slots[slot_key] = {"value": slot_val}

        try:
            intent_response = await intent.async_handle(
                self.hass,
                DOMAIN,
                decision.intent_name,
                formatted_slots,
                user_input.text,
                user_input.context,
                user_input.language,
            )
        except intent.IntentHandleError as err:
            _LOGGER.warning("Error handling intent %s: %s", decision.intent_name, err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                str(err),
            )

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )
