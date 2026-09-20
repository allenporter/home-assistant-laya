"""Test the Laya config flow."""

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.laya.const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DEVICE,
    CONF_FALLBACK_AGENT,
    CONF_IDLE_TIMEOUT,
    DOMAIN,
)


async def test_user_step_creates_entry(hass: HomeAssistant) -> None:
    """Test user step creates a local Laya entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_DEVICE: "cpu"},
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert result2["title"] == "Laya"
    assert result2["data"][CONF_DEVICE] == "cpu"


async def test_options_flow_thresholds(hass: HomeAssistant) -> None:
    """Test options flow to configure decision thresholds."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
        options={},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_COMPOUND_THRESHOLD: 0.85,
            CONF_CONFIDENCE_THRESHOLD: 0.60,
            CONF_IDLE_TIMEOUT: 120.0,
        },
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_COMPOUND_THRESHOLD] == 0.85
    assert entry.options[CONF_CONFIDENCE_THRESHOLD] == 0.60
    assert entry.options[CONF_IDLE_TIMEOUT] == 120.0


async def test_options_flow_fallback_agent(hass: HomeAssistant) -> None:
    """Test options flow to configure System 2 fallback conversation agent."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
        options={},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM

    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_FALLBACK_AGENT: "conversation.home_assistant",
        },
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_FALLBACK_AGENT] == "conversation.home_assistant"
