"""Tests for the laya component setup and unload."""

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.laya.const import CONF_DEVICE, DOMAIN


async def test_setup_and_unload_entry(hass: HomeAssistant) -> None:
    """Test successful setup and unload of a config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_DEVICE: "cpu"},
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data is not None

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED
