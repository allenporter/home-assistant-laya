"""Tests for the laya component setup and unload."""

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from unittest.mock import MagicMock, patch

from custom_components.laya.const import CONF_DEVICE, CONF_IDLE_TIMEOUT, DOMAIN
from custom_components.laya.engine import (
    LocalLayaEngine,
    async_unload_all_models,
    get_loaded_models,
)


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


async def test_unload_entry_preserves_warm_model_when_idle_timeout(
    hass: HomeAssistant,
    mock_load: MagicMock,
) -> None:
    """Test unloading a config entry with idle_timeout > 0 keeps the model warm in the pool."""
    await async_unload_all_models()
    try:
        with patch("custom_components.laya.LocalLayaEngine", LocalLayaEngine):
            entry1 = MockConfigEntry(
                domain=DOMAIN,
                data={CONF_DEVICE: "cpu", CONF_IDLE_TIMEOUT: 10.0},
            )
            entry1.add_to_hass(hass)
            assert await hass.config_entries.async_setup(entry1.entry_id)
            await hass.async_block_till_done()

            # Eagerly preloaded on setup
            assert mock_load.call_count == 1
            assert "cpu" in get_loaded_models()

            # Unload entry1
            assert await hass.config_entries.async_unload(entry1.entry_id)
            await hass.async_block_till_done()

            # Model is still warm in pool
            assert "cpu" in get_loaded_models()

            # Next config entry is set up
            entry2 = MockConfigEntry(
                domain=DOMAIN,
                data={CONF_DEVICE: "cpu", CONF_IDLE_TIMEOUT: 10.0},
            )
            entry2.add_to_hass(hass)
            assert await hass.config_entries.async_setup(entry2.entry_id)
            await hass.async_block_till_done()

            # Reused without calling laya.load again
            assert mock_load.call_count == 1

            assert await hass.config_entries.async_unload(entry2.entry_id)
            await hass.async_block_till_done()
    finally:
        await async_unload_all_models()
