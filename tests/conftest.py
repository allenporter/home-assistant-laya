"""Fixtures for the custom component."""

from collections.abc import AsyncGenerator, Generator
import logging
from unittest.mock import patch

import pytest
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.laya.const import (
    CONF_DEVICE,
    DOMAIN,
)
from custom_components.laya.engine import FakeDecisionEngine

_LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None, None, None]:
    """Enable custom integration."""
    _ = enable_custom_integrations  # unused
    yield


@pytest.fixture(name="mock_engine")
def mock_engine_fixture() -> FakeDecisionEngine:
    """Fixture for FakeDecisionEngine."""
    return FakeDecisionEngine()


@pytest.fixture(name="mock_local_engine", autouse=True)
def mock_local_engine_fixture(
    mock_engine: FakeDecisionEngine,
) -> Generator[FakeDecisionEngine, None, None]:
    """Patch LocalLayaEngine to use FakeDecisionEngine in tests."""
    with patch(
        "custom_components.laya.LocalLayaEngine",
        return_value=mock_engine,
    ):
        yield mock_engine


@pytest.fixture(autouse=True)
async def mock_dependencies(
    hass: HomeAssistant,
) -> None:
    """Set up component dependencies."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})


@pytest.fixture(name="platforms")
def mock_platforms() -> list[Platform]:
    """Fixture for platforms loaded by the integration."""
    return [Platform.CONVERSATION]


@pytest.fixture(name="config_entry")
async def mock_config_entry(
    hass: HomeAssistant,
) -> MockConfigEntry:
    """Fixture to create a mock configuration entry."""
    config_entry = MockConfigEntry(
        data={CONF_DEVICE: "cpu"},
        domain=DOMAIN,
        options={},
    )
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry


@pytest.fixture(name="setup_integration")
async def mock_setup_integration(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    platforms: list[Platform],
) -> AsyncGenerator[None, None]:
    """Set up the integration."""
    with patch(f"custom_components.{DOMAIN}.PLATFORMS", platforms):
        assert await async_setup_component(hass, DOMAIN, {})
        await hass.async_block_till_done()
        yield
