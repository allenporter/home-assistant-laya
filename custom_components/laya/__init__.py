"""Laya custom component."""

from __future__ import annotations

import logging

from collections.abc import Mapping
from typing import Any

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DEVICE,
    CONF_DOMAIN_FILTER_MODE,
    CONF_IDLE_TIMEOUT,
    CONF_RETRIEVER_TYPE,
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_DOMAIN_FILTER_MODE,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_RETRIEVER_TYPE,
)
from .engine import LocalLayaEngine
from .models import LayaConfigEntry, LayaData
from .speculative.flow import (
    DecisionFlow,
    create_decision_flow,
    create_exhaustive_flow,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION,)


def create_flow_from_options(options: Mapping[str, Any]) -> DecisionFlow:
    """Create a DecisionFlow configured from config entry options."""
    retriever_type = options.get(CONF_RETRIEVER_TYPE, DEFAULT_RETRIEVER_TYPE)
    confidence_threshold = float(
        options.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
    )
    compound_threshold = float(
        options.get(CONF_COMPOUND_THRESHOLD, DEFAULT_COMPOUND_THRESHOLD)
    )

    if retriever_type == "exhaustive":
        return create_exhaustive_flow(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
        )

    domain_filter_mode = options.get(
        CONF_DOMAIN_FILTER_MODE, DEFAULT_DOMAIN_FILTER_MODE
    )
    return create_decision_flow(
        confidence_threshold=confidence_threshold,
        compound_threshold=compound_threshold,
        domain_filter_mode=domain_filter_mode,
    )


async def async_setup_entry(hass: HomeAssistant, entry: LayaConfigEntry) -> bool:
    """Set up a config entry."""
    device = entry.data.get(CONF_DEVICE, DEFAULT_DEVICE)
    idle_timeout = float(
        entry.options.get(
            CONF_IDLE_TIMEOUT,
            entry.data.get(CONF_IDLE_TIMEOUT, DEFAULT_IDLE_TIMEOUT),
        )
    )
    engine = LocalLayaEngine(device=device, hass=hass, idle_timeout=idle_timeout)
    flow = create_flow_from_options(entry.options)

    entry.runtime_data = LayaData(engine=engine, flow=flow)

    # Eagerly preload model weights if keepalive is enabled (> 0)
    if idle_timeout > 0:
        await engine.async_load()

    await hass.config_entries.async_forward_entry_setups(
        entry,
        platforms=PLATFORMS,
    )

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LayaConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(
        entry,
        PLATFORMS,
    ):
        await entry.runtime_data.engine.async_unload()
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: LayaConfigEntry) -> None:
    """Reload config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
