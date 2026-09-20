"""Models and dataclasses for Laya."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .speculative import DecisionEngine, DecisionStrategy

type LayaConfigEntry = ConfigEntry[LayaData]


@dataclass(slots=True)
class LayaData:
    """Runtime data stored in ConfigEntry."""

    engine: DecisionEngine
    strategy: DecisionStrategy
