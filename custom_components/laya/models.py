"""Models and dataclasses for Laya."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .speculative.flow import DecisionFlow
from .speculative.scoring.engine import DecisionEngine

type LayaConfigEntry = ConfigEntry[LayaData]


@dataclass(slots=True)
class LayaData:
    """Runtime data stored in ConfigEntry."""

    engine: DecisionEngine
    flow: DecisionFlow
