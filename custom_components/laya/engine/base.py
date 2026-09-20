"""Base interfaces and models for decision engines (Laya / Jev)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


# --- Question Schemas ---


@dataclass(slots=True, frozen=True)
class ChoiceQuestion:
    """Categorical choice question with candidate criteria."""

    instructions: str
    criteria: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to engine question dictionary."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(slots=True, frozen=True)
class NoulQuestion:
    """Calibrated boolean (Yes/No) decision question."""

    instructions: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to engine question dictionary."""
        return {
            "type": "noul",
            "instructions": self.instructions,
        }


@dataclass(slots=True, frozen=True)
class ScoreQuestion:
    """Ordinal score question on a defined rubric."""

    instructions: str
    criteria: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to engine question dictionary."""
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


type Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


# --- Answer Schemas ---


@dataclass(slots=True, frozen=True)
class ChoiceAnswer:
    """Answer for a categorical choice question."""

    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class NoulAnswer:
    """Answer for a calibrated boolean (Yes/No) noul question."""

    noul: float  # P(true) in [0.0, 1.0]
    confidence: float
    action: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ScoreAnswer:
    """Answer for an ordinal score question."""

    score: float
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    legend: dict[str, str] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)


type Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer


# --- Engine Result ---


@dataclass(slots=True)
class PredictionResult:
    """Result of an engine prediction over a batch of questions."""

    answers: dict[str, Answer]
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class DecisionEngine(ABC):
    """Abstract interface for a System 1 decision engine.

    Evaluates a batch of typed questions over state in a single, parallel forward pass.
    """

    @property
    def loaded(self) -> bool:
        """Return whether the engine model is currently loaded in memory."""
        return True

    @abstractmethod
    async def async_predict(
        self,
        state: dict[str, Any] | str,
        questions: Mapping[str, Any],
    ) -> PredictionResult:
        """Evaluate a batch of questions over state in a single forward pass."""

    async def async_load(self) -> None:
        """Eagerly load model or initialize resources."""

    async def async_unload(self) -> None:
        """Release any held resources/models."""
