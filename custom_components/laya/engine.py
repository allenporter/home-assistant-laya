"""Local Laya decision engine running in-process via PyTorch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import gc
import logging
from typing import Any
from typing_extensions import override

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later

from .speculative.engine import DecisionEngine, PredictionResult
from .speculative.models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

_LOGGER = logging.getLogger(__name__)


class _ModelPoolEntry:
    """Entry in the process-level model cache keyed by device."""

    def __init__(self, device: str) -> None:
        """Initialize model pool entry for a device."""
        self.device = device
        self.agent: Any = None
        self.load_lock = asyncio.Lock()
        self.active_consumers: int = 0
        self.idle_timer_cancel: Callable[[], None] | None = None
        self.idle_timeout: float | None = None
        self.hass: HomeAssistant | None = None

    def cancel_idle_timer(self) -> None:
        """Cancel any pending idle unload timer."""
        if self.idle_timer_cancel is not None:
            self.idle_timer_cancel()
            self.idle_timer_cancel = None

    def schedule_idle_unload(self) -> None:
        """Schedule unload after idle_timeout when no active consumers remain."""
        self.cancel_idle_timer()
        if self.idle_timeout is None or self.idle_timeout < 0 or self.agent is None:
            return

        def _on_timeout(*_: Any) -> None:
            if self.active_consumers == 0:
                _LOGGER.debug(
                    "Idle timeout (%.1fs) reached for device %s; unloading Laya model",
                    self.idle_timeout,
                    self.device,
                )
                if self.hass is not None and self.hass.is_running:
                    self.hass.async_create_background_task(
                        self.async_unload(),
                        f"laya-idle-unload-{self.device}",
                    )
                else:
                    try:
                        asyncio.create_task(self.async_unload())
                    except RuntimeError:
                        pass

        if self.idle_timeout == 0:
            _on_timeout()
            return

        if self.hass is not None:
            self.idle_timer_cancel = async_call_later(
                self.hass, self.idle_timeout, _on_timeout
            )
        else:
            try:
                loop = asyncio.get_running_loop()
                handle = loop.call_later(self.idle_timeout, _on_timeout)
                self.idle_timer_cancel = handle.cancel
            except RuntimeError:
                pass

    async def async_unload(self) -> None:
        """Evict model from memory."""
        self.cancel_idle_timer()
        async with self.load_lock:
            if self.agent is not None:
                _LOGGER.debug("Unloading Laya model (%s) from memory", self.device)
                self.agent = None
                gc.collect()


_LOADED_MODELS: dict[str, _ModelPoolEntry] = {}


def get_loaded_models() -> dict[str, Any]:
    """Return map of currently loaded model agents by device."""
    return {
        device: entry.agent
        for device, entry in _LOADED_MODELS.items()
        if entry.agent is not None
    }


def _get_or_create_entry(device: str) -> _ModelPoolEntry:
    """Get or create model pool entry for device."""
    if device not in _LOADED_MODELS:
        _LOADED_MODELS[device] = _ModelPoolEntry(device)
    return _LOADED_MODELS[device]


async def async_unload_all_models() -> None:
    """Unload all cached models across all devices from memory immediately."""
    for entry in list(_LOADED_MODELS.values()):
        await entry.async_unload()
    _LOADED_MODELS.clear()


class LocalLayaEngine(DecisionEngine):
    """In-process Laya decision engine using local PyTorch model weights with idle unload."""

    def __init__(
        self,
        device: str = "auto",
        hass: HomeAssistant | None = None,
        idle_timeout: float | None = 0.0,
    ) -> None:
        """Initialize LocalLayaEngine."""
        self._device = device
        self._hass = hass
        self._idle_timeout = idle_timeout

    @property
    def _agent(self) -> Any:
        """Return the shared model agent for this engine's device if loaded."""
        entry = _LOADED_MODELS.get(self._device)
        return entry.agent if entry is not None else None

    @property
    def loaded(self) -> bool:
        """Return whether the model is currently loaded in memory."""
        return self._agent is not None

    def _load_model_sync(self) -> Any:
        """Synchronously load the Laya model checkpoint."""
        import laya

        kwargs: dict[str, Any] = {}
        if self._device != "auto":
            kwargs["device"] = self._device
        return laya.load(**kwargs)

    async def async_load(self) -> None:
        """Ensure the local model is loaded asynchronously in executor."""
        entry = _get_or_create_entry(self._device)
        if entry.agent is not None:
            entry.cancel_idle_timer()
            entry.hass = self._hass
            entry.idle_timeout = self._idle_timeout
            if (
                entry.active_consumers == 0
                and self._idle_timeout is not None
                and self._idle_timeout > 0
            ):
                entry.schedule_idle_unload()
            return

        async with entry.load_lock:
            if entry.agent is not None:
                entry.cancel_idle_timer()
                entry.hass = self._hass
                entry.idle_timeout = self._idle_timeout
                if (
                    entry.active_consumers == 0
                    and self._idle_timeout is not None
                    and self._idle_timeout > 0
                ):
                    entry.schedule_idle_unload()
                return

            _LOGGER.debug("Loading Laya model (device: %s)...", self._device)
            if self._hass is not None:
                agent = await self._hass.async_add_executor_job(self._load_model_sync)
            else:
                agent = await asyncio.to_thread(self._load_model_sync)
            _LOGGER.debug("Laya model successfully loaded")

            entry.agent = agent
            entry.hass = self._hass
            entry.idle_timeout = self._idle_timeout

            if (
                entry.active_consumers == 0
                and self._idle_timeout is not None
                and self._idle_timeout > 0
            ):
                entry.schedule_idle_unload()

    def _predict_sync(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        """Run synchronous forward pass on agent."""
        agent = self._agent
        if agent is None:
            raise RuntimeError("Laya model is not loaded in memory")
        return agent.predict(state, questions)

    async def async_predict(
        self,
        state: dict[str, Any] | str,
        questions: Mapping[str, Question | dict[str, Any]],
    ) -> PredictionResult:
        """Run inference over questions in executor thread."""
        entry = _get_or_create_entry(self._device)
        entry.cancel_idle_timer()
        entry.active_consumers += 1

        try:
            await self.async_load()

            serialized_questions: dict[str, dict[str, Any]] = {}
            for qid, q in questions.items():
                if isinstance(q, ChoiceQuestion):
                    serialized_questions[qid] = {
                        "type": "choice",
                        "instructions": q.instructions,
                        "criteria": q.criteria,
                    }
                elif isinstance(q, NoulQuestion):
                    serialized_questions[qid] = {
                        "type": "noul",
                        "instructions": q.instructions,
                    }
                elif isinstance(q, ScoreQuestion):
                    serialized_questions[qid] = {
                        "type": "score",
                        "instructions": q.instructions,
                        "criteria": q.criteria,
                    }
                elif isinstance(q, dict):
                    serialized_questions[qid] = q
                else:
                    serialized_questions[qid] = dict(q)

            if self._hass is not None:
                raw_res = await self._hass.async_add_executor_job(
                    self._predict_sync, state, serialized_questions
                )
            else:
                raw_res = await asyncio.to_thread(
                    self._predict_sync, state, serialized_questions
                )
        finally:
            entry.active_consumers -= 1
            if entry.active_consumers == 0:
                if self._idle_timeout == 0:
                    await self.async_unload(force=True)
                else:
                    entry.schedule_idle_unload()

        raw_answers = raw_res.get("answers", {})
        answers: dict[str, Answer] = {}

        for qid, ans_data in raw_answers.items():
            if not isinstance(ans_data, dict):
                continue
            qtype = ans_data.get("type")
            conf = float(ans_data.get("confidence", 0.0))
            action = ans_data.get("action", {})
            if qtype == "choice":
                raw_choice = ans_data.get("choice")
                choice_val = "" if raw_choice is None else str(raw_choice)
                answers[qid] = ChoiceAnswer(
                    choice=choice_val,
                    confidence=conf,
                    probabilities={
                        str(k): float(v)
                        for k, v in ans_data.get("probabilities", {}).items()
                    },
                    action=action if isinstance(action, dict) else {},
                )
            elif qtype == "noul":
                answers[qid] = NoulAnswer(
                    noul=float(ans_data.get("noul", 0.0)),
                    confidence=conf,
                    action=action if isinstance(action, dict) else {},
                )
            elif qtype == "score":
                answers[qid] = ScoreAnswer(
                    score=float(ans_data.get("score", 0.0)),
                    confidence=conf,
                    probabilities={
                        str(k): float(v)
                        for k, v in ans_data.get("probabilities", {}).items()
                    },
                    legend={
                        str(k): str(v) for k, v in ans_data.get("legend", {}).items()
                    },
                    action=action if isinstance(action, dict) else {},
                )

        return PredictionResult(
            answers=answers,
            model=raw_res.get("model"),
            usage=raw_res.get("usage", {}),
        )

    @override
    async def async_unload(self, force: bool = False) -> None:
        """Unload model from memory.

        If force is False and idle_timeout > 0, the model weights remain warm in the
        pool so subsequent config entries or tests can reuse them without re-loading.
        """
        entry = _LOADED_MODELS.get(self._device)
        if entry is None or entry.agent is None:
            return

        if not force and self._idle_timeout is not None and self._idle_timeout > 0:
            _LOGGER.debug(
                "async_unload called for device %s with idle_timeout=%.1fs; keeping model warm",
                self._device,
                self._idle_timeout,
            )
            if entry.active_consumers == 0:
                entry.schedule_idle_unload()
            return

        await entry.async_unload()


__all__ = [
    "LocalLayaEngine",
    "async_unload_all_models",
    "get_loaded_models",
]
