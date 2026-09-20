"""Local Laya decision engine running in-process via PyTorch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import gc
import logging
from typing import Any
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later

from .base import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionEngine,
    NoulAnswer,
    NoulQuestion,
    PredictionResult,
    ScoreAnswer,
    ScoreQuestion,
)

_LOGGER = logging.getLogger(__name__)


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
        self._agent: Any = None
        self._load_lock = asyncio.Lock()
        self._active_consumers: int = 0
        self._idle_timer_cancel: Callable[[], None] | None = None

    @property
    def loaded(self) -> bool:
        """Return whether the model is currently loaded in memory."""
        return self._agent is not None

    def _cancel_idle_timer(self) -> None:
        """Cancel any pending idle unload timer."""
        if self._idle_timer_cancel is not None:
            self._idle_timer_cancel()
            self._idle_timer_cancel = None

    def _schedule_idle_unload(self) -> None:
        """Schedule unload after idle_timeout when no active consumers remain."""
        self._cancel_idle_timer()
        if self._idle_timeout is None or self._idle_timeout < 0 or self._agent is None:
            return

        def _on_timeout(*_: Any) -> None:
            if self._active_consumers == 0:
                _LOGGER.debug(
                    "Idle timeout (%.1fs) reached with 0 consumers; unloading Laya model",
                    self._idle_timeout,
                )
                if self._hass is not None:
                    self._hass.async_create_background_task(
                        self.async_unload(),
                        "laya-idle-unload",
                    )
                else:
                    asyncio.create_task(self.async_unload())

        if self._idle_timeout == 0:
            _on_timeout()
            return

        if self._hass is not None:
            self._idle_timer_cancel = async_call_later(
                self._hass, self._idle_timeout, _on_timeout
            )
        else:
            loop = asyncio.get_running_loop()
            handle = loop.call_later(self._idle_timeout, _on_timeout)
            self._idle_timer_cancel = handle.cancel

    def _load_model_sync(self) -> Any:
        """Synchronously load the Laya model checkpoint."""
        import laya

        kwargs: dict[str, Any] = {}
        if self._device != "auto":
            kwargs["device"] = self._device
        return laya.load(**kwargs)

    async def async_load(self) -> None:
        """Ensure the local model is loaded asynchronously in executor."""
        if self._agent is not None:
            return

        async with self._load_lock:
            if self._agent is not None:
                return

            _LOGGER.debug("Loading Laya model (device: %s)...", self._device)
            if self._hass is not None:
                self._agent = await self._hass.async_add_executor_job(
                    self._load_model_sync
                )
            else:
                self._agent = await asyncio.to_thread(self._load_model_sync)
            _LOGGER.debug("Laya model successfully loaded")

            if (
                self._active_consumers == 0
                and self._idle_timeout is not None
                and self._idle_timeout > 0
            ):
                self._schedule_idle_unload()

    def _predict_sync(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, Any],
    ) -> dict[str, Any]:
        """Run synchronous forward pass on agent."""
        return self._agent.predict(state, questions)

    async def async_predict(
        self,
        state: dict[str, Any] | str,
        questions: Mapping[str, Any],
    ) -> PredictionResult:
        """Run inference over questions in executor thread."""
        self._cancel_idle_timer()
        self._active_consumers += 1

        try:
            await self.async_load()

            serialized_questions: dict[str, dict[str, Any]] = {}
            for qid, q in questions.items():
                if isinstance(q, (ChoiceQuestion, NoulQuestion, ScoreQuestion)):
                    serialized_questions[qid] = q.to_dict()
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
            self._active_consumers -= 1
            if self._active_consumers == 0:
                if self._idle_timeout == 0:
                    await self.async_unload()
                else:
                    self._schedule_idle_unload()

        raw_answers = raw_res.get("answers", {})
        answers: dict[str, Answer] = {}

        for qid, ans_data in raw_answers.items():
            qtype = ans_data.get("type")
            conf = float(ans_data.get("confidence", 0.0))
            action = ans_data.get("action", {})
            if qtype == "choice":
                answers[qid] = ChoiceAnswer(
                    choice=ans_data.get("choice", ""),
                    confidence=conf,
                    probabilities=ans_data.get("probabilities", {}),
                    action=action,
                )
            elif qtype == "noul":
                answers[qid] = NoulAnswer(
                    noul=float(ans_data.get("noul", 0.0)),
                    confidence=conf,
                    action=action,
                )
            elif qtype == "score":
                answers[qid] = ScoreAnswer(
                    score=float(ans_data.get("score", 0.0)),
                    confidence=conf,
                    probabilities=ans_data.get("probabilities", {}),
                    legend=ans_data.get("legend", {}),
                    action=action,
                )

        return PredictionResult(
            answers=answers,
            model=raw_res.get("model"),
            usage=raw_res.get("usage", {}),
        )

    async def async_unload(self) -> None:
        """Unload model from memory."""
        self._cancel_idle_timer()
        async with self._load_lock:
            if self._agent is not None:
                _LOGGER.debug("Unloading Laya model from memory")
                self._agent = None
                gc.collect()
