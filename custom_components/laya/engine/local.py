"""Local Laya decision engine running in-process via PyTorch."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from homeassistant.core import HomeAssistant

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


class LocalLayaEngine(DecisionEngine):
    """In-process Laya decision engine using the local PyTorch model weights."""

    def __init__(
        self,
        device: str = "auto",
        hass: HomeAssistant | None = None,
    ) -> None:
        """Initialize LocalLayaEngine."""
        self._device = device
        self._hass = hass
        self._agent: Any = None
        self._load_lock = asyncio.Lock()

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

            if self._hass is not None:
                self._agent = await self._hass.async_add_executor_job(
                    self._load_model_sync
                )
            else:
                self._agent = await asyncio.to_thread(self._load_model_sync)

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
        self._agent = None
