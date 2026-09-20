"""Speculative Fan-Out decision strategy with dynamic discovery and candidate retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.helpers import intent

from ..engine import DecisionEngine, PredictionResult
from ..models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from .base import Decision, DecisionStrategy, StrategyContext
from .discovery import (
    ONOFF_DOMAINS,
    discover_intents,
    get_allowed_domains_for_intents,
    lexical_score,
    tokenize,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50
DEFAULT_COMPOUND_THRESHOLD: float = 0.50


class SpeculativeFanOutStrategy(DecisionStrategy):
    """Speculative fan-out strategy with dynamic intent and candidate discovery."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    ) -> None:
        """Initialize SpeculativeFanOutStrategy."""
        self._confidence_threshold = confidence_threshold
        self._compound_threshold = compound_threshold

    @property
    def confidence_threshold(self) -> float:
        """Return the confidence threshold."""
        return self._confidence_threshold

    @property
    def compound_threshold(self) -> float:
        """Return the compound threshold."""
        return self._compound_threshold

    def _discover_intents(
        self, context: StrategyContext, utterance: str
    ) -> dict[str, str]:
        """Discover and rank candidate intent schemas."""
        return discover_intents(context, utterance, max_options=5)

    def _rank_entities(
        self,
        context: StrategyContext,
        utterance: str,
        top_area_ids: set[str] | None = None,
        allowed_domains: set[str] | None = None,
    ) -> dict[str, str]:
        """Discover and rank candidate exposed entities."""
        query_tokens = tokenize(utterance)
        scored_entities: list[tuple[float, str, str]] = []
        target_domains = (
            allowed_domains if allowed_domains is not None else ONOFF_DOMAINS
        )

        states = context.states if context.states is not None else []
        for state in states:
            domain = state.domain
            if not domain or domain not in target_domains:
                continue

            entity_id = state.entity_id
            friendly_name = state.name

            area_name = ""
            area_id = ""
            if context.entity_registry:
                entry = context.entity_registry.async_get(entity_id)
                if entry and entry.area_id:
                    area_id = entry.area_id
                    if context.area_registry:
                        area = context.area_registry.async_get_area(entry.area_id)
                        if area and area.name:
                            area_name = area.name

            name_score = lexical_score(query_tokens, friendly_name, utterance)
            id_score = lexical_score(
                query_tokens, entity_id.replace("_", " "), utterance
            )
            score = max(name_score, id_score)

            if domain in query_tokens:
                score += 0.5

            if area_name and lexical_score(query_tokens, area_name, utterance) > 0:
                score += 1.0
            elif top_area_ids and area_id and area_id in top_area_ids:
                score += 1.0

            description = f"{friendly_name} ({domain})"
            if area_name:
                description += f" in {area_name}"

            scored_entities.append((score, entity_id, description))

        scored_entities.sort(key=lambda x: x[0], reverse=True)

        criteria: dict[str, str] = {}
        for _, eid, desc in scored_entities[:20]:
            criteria[eid] = desc

        if criteria:
            criteria["none"] = "None of the listed devices"

        return criteria

    def _rank_areas(self, context: StrategyContext, utterance: str) -> dict[str, str]:
        """Discover and rank candidate areas."""
        query_tokens = tokenize(utterance)
        scored_areas: list[tuple[float, str, str]] = []

        areas = (
            context.area_registry.async_list_areas() if context.area_registry else []
        )

        for area in areas:
            if not area or not area.name:
                continue
            score = lexical_score(query_tokens, area.name, utterance)
            id_score = lexical_score(query_tokens, area.id.replace("_", " "), utterance)
            scored_areas.append((max(score, id_score), area.id, area.name))

        scored_areas.sort(key=lambda x: x[0], reverse=True)

        criteria: dict[str, str] = {}
        for _, aid, name in scored_areas[:20]:
            criteria[aid] = name

        if criteria:
            criteria["none"] = "None of the listed areas"

        return criteria

    @staticmethod
    def _safe_choice(answers: dict[str, Answer], key: str) -> str | None:
        """Safely extract string choice from answer primitive."""
        answer = answers.get(key)
        if isinstance(answer, ChoiceAnswer) and answer.choice:
            return answer.choice
        return None

    def _parse_evaluation_response(
        self,
        response: PredictionResult,
        utterance: str,
        context: StrategyContext,
    ) -> Decision:
        """Safely parse and validate engine PredictionResult."""
        if not isinstance(response, PredictionResult) or not isinstance(
            response.answers, dict
        ):
            _LOGGER.warning(
                "Decision engine evaluation returned non-PredictionResult response (%s): %r",
                type(response).__name__,
                response,
            )
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Malformed engine response: expected PredictionResult",
            )

        answers = response.answers

        # 1. Check compound command condition safely
        compound_ans = answers.get("is_compound")
        if (
            isinstance(compound_ans, NoulAnswer)
            and compound_ans.noul > self._compound_threshold
        ):
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                is_compound=True,
                escalation_reason="Compound command detected",
                raw_answers=answers,
            )

        # 2. Check intent choice and confidence safely
        intent_ans = answers.get("intent")
        if not isinstance(intent_ans, ChoiceAnswer):
            _LOGGER.warning(
                "Response missing or invalid 'intent' answer: %r", intent_ans
            )
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Missing or invalid intent answer",
                raw_answers=answers,
            )

        intent_choice = intent_ans.choice if intent_ans.choice else None
        top_prob = (
            intent_ans.probabilities.get(intent_choice, intent_ans.confidence)
            if intent_choice
            else intent_ans.confidence
        )

        if (
            not intent_choice
            or intent_choice.lower() in ("unmatched", "none", "other", "")
            or top_prob < self._confidence_threshold
        ):
            return Decision(
                intent_name=None,
                confidence=top_prob,
                should_escalate=True,
                escalation_reason="Unhandled intent or low confidence",
                raw_answers=answers,
            )

        # 3. Resolve targets and slots safely
        slots: dict[str, Any] = {}
        target_type_choice = self._safe_choice(answers, "target_type")
        target_area_choice = self._safe_choice(answers, "target_area")
        target_entity_choice = self._safe_choice(answers, "target_entity")

        resolved_entity: str | None = None
        resolved_area: str | None = None
        resolved_domain: str | None = None

        has_valid_area = target_area_choice not in (None, "none")
        has_valid_entity = target_entity_choice not in (None, "none")

        if (target_type_choice == "area" and has_valid_area) or (
            has_valid_area and not has_valid_entity
        ):
            area_entry = (
                context.area_registry.async_get_area(target_area_choice)
                if context.area_registry and target_area_choice
                else None
            )
            resolved_area = (
                area_entry.name if area_entry and area_entry.name else None
            ) or target_area_choice
            slots["area"] = resolved_area

            for dom in (
                "light",
                "switch",
                "cover",
                "climate",
                "media_player",
                "fan",
            ):
                if dom in utterance.lower():
                    resolved_domain = dom
                    slots["domain"] = dom
                    break
        elif has_valid_entity:
            resolved_entity = target_entity_choice
            slots["entity_id"] = target_entity_choice
            if target_entity_choice and "." in target_entity_choice:
                resolved_domain = target_entity_choice.split(".", 1)[0]

        # 4. Extract continuous numeric parameters via regex safely
        brightness_match = re.search(r"(\d+)\s*%", utterance)
        if brightness_match:
            try:
                slots["brightness"] = int(brightness_match.group(1))
            except ValueError:
                pass

        temp_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:degrees|deg|°)", utterance, re.IGNORECASE
        )
        if temp_match:
            try:
                slots["temperature"] = float(temp_match.group(1))
            except ValueError:
                pass

        return Decision(
            intent_name=intent_choice,
            entity_id=resolved_entity,
            area_name=resolved_area,
            domain=resolved_domain,
            slots=slots,
            confidence=top_prob,
            should_escalate=False,
            raw_answers=answers,
        )

    def _build_questions(
        self,
        context: StrategyContext,
        utterance: str,
    ) -> tuple[dict[str, Question], dict[str, Any]]:
        """Build canonical Question objects and serialized dictionary representation."""
        intent_criteria = self._discover_intents(context, utterance)
        area_criteria = self._rank_areas(context, utterance)

        handlers_map: dict[str, intent.IntentHandler] = {}
        if context.hass:
            handlers_map = {h.intent_type: h for h in intent.async_get(context.hass)}

        candidate_intent_types = [k for k in intent_criteria.keys() if k != "unmatched"]
        allowed_domains = get_allowed_domains_for_intents(
            candidate_intent_types, handlers_map
        )

        entity_criteria = self._rank_entities(
            context,
            utterance,
            set(area_criteria.keys()),
            allowed_domains=allowed_domains,
        )

        canonical: dict[str, Question] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,  # type: ignore[arg-type]
            ),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ),
        }

        serialized: dict[str, Any] = {
            "intent": {
                "type": "choice",
                "instructions": "Determine the primary Home Assistant action",
                "criteria": intent_criteria,
            },
            "is_compound": {
                "type": "noul",
                "instructions": "Does the request contain multiple distinct commands or conjunctions?",
            },
        }

        if len(entity_criteria) > 1:
            canonical["target_entity"] = ChoiceQuestion(
                instructions="Which entity is the user referring to?",
                criteria=entity_criteria,  # type: ignore[arg-type]
            )
            serialized["target_entity"] = {
                "type": "choice",
                "instructions": "Which entity is the user referring to?",
                "criteria": entity_criteria,
            }

        if len(area_criteria) > 1:
            canonical["target_area"] = ChoiceQuestion(
                instructions="Which area or room is the user referring to?",
                criteria=area_criteria,  # type: ignore[arg-type]
            )
            serialized["target_area"] = {
                "type": "choice",
                "instructions": "Which area or room is the user referring to?",
                "criteria": area_criteria,
            }

        if len(entity_criteria) > 1 and len(area_criteria) > 1:
            target_type_crit = {
                "entity": "A specific individual device or appliance",
                "area": "An entire room or area",
            }
            canonical["target_type"] = ChoiceQuestion(
                instructions="Is the user targeting an individual device or an entire area?",
                criteria=target_type_crit,
            )
            serialized["target_type"] = {
                "type": "choice",
                "instructions": "Is the user targeting an individual device or an entire area?",
                "criteria": target_type_crit,
            }

        return canonical, serialized

    async def async_decide(
        self,
        engine: DecisionEngine,
        text: str,
        context: StrategyContext,
    ) -> Decision:
        """Evaluate utterance using speculative fan-out typed questions."""
        canonical_questions, _serialized_questions = self._build_questions(
            context, text
        )

        state = {
            "utterance": text,
            "home": context.home_name,
        }

        try:
            prediction = await engine.async_predict(
                state=state, questions=canonical_questions
            )
            return self._parse_evaluation_response(prediction, text, context)
        except Exception as err:
            _LOGGER.warning("Engine prediction failed: %s", err)
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=f"Engine prediction error: {err}",
            )
