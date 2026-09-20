"""Speculative Fan-Out decision strategy with dynamic discovery and candidate retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any

import voluptuous as vol
from homeassistant.helpers import intent

from ..const import (
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_OPTIONS_PER_QUESTION,
)
from ..engine.base import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionEngine,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from .base import Decision, DecisionStrategy, StrategyContext

_LOGGER = logging.getLogger(__name__)

SUPPORTED_STRATEGY_SLOTS: frozenset[str] = frozenset(
    {"name", "area", "domain", "floor", "device_class", "brightness"}
)


def get_handler_slot_info(
    handler: intent.IntentHandler,
) -> tuple[set[str], set[str]]:
    """Extract (supported_slots, required_slots) from a Home Assistant IntentHandler."""
    supported: set[str] = set()
    required: set[str] = set()

    # Inspect required_slots and optional_slots if present
    req_slots = getattr(handler, "required_slots", None)
    if isinstance(req_slots, dict):
        req_keys = set(req_slots.keys())
        required.update(req_keys)
        supported.update(req_keys)
    opt_slots = getattr(handler, "optional_slots", None)
    if isinstance(opt_slots, dict):
        supported.update(opt_slots.keys())

    # Inspect slot_schema if present
    schema = getattr(handler, "slot_schema", None)

    if isinstance(schema, dict):
        for key in schema:
            is_req = isinstance(key, vol.Required)
            names: list[str] = []
            if isinstance(key, str):
                names.append(key)
            elif isinstance(key, vol.Marker) and isinstance(key.schema, str):
                names.append(key.schema)
            elif isinstance(key, vol.Any):
                for validator in key.validators:
                    if isinstance(validator, str):
                        names.append(validator)
                    elif isinstance(validator, vol.Marker) and isinstance(
                        validator.schema, str
                    ):
                        names.append(validator.schema)
            for name in names:
                supported.add(name)
                if is_req:
                    required.add(name)

    # Fallback for handlers with no explicit schema
    if not supported:
        supported = {"name", "area", "domain", "floor"}

    return supported, required


def can_fulfill_intent(handler: intent.IntentHandler) -> bool:
    """Check if the strategy has capabilities to satisfy all required slots of an intent."""
    _, required = get_handler_slot_info(handler)
    return required.issubset(SUPPORTED_STRATEGY_SLOTS)


CONTROLLABLE_DOMAINS = {
    "light",
    "switch",
    "climate",
    "media_player",
    "fan",
    "cover",
    "vacuum",
    "lock",
    "valve",
    "humidifier",
    "water_heater",
    "scene",
    "script",
}


STOP_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "in",
    "at",
    "for",
    "of",
    "or",
    "and",
    "is",
    "it",
    "please",
    "my",
    "your",
}


def _tokenize(text: str) -> set[str]:
    """Tokenize string into lowercase alphanumeric words, excluding common stop words."""
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _token_match(q_token: str, cand_token: str) -> bool:
    """Check if query token matches candidate token via equality or prefix/stem matching."""
    if q_token == cand_token:
        return True
    if len(q_token) >= 3 and len(cand_token) >= 3:
        if q_token.startswith(cand_token) or cand_token.startswith(q_token):
            return True
    return False


def _lexical_score(query_tokens: set[str], candidate: str, full_query: str) -> float:
    """Compute lexical matching score between query tokens and candidate name or description."""
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    matches = 0
    for q in query_tokens:
        for c in cand_tokens:
            if _token_match(q, c):
                matches += 1
                break

    score = float(matches)
    cand_lower = candidate.lower()
    query_lower = full_query.lower()
    if cand_lower in query_lower or query_lower in cand_lower:
        score += 2.0
    return score


class SpeculativeFanOutStrategy(DecisionStrategy):
    """Speculative fan-out strategy with dynamic intent and candidate discovery."""

    def __init__(
        self,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize SpeculativeFanOutStrategy."""
        self._compound_threshold = compound_threshold
        self._confidence_threshold = confidence_threshold

    def _discover_intents(
        self,
        context: StrategyContext,
        query_tokens: set[str],
        full_query: str,
        active_domains: set[str],
    ) -> tuple[dict[str, str], set[str]]:
        """Dynamically discover and rank candidate intents using Home Assistant intent handlers."""
        handlers = intent.async_get(context.hass) if context.hass else []
        handlers_by_name: dict[str, intent.IntentHandler] = {}

        scored_intents: list[tuple[float, str, str]] = []
        for handler in handlers:
            it_name = handler.intent_type
            if it_name in handlers_by_name:
                continue

            # Check if intent requires slots we cannot fulfill
            if not can_fulfill_intent(handler):
                continue

            handlers_by_name[it_name] = handler

            # Check domain compatibility if handler specifies domains or platforms
            req_domains = getattr(handler, "required_domains", None) or getattr(
                handler, "platforms", None
            )
            if req_domains and active_domains and not (req_domains & active_domains):
                continue

            domain = getattr(handler, "domain", None)
            if (
                domain
                and domain != "homeassistant"
                and active_domains
                and domain not in active_domains
            ):
                continue

            # Read metadata directly from the Home Assistant IntentHandler
            desc = (
                getattr(handler, "description", None)
                or handler.__doc__
                or f"Handle {it_name.replace('Hass', '').strip()}"
            )

            # Score intent against query tokens using the intent description and name
            desc_score = _lexical_score(query_tokens, desc, full_query)
            name_score = _lexical_score(
                query_tokens, it_name.replace("Hass", " "), full_query
            )
            score = desc_score + name_score
            scored_intents.append((score, it_name, desc))

        # Fallback if no handlers were registered in Home Assistant
        if not scored_intents:
            scored_intents = [
                (1.0, "HassTurnOn", "Turn on or activate device"),
                (1.0, "HassTurnOff", "Turn off or deactivate device"),
            ]

        # Sort descending by score
        scored_intents.sort(key=lambda item: item[0], reverse=True)

        # Select top candidate intents (preferring positively scored intents, leaving 1 slot for other)
        max_intents = max(1, MAX_OPTIONS_PER_QUESTION - 1)
        selected_criteria: dict[str, str] = {}
        candidate_supported_slots: set[str] = set()

        positive_intents = [item for item in scored_intents if item[0] > 0.0]
        intents_to_consider = positive_intents if positive_intents else scored_intents

        for _, it_name, desc in intents_to_consider[:max_intents]:
            selected_criteria[it_name] = desc
            if it_name in handlers_by_name:
                supp, _ = get_handler_slot_info(handlers_by_name[it_name])
                candidate_supported_slots.update(supp)

        if not candidate_supported_slots:
            candidate_supported_slots = {"name", "area", "domain", "floor"}

        # Always include fallback option for escalation
        selected_criteria["other"] = "none of the other options fits"
        return selected_criteria, candidate_supported_slots

    def _rank_areas(
        self,
        context: StrategyContext,
        query_tokens: set[str],
        full_query: str,
    ) -> dict[str, str]:
        """Rank candidate areas based on utterance tokens."""
        scored_areas: list[tuple[float, str, str]] = []
        for area in context.area_registry.areas.values():
            name = area.name or area.id
            score = _lexical_score(query_tokens, name, full_query)
            scored_areas.append((score, area.id, name))

        scored_areas.sort(key=lambda item: item[0], reverse=True)
        return {
            area_id: name
            for _, area_id, name in scored_areas[:MAX_OPTIONS_PER_QUESTION]
        }

    def _rank_entities(
        self,
        context: StrategyContext,
        query_tokens: set[str],
        full_query: str,
        active_domains: set[str],
        top_area_names: set[str],
    ) -> dict[str, str]:
        """Rank candidate entities based on lexical matching and area/domain context."""
        scored_entities: list[tuple[float, str, str]] = []

        for state in context.states:
            domain = state.entity_id.split(".")[0]
            if domain not in CONTROLLABLE_DOMAINS:
                continue

            name = state.attributes.get("friendly_name", state.entity_id)
            name_score = _lexical_score(query_tokens, name, full_query)
            id_score = _lexical_score(query_tokens, state.entity_id, full_query)
            score = max(name_score, id_score)

            # Boost if entity's domain matches tokens in query
            if domain in query_tokens:
                score += 0.5

            # Boost if entity's area matches a top matched area
            entity_entry = (
                context.entity_registry.async_get(state.entity_id)
                if context.entity_registry
                else None
            )
            area_id = entity_entry.area_id if entity_entry else None
            if area_id and area_id.lower() in {a.lower() for a in top_area_names}:
                score += 1.0

            scored_entities.append((score, state.entity_id, name))

        scored_entities.sort(key=lambda item: item[0], reverse=True)
        return {
            ent_id: name
            for _, ent_id, name in scored_entities[:MAX_OPTIONS_PER_QUESTION]
        }

    def _build_questions(
        self, text: str, context: StrategyContext
    ) -> tuple[dict[str, Question], dict[str, str], dict[str, str]]:
        """Construct upstream and speculative question schemas dynamically within cardinality limits."""
        query_tokens = _tokenize(text)

        # Discover active controllable domains present in state
        active_domains = {
            state.entity_id.split(".")[0]
            for state in context.states
            if state.entity_id.split(".")[0] in CONTROLLABLE_DOMAINS
        }

        # Dynamic Intent Criteria & Supported Slots (from Home Assistant intent registry)
        intent_criteria, candidate_supported_slots = self._discover_intents(
            context, query_tokens, text, active_domains
        )

        # Dynamic Candidate Areas
        area_criteria = self._rank_areas(context, query_tokens, text)

        # Dynamic Candidate Entities
        matched_area_names = set(area_criteria.values())
        entity_criteria = self._rank_entities(
            context, query_tokens, text, active_domains, matched_area_names
        )

        # Dynamic Candidate Domains
        domain_criteria: dict[str, str] = {}
        # Rank active domains by keyword match with query
        scored_domains = [
            (1.0 if d in query_tokens else 0.0, d) for d in active_domains
        ]
        scored_domains.sort(key=lambda item: item[0], reverse=True)
        for _, d in scored_domains[:MAX_OPTIONS_PER_QUESTION]:
            domain_criteria[d] = f"{d.replace('_', ' ').capitalize()} devices"

        if not domain_criteria:
            domain_criteria = {
                "light": "Lighting devices and lamps",
                "switch": "Switches and power outlets",
            }

        questions: dict[str, Question] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,
            ),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ),
        }

        # Dynamically determine target scope options supported by candidate intents
        target_type_criteria: dict[str, str] = {}
        if "name" in candidate_supported_slots:
            target_type_criteria["entity"] = "A specific individual device or appliance"
        if "area" in candidate_supported_slots:
            target_type_criteria["area"] = "An entire room or area"
        if "domain" in candidate_supported_slots:
            target_type_criteria["domain_all"] = (
                "All devices of a domain across the home"
            )

        if len(target_type_criteria) > 1:
            questions["target_type"] = ChoiceQuestion(
                instructions="What scope is targeted?",
                criteria=target_type_criteria,
            )

        if "domain" in candidate_supported_slots and domain_criteria:
            questions["target_domain"] = ChoiceQuestion(
                instructions="What device domain is targeted?",
                criteria=domain_criteria,
            )

        if "area" in candidate_supported_slots and area_criteria:
            questions["target_area"] = ChoiceQuestion(
                instructions="Which area is mentioned?",
                criteria=area_criteria,
            )

        if "name" in candidate_supported_slots and entity_criteria:
            questions["target_entity"] = ChoiceQuestion(
                instructions="Which specific device is targeted?",
                criteria=entity_criteria,
            )

        if "light" in active_domains:
            questions["light_action"] = ChoiceQuestion(
                instructions="Action for lights",
                criteria={
                    "turn_off": "Turn lights off",
                    "turn_on": "Turn lights on",
                    "dim": "Dim or brighten lights",
                },
            )

        return questions, area_criteria, entity_criteria

    async def async_decide(
        self,
        engine: DecisionEngine,
        text: str,
        context: StrategyContext,
    ) -> Decision:
        """Execute speculative fan-out forward pass and route with code."""
        questions, _, _ = self._build_questions(text, context)

        state = {
            "user_query": text,
            "home": context.home_name,
        }

        result = await engine.async_predict(state, questions)
        answers = result.answers

        # Upstream compound check
        is_compound_noul = 0.0
        if "is_compound" in answers:
            compound_answer = answers["is_compound"]
            if isinstance(compound_answer, NoulAnswer):
                is_compound_noul = compound_answer.noul

        is_compound = is_compound_noul > self._compound_threshold
        if is_compound:
            return Decision(
                intent_name=None,
                confidence=is_compound_noul,
                is_compound=True,
                should_escalate=True,
                escalation_reason="Compound command detected",
                active_keys={"is_compound"},
                raw_answers=answers,
            )

        intent_ans = answers.get("intent")
        intent_name = (
            intent_ans.choice if isinstance(intent_ans, ChoiceAnswer) else "other"
        )
        intent_conf = (
            intent_ans.confidence if isinstance(intent_ans, ChoiceAnswer) else 0.0
        )
        top_prob = (
            intent_ans.probabilities.get(intent_name, intent_conf)
            if isinstance(intent_ans, ChoiceAnswer) and intent_ans.probabilities
            else intent_conf
        )

        if intent_name in ("other", "none") or top_prob < self._confidence_threshold:
            return Decision(
                intent_name=None,
                confidence=top_prob,
                is_compound=False,
                should_escalate=True,
                escalation_reason="Unhandled intent or low confidence",
                active_keys={"intent"},
                raw_answers=answers,
            )

        target_type_ans = answers.get("target_type")
        if isinstance(target_type_ans, ChoiceAnswer):
            target_type = target_type_ans.choice
        elif "target_entity" in answers:
            target_type = "entity"
        elif "target_area" in answers:
            target_type = "area"
        else:
            target_type = "entity"

        domain_ans = answers.get("target_domain")
        domain = domain_ans.choice if isinstance(domain_ans, ChoiceAnswer) else "light"

        active_keys = {"intent", "is_compound"}
        if "target_type" in answers:
            active_keys.add("target_type")
        if "target_domain" in answers:
            active_keys.add("target_domain")

        slots: dict[str, Any] = {}

        if target_type == "area" and "target_area" in answers:
            area_ans = answers["target_area"]
            if isinstance(area_ans, ChoiceAnswer):
                active_keys.add("target_area")
                slots["area"] = area_ans.choice
                slots["domain"] = domain
        elif target_type == "entity" and "target_entity" in answers:
            entity_ans = answers["target_entity"]
            if isinstance(entity_ans, ChoiceAnswer):
                active_keys.add("target_entity")
                slots["entity_id"] = entity_ans.choice
        elif target_type == "domain_all":
            slots["domain"] = domain

        # Slot filling heuristics for domains
        if domain == "light":
            if "light_action" in answers:
                active_keys.add("light_action")
            if bright_match := re.search(r"(\d+)\s*%", text):
                try:
                    slots["brightness"] = int(bright_match.group(1))
                except ValueError:
                    pass
        elif domain == "climate":
            if temp_match := re.search(
                r"(?:to|at)?\s*(\d+(?:\.\d+)?)\s*(?:degrees|deg|°|[cf]\b|celsius|fahrenheit)?",
                text,
                re.IGNORECASE,
            ):
                try:
                    slots["temperature"] = float(temp_match.group(1))
                except ValueError:
                    pass

        return Decision(
            intent_name=intent_name,
            slots=slots,
            confidence=top_prob,
            is_compound=False,
            should_escalate=False,
            active_keys=active_keys,
            raw_answers=answers,
        )
