"""Speculative Fan-Out decision strategy (naive prototype placeholder).

WARNING: This is a placeholder / prototype strategy demonstrating the forward pass
and Python-code routing pattern. It contains known simplifications, silly shortcuts,
and prototype heuristics (such as hardcoded domains, static question schemas, arbitrary
entity truncation without ranking, and basic regex slot extraction).

A production strategy should dynamically synthesize question graphs based on registered
Home Assistant intents, rank candidate entities (via BM25 / vector search / floor scoping),
and use dedicated slot-filling heads.
"""

from __future__ import annotations

import logging
import re
from typing import Any

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
    ScoreAnswer,
    ScoreQuestion,
)
from .base import Decision, DecisionStrategy, StrategyContext

_LOGGER = logging.getLogger(__name__)

# Prototype hardcoded set of controllable domains in Home Assistant.
# TODO: Improve this! Instead of a hardcoded set of domains, dynamically inspect
# registered intents and determine candidate domains/actions based on available entities
# and device capabilities.
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


class SpeculativeFanOutStrategy(DecisionStrategy):
    """Naive prototype speculative fan-out strategy."""

    def __init__(
        self,
        compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize SpeculativeFanOutStrategy."""
        self._compound_threshold = compound_threshold
        self._confidence_threshold = confidence_threshold

    def _build_questions(
        self, context: StrategyContext
    ) -> tuple[dict[str, Question], dict[str, str], dict[str, str]]:
        """Construct upstream and speculative question schemas dynamically within cardinality limits."""
        # Discover active controllable domains and entities dynamically from state
        active_domains: set[str] = set()
        entity_criteria: dict[str, str] = {}

        for state in context.states:
            domain = state.entity_id.split(".")[0]
            if domain in CONTROLLABLE_DOMAINS:
                active_domains.add(domain)
                # TODO: This is a dumb prototype shortcut that arbitrarily drops entities
                # beyond MAX_OPTIONS_PER_QUESTION without semantic ranking or room-scoping.
                # A proper implementation should rank entities (e.g. by room context, recency,
                # or retrieval score) rather than arbitrarily truncating the list.
                if len(entity_criteria) < MAX_OPTIONS_PER_QUESTION:
                    name = state.attributes.get("friendly_name", state.entity_id)
                    entity_criteria[state.entity_id] = name

        # Build target_domain criteria dynamically
        domain_criteria: dict[str, str] = {}
        for d in sorted(active_domains)[:MAX_OPTIONS_PER_QUESTION]:
            domain_criteria[d] = f"{d.replace('_', ' ').capitalize()} devices"

        if not domain_criteria:
            domain_criteria = {
                "light": "Lighting devices and lamps",
                "switch": "Switches and power outlets",
            }

        # Build area criteria
        area_criteria: dict[str, str] = {}
        for area in list(context.area_registry.areas.values())[
            :MAX_OPTIONS_PER_QUESTION
        ]:
            area_criteria[area.id] = area.name or area.id

        # TODO: This fixed set of speculative questions is a prototype hack.
        # In a real strategy, this question graph should be dynamically synthesized
        # from registered intent schemas and available device features rather than
        # using a static dictionary of questions.
        questions: dict[str, Question] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria={
                    "HassTurnOn": "Turn on or activate device, light, or appliance",
                    "HassTurnOff": "Turn off or stop device, light, or appliance",
                    "HassClimateSetTemperature": "Adjust thermostat temperature",
                    "none": "Other, query status, or general conversational query",
                },
            ),
            "target_type": ChoiceQuestion(
                instructions="What scope is targeted?",
                criteria={
                    "area": "An entire room or area",
                    "entity": "A specific individual device or appliance",
                    "domain_all": "All devices of a domain across the home",
                },
            ),
            "target_domain": ChoiceQuestion(
                instructions="What device domain is targeted?",
                criteria=domain_criteria,
            ),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ),
            "light_action": ChoiceQuestion(
                instructions="Action for lights",
                criteria={
                    "turn_off": "Turn lights off",
                    "turn_on": "Turn lights on",
                    "dim": "Dim or brighten lights",
                },
            ),
            "target_temperature": ScoreQuestion(
                instructions="What is the desired temperature setting?",
                criteria=[
                    "Cool / low (18°C or below)",
                    "Moderate (19°C - 21°C)",
                    "Warm (22°C - 24°C)",
                    "Hot (25°C or above)",
                ],
            ),
        }

        if area_criteria:
            questions["target_area"] = ChoiceQuestion(
                instructions="Which area is mentioned?",
                criteria=area_criteria,
            )

        if entity_criteria:
            questions["target_entity"] = ChoiceQuestion(
                instructions="Which specific device is targeted?",
                criteria=entity_criteria,
            )

        return questions, area_criteria, entity_criteria

    async def async_decide(
        self,
        engine: DecisionEngine,
        text: str,
        context: StrategyContext,
    ) -> Decision:
        """Execute speculative fan-out forward pass and route with code."""
        questions, _, _ = self._build_questions(context)

        state = {
            "user_query": text,
            "home": context.home_name,
        }

        result = await engine.async_predict(state, questions)
        answers = result.answers

        # Upstream evaluation
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
            intent_ans.choice if isinstance(intent_ans, ChoiceAnswer) else "none"
        )
        intent_conf = (
            intent_ans.confidence if isinstance(intent_ans, ChoiceAnswer) else 0.0
        )

        if intent_name == "none" or intent_conf < self._confidence_threshold:
            return Decision(
                intent_name=None,
                confidence=intent_conf,
                is_compound=False,
                should_escalate=True,
                escalation_reason="Unhandled intent or low confidence",
                active_keys={"intent"},
                raw_answers=answers,
            )

        target_type_ans = answers.get("target_type")
        target_type = (
            target_type_ans.choice
            if isinstance(target_type_ans, ChoiceAnswer)
            else "entity"
        )

        domain_ans = answers.get("target_domain")
        domain = domain_ans.choice if isinstance(domain_ans, ChoiceAnswer) else "light"

        active_keys = {"intent", "target_type", "target_domain", "is_compound"}
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

        if domain == "light" and "light_action" in answers:
            active_keys.add("light_action")
            # Prototype heuristic: fast regex extraction as a pragmatic supplement
            # TODO: Replace with generalized entity/slot extraction models or heads
            if bright_match := re.search(r"(\d+)\s*%", text):
                try:
                    slots["brightness"] = int(bright_match.group(1))
                except ValueError:
                    pass
        elif domain == "climate":
            if "target_temperature" in answers:
                temp_ans = answers["target_temperature"]
                if isinstance(temp_ans, ScoreAnswer):
                    active_keys.add("target_temperature")
                    slots["temperature_level"] = temp_ans.score

            # Prototype heuristic: fast regex extraction as a pragmatic supplement
            # TODO: Replace with generalized entity/slot extraction models or heads
            if temp_match := re.search(
                r"(?:to|at)?\s*(\d+(?:\.\d+)?)\s*(?:degrees|deg|°)?",
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
            confidence=intent_conf,
            is_compound=False,
            should_escalate=False,
            active_keys=active_keys,
            raw_answers=answers,
        )
