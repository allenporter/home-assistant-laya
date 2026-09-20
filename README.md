# home-assistant-laya

A high-performance, local System 1 conversation agent for [Home Assistant Assist](https://www.home-assistant.io/voice_control/) powered by the [Laya](https://huggingface.co/collections/laya) open-weight decision engine.

## Overview

Traditional large language models (System 2) are expressive and versatile, but introduce high inference latency (1–5+ seconds), variable response times, and heavy memory requirements that can strain local home automation servers.

**Laya** is a multilingual, non-autoregressive decision model designed specifically for low-latency decision routing (~200ms on CPU, faster on GPU/MPS). Instead of generating tokens sequentially, it evaluates structured decisions across multiple typed questions in a single forward pass:

- **Choice**: Categorical routing (e.g., selecting intent, domain, target entity, or area).
- **Noul**: Calibrated probability estimation (e.g., detecting if a request is compound or ambiguous).
- **Score**: Ordinal level evaluation along defined criteria rubrics.

### Architecture

```text
                             User Utterance
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   LayaConversationEntity  │
                     │   (Home Assistant Assist) │
                     └─────────────┬─────────────┘
                                   │
             assist-exposed entities│ (async_should_expose)
                                   ▼
                     ┌───────────────────────────┐
                     │  SpeculativeFanOutStrategy │
                     └─────────────┬─────────────┘
                                   │
           batch typed questions & │ (Single Forward Pass)
           current home state      ▼
                     ┌───────────────────────────┐
                     │      LocalLayaEngine      │
                     │ (In-process PyTorch model)│
                     └─────────────┬─────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
  High Confidence Match                        Compound or Ambiguous
  (Intent & Slots)                             (P(compound) > threshold)
            │                                             │
            ▼                                             ▼
  Home Assistant Intent                    System 2 Fallback Agent
  (intent.async_handle)                    (e.g., Local LLM or Cloud)
```

### Speculative Fan-Out Strategy

1. **Assist-Exposed Filtering**: Respects user exposure settings by querying `async_should_expose(hass, conversation.DOMAIN, entity_id)` so private or unexposed devices are never selected.
2. **Dynamic Domain Discovery**: Inspects state machine entities to dynamically configure selectable domains (such as `light`, `switch`, `climate`, `fan`, `cover`, `vacuum`).
3. **Deterministic Slot Extraction**: Fast regex-based slot extraction extracts numbers for commands like brightness (_"set to 50%"_ $\to$ `50`) and temperature (_"set to 65 degrees"_ $\to$ `65.0`).
4. **Cardinality Budgeting**: Caps candidate areas and entities to at most 20 options per question (`MAX_OPTIONS_PER_QUESTION = 20`) to remain within the model's optimal accuracy bounds.
5. **Hybrid Escalation**: Classifies compound utterances (e.g. _"turn off the patio light and start the dishwasher"_). When detected, requests gracefully escalate to a configured fallback conversation agent.

---

## Configuration & Low-Level Data

When configured via the Home Assistant UI or created programmatically, the integration accepts the following configuration fields:

| Field                  | Location           | Type    | Default  | Description                                                                                                |
| :--------------------- | :----------------- | :------ | :------- | :--------------------------------------------------------------------------------------------------------- |
| `device`               | `data`             | `str`   | `"auto"` | Execution device: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`.                                                 |
| `idle_timeout`         | `data` / `options` | `float` | `0.0`    | Seconds to retain weights in RAM after an inference before unloading. `0.0` unloads immediately when idle. |
| `compound_threshold`   | `options`          | `float` | `0.65`   | Probability threshold above which an utterance is flagged as compound and escalated.                       |
| `confidence_threshold` | `options`          | `float` | `0.50`   | Minimum confidence score required to dispatch an intent action.                                            |
| `fallback_agent`       | `options`          | `str`   | `None`   | Entity ID or agent ID of the System 2 fallback agent (e.g. `"conversation.home_assistant"`).               |

---

## Recommended Settings for Evaluation & Benchmarking

### Why `idle_timeout` Matters

- **Production HA Servers (`idle_timeout = 0.0`)**: By default, `idle_timeout` is set to `0.0`. Once all concurrent requests finish, the model weights (~1.2 GB) are unloaded from memory immediately. This keeps memory free on standard smart home servers when Assist is idle.
- **Evaluation & Benchmarks (`idle_timeout > 0`)**: During evaluations or automated benchmarks, reloading model weights for every test sentence incurs a 20–30 second disk/CPU initialization penalty per sentence. Setting `idle_timeout` to `30.0` or `60.0`:
  1. **Eagerly preloads** model weights into memory during integration setup (`async_setup_entry`).
  2. **Keeps weights resident in RAM** between consecutive utterances as long as utterances arrive within the timeout window.

### Programmatic Setup in Tests & Benchmark Scripts

```python
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.laya.const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DEVICE,
    CONF_FALLBACK_AGENT,
    CONF_IDLE_TIMEOUT,
    DOMAIN,
)


async def setup_laya_for_eval(hass: HomeAssistant) -> MockConfigEntry:
    """Create and set up a warm Laya entry for evaluation."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Laya (Benchmark)",
        data={
            CONF_DEVICE: "cuda",      # or "mps" / "cpu"
            CONF_IDLE_TIMEOUT: 60.0,  # Eager preload + keep weights warm
        },
        options={
            CONF_IDLE_TIMEOUT: 60.0,
            CONF_COMPOUND_THRESHOLD: 0.65,
            CONF_CONFIDENCE_THRESHOLD: 0.50,
            CONF_FALLBACK_AGENT: "conversation.home_assistant",
        },
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry
```

---

## Development & Testing

### Environment Setup

```bash
$ script/setup
```

### Running Tests

```bash
$ script/test
```

### Running Linters

```bash
$ script/lint
```
