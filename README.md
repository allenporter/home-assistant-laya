# home-assistant-laya

A fast, local System 1 conversation agent for [Home Assistant Assist](https://www.home-assistant.io/voice_control/) powered by the [Laya](https://huggingface.co/collections/laya) open-weight decision engine.

## Why Laya?

When you ask your smart home to _"turn on the kitchen lights"_, you want it to happen immediately—not after waiting 3 seconds for a local LLM to generate tokens word-by-word.

Everyday smart home commands don't need text generation; they need **structured actions and entity targets**. Coercing a generative LLM into outputting tool calls creates a mismatch: you generate conversational prose just to parse it back into code.

**Laya** acts as your smart home's reflex: a lightweight, non-autoregressive **System 1 decision model** that evaluates structured choices in a single forward pass in **~200ms on CPU** (even faster on GPU or Apple Silicon MPS).

- **Instant Execution**: Routine commands (on/off, brightness, climate) execute locally with sub-second response times.
- **Speculative Fan-Out**: Evaluates intent, area, entity, and domain questions in parallel in one forward pass.
- **System 1 + System 2 Partnership**: Acts as a fast front door for Home Assistant. Routine commands execute immediately; compound requests or conversational chat are handed off to your configured fallback LLM.

---

## How It Works: Speculative Fan-Out

Rather than asking questions sequentially in a slow conversational cascade, Laya evaluates all potentially relevant questions upfront in a single batch.

For example, when a user says:

> _"Turn off the kitchen lights"_

Laya evaluates these questions simultaneously in a single forward pass:

1. **Primary Intent**: What action is requested? (`HassTurnOff`)
2. **Target Area**: Which room is mentioned? (`kitchen`)
3. **Target Device**: Which device is targeted? (`light.kitchen_light`)
4. **Light Action**: What adjustment to make? (`turn_off`)
5. **Is Compound**: Does this contain multiple distinct instructions? (`P(compound) = 0.05`)

Notice that the `light_action` question is asked _before_ the model even knows if the user is targeting a light. Because non-autoregressive models run questions in parallel, asking 5 questions takes the same ~200ms as asking 1. Python code—not the model—simply discards answers that turn out to be irrelevant.

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
           batch typed questions & │ (Single Forward Pass ~200ms)
           current home state      ▼
                     ┌───────────────────────────┐
                     │      LocalLayaEngine      │
                     │ (In-process PyTorch model)│
                     └─────────────┬─────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
  High Confidence Match                         Compound or Low Confidence
  (Intent & Target Slots)                       (P(compound) > thres or Conf < thres)
            │                                             │
            ▼                                             ▼
  Home Assistant Intent                         System 2 Fallback Agent
  (intent.async_handle)                         (e.g., Local LLM or Cloud)
```

### Key Decision Design Principles

- **Respects Assist Exposure**: Only considers devices and areas explicitly exposed under _Expose to Assist_.
- **Dynamic HA Intent Introspection**: Inspects Home Assistant's registered `IntentHandler`s and slot schemas to only prompt intents the strategy can fulfill.
- **Calibrated Confidence**: Choice confidence is calibrated across candidate set size $N$ using the dispersion formula:
  $$\text{confidence} = \max\left(0, \min\left(1, \frac{N \times p_{\max} - 1}{N - 1}\right)\right)$$
  Uniform uncertainty ($p_{\max} = 1/N$) maps to `0.0`, while complete certainty maps to `1.0`, ensuring candidates are not penalized by softmax dilution.
- **Closed-Set Routing**: Intent questions strictly evaluate registered Home Assistant intents without artificial fallback options. Unhandled or out-of-domain requests naturally produce diffuse probability distributions that drop below threshold and escalate.
- **Instant Numeric Extraction**: Deterministic regex extractors instantly pull out target percentages and temperatures (e.g., _"set thermostat to 68 degrees"_ $\to$ `68.0`).

---

## Configuration

Configure Laya in the Home Assistant UI or programmatically via config entry options:

| Field                  | Type    | Default  | Description                                                                                        |
| :--------------------- | :------ | :------- | :------------------------------------------------------------------------------------------------- |
| `device`               | `str`   | `"auto"` | Hardware device: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`.                                          |
| `idle_timeout`         | `float` | `0.0`    | Seconds to hold model weights in RAM after a query (`0.0` unloads immediately to free ~1.2GB RAM). |
| `confidence_threshold` | `float` | `0.30`   | Minimum calibrated confidence ($\frac{N \cdot p_{\max} - 1}{N - 1}$) required to execute directly. |
| `compound_threshold`   | `float` | `0.65`   | Probability threshold where requests are treated as multi-step commands and escalated.             |
| `fallback_agent`       | `str`   | `None`   | Conversation agent entity ID for System 2 fallback (e.g., `"conversation.home_assistant"`).        |

---

## Benchmarks & Evaluation

When benchmarking large test sets, set `idle_timeout: 60.0` to keep model weights warm in memory across predictions instead of reloading on each sentence:

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
            CONF_DEVICE: "auto",  # or "cuda" / "mps" / "cpu"
            CONF_IDLE_TIMEOUT: 60.0,  # Preload and keep weights warm
        },
        options={
            CONF_IDLE_TIMEOUT: 60.0,
            CONF_COMPOUND_THRESHOLD: 0.65,
            CONF_CONFIDENCE_THRESHOLD: 0.30,
            CONF_FALLBACK_AGENT: "conversation.home_assistant",
        },
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry
```

---

## Contributing

For instructions on setting up your development environment, downloading model weights, running the test suite, and linting, see [CONTRIBUTING.md](CONTRIBUTING.md).
