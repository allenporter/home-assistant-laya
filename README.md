# home-assistant-laya

A fast, local System 1 conversation agent for [Home Assistant Assist](https://www.home-assistant.io/voice_control/) powered by the [Laya](https://huggingface.co/collections/laya) open-weight decision engine.

## Why Laya?

When you ask your smart home to _"turn on the kitchen lights"_, you want it to happen immediately—not after waiting 3 seconds for a local LLM to finish thinking.

Large language models (System 2) are fantastic for answering trivia or drafting emails, but they are often overkill for everyday home voice commands. Generating tokens word-by-word introduces latency, variable response times, and high memory usage.

**Laya** takes a different approach. It acts like your smart home's nervous system: a lightweight, non-autoregressive **System 1 decision model** that evaluates choices in a single forward pass in **~200ms on CPU** (and even faster on GPU or Apple Silicon MPS).

Instead of guessing tokens, Laya resolves structured questions simultaneously:

- **Choice**: Categorical decisions (e.g. _What's the intent? Which domain? Which room or device?_).
- **Noul**: Calibrated confidence ($P(\text{true}) \in [0.0, 1.0]$) to know when a command is compound or ambiguous.
- **Score**: Discrete level ratings along defined rubrics.

### How it Works

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
  (Intent & Target Slots)                      (P(compound) > threshold)
            │                                             │
            ▼                                             ▼
  Home Assistant Intent                    System 2 Fallback Agent
  (intent.async_handle)                    (e.g., Local LLM or Cloud)
```

### Speculative Fan-Out

Rather than asking questions one-by-one in a slow conversation tree, Laya uses **speculative fan-out** to evaluate everything at once:

1. **Respects Your Privacy**: It only ever considers devices and areas that you have explicitly enabled under _Expose to Assist_.
2. **Adapts to Your Home**: It dynamically detects active domains in your home state (such as `light`, `switch`, `climate`, `fan`, `cover`, `vacuum`) so you don't have to manually configure them.
3. **Instant Number Extraction**: Fast, deterministic regex extractors instantly pull out target percentages and temperatures (e.g. _"set thermostat to 68 degrees"_ $\to$ `68.0`, _"dim lights to 40%"_ $\to$ `40`).
4. **Keeps Choices Manageable**: Questions are capped at 20 options each (`MAX_OPTIONS_PER_QUESTION = 20`) to keep the decision space optimal and accurate.
5. **Knows When to Ask for Help**: If you ask something complex like _"turn off the porch light and lock the front door"_, Laya spots that it's a compound command and gracefully hands it off to your configured System 2 fallback agent.

---

## Configuration & Low-Level Settings

You can configure Laya directly in the Home Assistant UI or programmatically via config entry data:

| Field                  | Location           | Type    | Default  | Description                                                                              |
| :--------------------- | :----------------- | :------ | :------- | :--------------------------------------------------------------------------------------- |
| `device`               | `data`             | `str`   | `"auto"` | Hardware device: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`.                                |
| `idle_timeout`         | `data` / `options` | `float` | `0.0`    | Seconds to hold model weights in RAM after a query. `0.0` unloads immediately when idle. |
| `compound_threshold`   | `options`          | `float` | `0.65`   | Probability threshold where commands are flagged as compound and escalated.              |
| `confidence_threshold` | `options`          | `float` | `0.50`   | Minimum confidence score needed to execute an intent directly.                           |
| `fallback_agent`       | `options`          | `str`   | `None`   | Entity or agent ID for System 2 fallback (e.g. `"conversation.home_assistant"`).         |

---

## Running Evaluations & Benchmarks

### The RAM vs. Latency Trade-off (`idle_timeout`)

Model weights take about ~1.2 GB of RAM:

- **For everyday home use (`idle_timeout = 0.0`)**: If you run Home Assistant on a Raspberry Pi or shared server, you probably don't want 1.2 GB of memory tied up all day. By default, Laya unloads its weights immediately after answering your request, freeing up RAM.
- **For running evals and benchmarks (`idle_timeout > 0`)**: If you are benchmarking 100 test sentences in a script, you don't want to wait 20–30 seconds for the model to reload on every single sentence. Setting `idle_timeout: 60.0`:
  1. **Preloads the model** into memory during setup (`async_setup_entry`).
  2. **Keeps the model warm** between requests, letting you evaluate at full ~200ms speed.

### Programmatic Setup in Tests & Benchmark Scripts

Here is how you can spin up a warm, preloaded instance in your benchmark script:

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
            CONF_DEVICE: "cuda",  # or "mps" / "cpu"
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

## Contributing & Development

### Setup Environment

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
