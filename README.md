# Laya for Home Assistant

A local conversation agent for [Home Assistant Assist](https://www.home-assistant.io/voice_control/) powered by the [Laya](https://huggingface.co/collections/laya) open-weight decision engine.

Laya is a lightweight, non-autoregressive decision model that evaluates structured choices in a single forward pass in ~200ms on a standard CPU. It is designed to act as a fast front-end for Home Assistant Assist, dispatching routine device control commands directly while escalating compound or conversational queries to a configured fallback LLM.

Instead of generating text token-by-token and parsing the output back into code, Laya selects the most probable structured action and entity targets from a predefined set of choices.

Features:

- **Local Execution**: Routine commands (on/off, brightness, climate) execute locally with typical CPU response times of ~200ms.
- **Speculative Fan-Out**: Evaluates intent, area, entity, and domain questions in parallel during a single forward pass.
- **Agent Escalation**: Handles routine commands directly and routes compound requests or conversational chat to a configured generative conversation agent fallback.

## Speculative Fan-Out

Laya evaluates all potentially relevant questions about the user's request upfront in a single batch.

For example, given the command _"Turn off the kitchen lights"_, Laya evaluates these questions simultaneously:

1. **Primary Intent**: What action is requested? (`HassTurnOff`)
2. **Target Area**: Which room is mentioned? (`kitchen`)
3. **Target Device**: Which device is targeted? (`light.kitchen_light`)
4. **Light Action**: What adjustment to make? (`turn_off`)
5. **Is Compound**: Does this contain multiple distinct instructions? (`P(compound) = 0.05`)

Because non-autoregressive models run questions in parallel, evaluating five questions takes the same ~200ms duration as evaluating one. Answers that turn out to be irrelevant are discarded.

```text
                             User Utterance
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   LayaConversationEntity  │
                     │   (Home Assistant Assist) │
                     └─────────────┬─────────────┘
                                   │
             Assist-exposed entities (async_should_expose)
                                   ▼
                     ┌───────────────────────────┐
                     │  SpeculativeFanOutStrategy│
                     └─────────────┬─────────────┘
                                   │
           Batch typed questions & current home state
                   (Single forward pass ~200ms)
                                   ▼
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
  Home Assistant Intent                         Fallback Generative Agent
  (intent.async_handle)                         (e.g., Local LLM or Cloud API)
```

## Assist Integration

- **Respects Assist Exposure**: Only considers devices and areas explicitly exposed under the Home Assistant _Expose to Assist_ settings.
- **Intent Introspection**: Inspects Home Assistant's registered `IntentHandler`s and slot schemas to only present intents that the model can fulfill.
- **Calibrated Confidence**: Choice confidence is normalized across candidate set size $N$ ($\frac{N \cdot p_{\max} - 1}{N - 1}$), scaling from 0.0 (uniform uncertainty) to 1.0 (certainty).
- **Closed-Set Routing**: Questions evaluate registered Home Assistant intents directly. Unsupported requests or conversational chatter produce low-confidence distributions that escalate to the fallback agent.
- **Numeric Extraction**: Deterministic regex extractors pull out target percentages and temperatures.

## Configuration

Configure Laya in the Home Assistant UI or programmatically via config entry options:

| Field                  | Type    | Default  | Description                                                                                        |
| :--------------------- | :------ | :------- | :------------------------------------------------------------------------------------------------- |
| `device`               | `str`   | `"auto"` | Hardware device: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`.                                          |
| `idle_timeout`         | `float` | `0.0`    | Seconds to hold model weights in RAM after a query (`0.0` unloads immediately to free ~1.2GB RAM). |
| `confidence_threshold` | `float` | `0.30`   | Minimum calibrated confidence ($\frac{N \cdot p_{\max} - 1}{N - 1}$) required to execute directly. |
| `compound_threshold`   | `float` | `0.65`   | Probability threshold where requests are treated as multi-step commands and escalated.             |
| `fallback_agent`       | `str`   | `None`   | Conversation agent entity ID for fallback (e.g., `"conversation.home_assistant"`).                 |

## Contributing

For instructions on setting up your development environment, downloading model weights, running tests, benchmarks, and linting, see [CONTRIBUTING.md](CONTRIBUTING.md).
