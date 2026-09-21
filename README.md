# Laya for Home Assistant

A fast, fully local conversation agent for [Home Assistant Assist](https://www.home-assistant.io/voice_control/) powered by the open-weight [Laya](https://huggingface.co/collections/laya) decision engine.

Laya is a lightweight, non-autoregressive decision model that evaluates structured choices in a single forward pass. It integrates directly as a native **Conversation Agent** in Home Assistant Assist voice pipelines—dispatching routine device control commands directly with zero cloud reliance, while seamlessly escalating compound or open-ended conversational queries to a configured fallback LLM.

Instead of generating free-form text token-by-token and parsing the output back into code, Laya scores the probability of registered Home Assistant intents and exposed entities directly in a single pass.

## Highlights

- **Native Assist Voice Agent**: Selectable as a first-class conversation agent in Assist pipelines (Voice PE, satellite microphones, dashboard Assist).
- **100% Local & Private**: Runs entirely in-process on local hardware (CPU, CUDA, or Apple Silicon). No external API keys, cloud subscriptions, or audio/device state leaving your local network.
- **Fast Single-Pass Execution**: Avoids the multi-second latency and overhead of autoregressive generative token generation.
- **Zero Hallucinated Targets**: Candidate targets are constrained to entities and areas currently exposed to Assist (`async_should_expose`).
- **Speculative Fan-Out**: Evaluates intent, area, entity, and compound questions simultaneously in a single forward pass.
- **Smart Escalation**: Accurately executes simple home automation commands and escalates ambiguous or compound requests to a generative LLM fallback.

## Performance & Hardware

Because Laya is non-autoregressive, questions are evaluated concurrently in a single forward pass rather than generating words sequentially. Inference latency depends on hardware architecture and batch size:

| Hardware                             | Questions Evaluated    | Reported Latency  | Notes                                                                  |
| :----------------------------------- | :--------------------- | :---------------- | :--------------------------------------------------------------------- |
| **GPU (NVIDIA T4 / CUDA)**           | 1 question             | ~33 ms – 40 ms    | [Model Card Benchmarks](https://huggingface.co/convaiinnovations/laya) |
| **GPU (NVIDIA T4 / CUDA)**           | 10 questions (batched) | ~72 ms – 159 ms   | Concurrent speculative fan-out                                         |
| **Apple Silicon (M-series MPS/MLX)** | 1–10 questions         | ~50 ms – 140 ms   | Accelerated unified memory inference                                   |
| **Standard CPU (x86_64 / ARM)**      | Multi-question batch   | ~200 ms – 500 ms+ | Scales with CPU core count & AVX/SIMD capabilities                     |

_Note: Initial model load incurs a warm-up phase. For production use, hardware acceleration (CUDA, Apple Silicon Metal/MPS) provides the most responsive voice assistant experience._

## How It Works

When a user speaks or types an utterance, Laya executes a modular 5-stage decision pipeline:

1. **Request Processing**: Normalizes text and tokenizes the user's utterance.
2. **Candidate Retrieval**: Introspects registered Home Assistant `IntentHandler`s and Assist-exposed entities/areas to build candidate target sets.
3. **Hydration**: Packages current entity state and candidate choices into structured decision questions.
4. **Scoring**: Evaluates all questions simultaneously via the local model in a single forward pass.
5. **Resolution**: Inspects calibrated confidence scores. If confidence exceeds the threshold, it dispatches the native Home Assistant intent; otherwise, it escalates to your configured fallback agent.

```text
                             User Utterance
                    ("Turn off the kitchen lights")
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   LayaConversationEntity  │
                     │  (Home Assistant Assist)  │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │       DecisionFlow        │
                     │  Request → Retrieval →    │
                     │  Hydration → Scoring →    │
                     │        Resolution         │
                     └─────────────┬─────────────┘
                                   │
                  Batch typed questions & home state
                        (Single forward pass)
                                   ▼
                     ┌───────────────────────────┐
                     │      LocalLayaEngine      │
                     │   (Local CPU / GPU ONNX)  │
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
