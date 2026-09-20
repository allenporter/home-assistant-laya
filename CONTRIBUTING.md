# Contributing & Development

Thank you for your interest in contributing to `home-assistant-laya`!

## Setting Up Your Environment

This project uses `uv` for fast, reproducible Python environment management.

```bash
$ script/setup
```

This will set up your virtual environment, install all development dependencies, and install pre-commit hooks.

## Downloading Model Weights

To pre-download the Laya model weights into your local Hugging Face cache (e.g. for running live inference tests or offline development):

```bash
$ script/download-model
```

The weights (~1.2 GB) will be stored in `~/.cache/huggingface/hub/models--convaiinnovations--laya`.

## Running Tests

### Fast Unit Tests

Run the main unit test suite:

```bash
$ script/test
```

This runs all mock-based tests and completes in ~3 seconds.

### Live Model Inference Tests

To run the live inference test using the real Laya PyTorch model:

```bash
$ script/test -m slow
```

_Note: If the model weights have not been downloaded yet, this test will fail with instructions to run `script/download-model`._

## Running Linters & Formatters

To run all code formatting, linting, and type checking tools:

```bash
$ script/lint
```

This checks formatting with `ruff`, type consistency with `ty`, spelling with `codespell`, and documentation syntax with `prettier`.

## Benchmarks & Evaluation

When benchmarking large test sets, set `idle_timeout: 60.0` to keep model weights warm in memory across predictions:

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
