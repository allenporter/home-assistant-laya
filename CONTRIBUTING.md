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

_Note: If the model weights have not been downloaded yet, this test will skip with instructions to run `script/download-model`._

## Running Linters & Formatters

To run all code formatting, linting, and type checking tools:

```bash
$ script/lint
```

This checks formatting with `ruff`, type consistency with `ty`, spelling with `codespell`, and documentation syntax with `prettier`.
