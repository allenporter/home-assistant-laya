"""Laya-specific evaluation fixtures.

These fixtures provide client and decision engine adapters tailored to
the local Laya System One model running on CPU.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os

from huggingface_hub import snapshot_download
import huggingface_hub.constants as hf_constants
import pytest

from custom_components.laya.engine import LocalLayaEngine
from custom_components.laya.speculative import (
    DecisionEngine,
    DecisionStrategy,
    SpeculativeFanOutStrategy,
)
from custom_components.laya.speculative.inmemory.engine import FakeDecisionEngine


@pytest.fixture(scope="session", name="require_laya_model")
def require_laya_model_fixture() -> None:
    """Ensure Laya model weights are cached locally for live tests.

    Fails the test suite loudly with pytest.fail if the model is not found in the cache.
    """
    hf_constants.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        snapshot_download("convaiinnovations/laya", local_files_only=True)
    except Exception as err:
        pytest.fail(
            f"Laya model weights not found in local cache ({err}). "
            "Run './script/download-model' first."
        )


@pytest.fixture(name="laya_strategy")
def laya_strategy_fixture() -> SpeculativeFanOutStrategy:
    """Fixture providing a default SpeculativeFanOutStrategy for Laya."""
    return SpeculativeFanOutStrategy()


@pytest.fixture(name="strategy")
def strategy_alias_fixture(
    laya_strategy: SpeculativeFanOutStrategy,
) -> DecisionStrategy:
    """Alias for laya_strategy fixture."""
    return laya_strategy


@pytest.fixture(name="mock_laya_engine")
def mock_laya_engine_fixture() -> DecisionEngine:
    """Fixture providing a mock DecisionEngine for strategy testing."""
    return FakeDecisionEngine()


@pytest.fixture(scope="module", name="live_laya_engine")
async def live_laya_engine_fixture(
    require_laya_model: None,
) -> AsyncGenerator[LocalLayaEngine, None]:
    """Provide a warm LocalLayaEngine instance across test cases in the module.

    Depends on require_laya_model to ensure weights are present before loading.
    """
    engine = LocalLayaEngine(device="cpu", idle_timeout=None)
    try:
        await engine.async_load()
        yield engine
    finally:
        await engine.async_unload(force=True)


@pytest.fixture(name="live_engine")
def live_engine_alias_fixture(
    live_laya_engine: LocalLayaEngine,
) -> LocalLayaEngine:
    """Alias for live_laya_engine fixture."""
    return live_laya_engine
