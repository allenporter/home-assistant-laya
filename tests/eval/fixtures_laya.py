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

from custom_components.laya.const import (
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from custom_components.laya.engine import LocalLayaEngine
from custom_components.laya.speculative.flow import (
    DecisionFlow,
    FlowConfig,
    create_decision_flow,
)
from custom_components.laya.speculative.scoring.engine import DecisionEngine
from custom_components.laya.speculative.testing.engine import FakeDecisionEngine


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


@pytest.fixture(name="flow")
def flow_fixture() -> DecisionFlow:
    """Fixture providing a default DecisionFlow for Laya."""
    return create_decision_flow(
        FlowConfig(
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            compound_threshold=DEFAULT_COMPOUND_THRESHOLD,
        )
    )


@pytest.fixture(name="flow_standard")
def flow_standard_fixture() -> DecisionFlow:
    """Fixture providing an unpruned DecisionFlow."""
    return create_decision_flow(
        FlowConfig(
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            compound_threshold=DEFAULT_COMPOUND_THRESHOLD,
            domain_filter_mode="none",
        )
    )


@pytest.fixture(name="flow_pruned")
def flow_pruned_fixture() -> DecisionFlow:
    """Fixture providing an IntentPruned DecisionFlow."""
    return create_decision_flow(
        FlowConfig(
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            compound_threshold=DEFAULT_COMPOUND_THRESHOLD,
            domain_filter_mode="strict",
        )
    )


@pytest.fixture(name="flow_boosted")
def flow_boosted_fixture() -> DecisionFlow:
    """Fixture providing a DomainBoosted DecisionFlow."""
    return create_decision_flow(
        FlowConfig(
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            compound_threshold=DEFAULT_COMPOUND_THRESHOLD,
            domain_filter_mode="boost",
        )
    )


@pytest.fixture(name="mock_laya_engine")
def mock_laya_engine_fixture() -> DecisionEngine:
    """Fixture providing a mock DecisionEngine for flow testing."""
    return FakeDecisionEngine()


@pytest.fixture(scope="module", name="live_engine")
async def live_engine_fixture(
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
