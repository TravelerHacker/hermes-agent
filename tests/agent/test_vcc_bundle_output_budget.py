"""CPU proofs against captured native VccEngine (danbiocchi/hermes-vcc-engine @ ddc3a1d)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BUNDLE_PLUGIN = Path("/opt/cursor/artifacts/source-bundle/plugin")
if _BUNDLE_PLUGIN.is_dir():
    sys.path.insert(0, str(_BUNDLE_PLUGIN))

pytestmark = pytest.mark.skipif(
    not (_BUNDLE_PLUGIN / "vcc" / "engine.py").is_file(),
    reason="native VCC bundle not extracted at /opt/cursor/artifacts/source-bundle",
)

_CFG = {
    "model": {"context_length": 1_000_000},
    "providers": {
        "tensorfold": {
            "api": "http://127.0.0.1:9/v1",
            "api_key": "k",
            "output_token_budget": 150_000,
        },
    },
    "compression": {"threshold": 0.5},
    "context": {"engine": "vcc"},
}


@pytest.mark.parametrize("budget", [100_000, 150_000])
def test_native_vcc_engine_update_model_reserves_output(budget: int):
    from vcc.engine import VccEngine

    engine = VccEngine()
    engine.update_model(
        "deepseek-r1",
        1_000_000,
        base_url="http://127.0.0.1:9/v1",
        provider="custom",
        max_tokens=budget,
    )
    assert engine.max_tokens == budget
    assert engine.threshold_tokens == int((1_000_000 - budget) * engine.threshold_percent)


@pytest.mark.parametrize("budget", [100_000, 150_000])
def test_native_vcc_via_host_init_aligns_wire_and_reserve(budget: int):
    from vcc.engine import VccEngine

    cfg = {**_CFG, "providers": {**_CFG["providers"]}}
    cfg["providers"]["tensorfold"]["output_token_budget"] = budget
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.agent_init._select_context_engine", return_value=VccEngine()),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="deepseek-r1",
            api_key="k",
            base_url="http://127.0.0.1:9/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.max_tokens == budget
    assert agent.context_compressor.max_tokens == budget
    kw = agent._build_api_kwargs([{"role": "user", "content": "x"}], tools_for_api=[])
    assert kw.get("max_tokens") == budget


def test_misaligned_engine_raises_not_silent_drop():
    class _BareEngine:
        name = "bare"

        def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
            pass

        def bind_session_state(self, **kwargs):
            pass

    cfg = dict(_CFG)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.agent_init._select_context_engine", return_value=_BareEngine()),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from hermes_cli.output_token_budget import OutputTokenBudgetAlignmentError
        from run_agent import AIAgent

        with pytest.raises(OutputTokenBudgetAlignmentError):
            AIAgent(
                model="deepseek-r1",
                api_key="k",
                base_url="http://127.0.0.1:9/v1",
                provider="custom",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
