"""CPU-only wire/compaction contracts for custom OpenAI-compat routes at 1M context.

Behavior review for DeepSeek/TensorFold-style trials: no network, no model load.
Asserts transport assembly and ContextCompressor reservation math only.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.transports.chat_completions import ChatCompletionsTransport


CFG_MINIMAL = {"agent": {}, "compression": {"threshold": 0.5}, "context": {"engine": "compressor"}}


def _custom_agent(*, max_tokens, context_length=1_000_000, config=None):
    cfg = config or {**CFG_MINIMAL, "model": {"context_length": context_length}}
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=context_length),
        patch.object(ContextCompressor, "get_model_context_length", return_value=context_length, create=True),
    ):
        from run_agent import AIAgent

        return AIAgent(
            model="deepseek-r1",
            api_key="fixture-key-0123456789",
            base_url="http://127.0.0.1:9/v1",
            provider="custom",
            max_tokens=max_tokens,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def _wire_kwargs(agent, messages=None):
    msgs = messages or [{"role": "user", "content": "probe"}]
    return agent._build_api_kwargs(msgs, tools_for_api=[])


class TestRuntimeIgnoresLegacyUserOutputCaps:
    def test_gateway_runtime_omits_model_and_env_caps(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_MAX_TOKENS", "13")
        config = {
            "model": {"default": "fixture", "provider": "local-fixture", "max_tokens": 17, "context_length": 1_000_000},
            "providers": {
                "local-fixture": {"api": "http://127.0.0.1:1/v1", "api_key": "fixture", "max_output_tokens": 19},
            },
        }
        (tmp_path / "config.yaml").write_text(json.dumps(config))
        from gateway.run import _resolve_runtime_agent_kwargs

        runtime = _resolve_runtime_agent_kwargs()
        assert runtime.get("max_tokens") is None


class TestCompressorReservation1M:
    @pytest.mark.parametrize("reserve,expected_trigger", [(150_000, 425_000), (100_000, 450_000)])
    def test_threshold_uses_context_minus_output_reserve(self, reserve, expected_trigger):
        agent = _custom_agent(max_tokens=reserve, context_length=1_000_000)
        cc = agent.context_compressor
        assert cc.context_length == 1_000_000
        assert cc.max_tokens == reserve
        assert cc.threshold_tokens == expected_trigger

    def test_unset_agent_max_tokens_assumes_no_output_reserve(self):
        agent = _custom_agent(max_tokens=None, context_length=1_000_000)
        assert agent.context_compressor.max_tokens is None
        assert agent.context_compressor.threshold_tokens == 500_000


class TestCustomRouteWireAssembly:
    def test_explicit_agent_max_tokens_on_wire_as_max_tokens(self):
        agent = _custom_agent(max_tokens=150_000)
        kw = _wire_kwargs(agent)
        assert kw.get("max_tokens") == 150_000
        assert "max_completion_tokens" not in kw

    def test_no_agent_max_tokens_omits_openai_compat_cap(self):
        agent = _custom_agent(max_tokens=None)
        kw = _wire_kwargs(agent)
        assert "max_tokens" not in kw
        assert "max_completion_tokens" not in kw

    def test_request_overrides_can_set_wire_cap_without_agent_field(self):
        agent = _custom_agent(max_tokens=None)
        agent.request_overrides = {"max_tokens": 150_000}
        kw = _wire_kwargs(agent)
        assert kw.get("max_tokens") == 150_000

    def test_transport_direct_build_matches_agent_path(self):
        agent = _custom_agent(max_tokens=100_000)
        direct = ChatCompletionsTransport().build_kwargs(
            agent.model,
            [{"role": "user", "content": "x"}],
            max_tokens=agent.max_tokens,
            max_tokens_param_fn=agent._max_tokens_param,
            provider_profile=None,
            base_url=agent.base_url,
            is_custom_provider=True,
        )
        wire = _wire_kwargs(agent)
        assert direct.get("max_tokens") == wire.get("max_tokens") == 100_000


class TestContextLengthPinPrecedence:
    def test_model_context_length_pins_compressor_window(self):
        agent = _custom_agent(max_tokens=150_000, context_length=1_000_000)
        assert agent._config_context_length == 1_000_000
        assert agent.context_compressor.context_length == 1_000_000
