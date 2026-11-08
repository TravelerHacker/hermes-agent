"""Config → runtime → wire paths for custom providers (no network, no LLM).

Proves what YAML can and cannot reach on gateway/CLI resolution at b200f5bc.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.transports.chat_completions import ChatCompletionsTransport


TENSORFOLD = {
    "model": {
        "default": "deepseek-r1",
        "provider": "custom:tensorfold",
        "context_length": 1_000_000,
    },
    "providers": {
        "tensorfold": {
            "api": "http://127.0.0.1:9/v1",
            "api_key": "fixture-key",
            "model": "deepseek-r1",
            "extra_body": {"user": "proxy-user"},
            "request_overrides": {"max_tokens": 150_000},
        },
    },
    "compression": {"threshold": 0.5},
    "context": {"engine": "compressor"},
}


def test_provider_request_overrides_key_not_in_known_provider_keys(tmp_path, monkeypatch):
    """``providers.*.request_overrides`` is not lifted; only ``extra_body`` becomes runtime overrides."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(TENSORFOLD))
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="custom:tensorfold", target_model="deepseek-r1")
    ro = runtime.get("request_overrides") or {}
    assert ro == {"extra_body": {"user": "proxy-user"}}
    assert "max_tokens" not in ro
    assert runtime.get("max_tokens") is None


def test_gateway_runtime_forwards_extra_body_request_overrides_to_agent_wire(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = dict(TENSORFOLD)
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))

    from gateway.run import _resolve_runtime_agent_kwargs

    runtime = _resolve_runtime_agent_kwargs()
    assert "max_tokens" not in (runtime.get("request_overrides") or {})

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            provider=runtime["provider"],
            model="deepseek-r1",
            request_overrides=dict(runtime.get("request_overrides") or {}),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        kw = agent._build_api_kwargs([{"role": "user", "content": "probe"}], tools_for_api=[])
    assert "max_tokens" not in kw
    assert kw.get("extra_body") == {"user": "proxy-user"}


def test_runtime_request_overrides_max_tokens_reaches_wire_when_present(tmp_path, monkeypatch):
    """When runtime carries top-level ``request_overrides.max_tokens``, transport includes it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(TENSORFOLD))

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="deepseek-r1",
            api_key="fixture",
            base_url="http://127.0.0.1:9/v1",
            provider="custom",
            request_overrides={"max_tokens": 100_000},
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        kw = agent._build_api_kwargs([{"role": "user", "content": "x"}], tools_for_api=[])
    assert kw.get("max_tokens") == 100_000
    assert agent.context_compressor.max_tokens is None


class _RecordingVccLike(ContextCompressor):
    """Stand-in for hermes-vcc (ContextCompressor subclass) to capture host update_model wiring."""

    @property
    def name(self) -> str:
        return "hermes-vcc"

    def clone_for_agent(self):
        clone = _RecordingVccLike(model=self.model)
        clone.last_update_kwargs = getattr(self, "last_update_kwargs", None)
        return clone


def test_plugin_context_engine_receives_max_tokens_on_update_model():
    """Host passes output reserve into plugin ``update_model`` (ContextCompressor-shaped engines)."""
    cfg = {
        **TENSORFOLD,
        "context": {"engine": "hermes-vcc", "vcc": {"debug": False}},
    }
    engine = _RecordingVccLike(model="deepseek-r1")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.agent_init._select_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="deepseek-r1",
            api_key="fixture",
            base_url="http://127.0.0.1:9/v1",
            provider="custom",
            max_tokens=150_000,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.context_compressor is engine
    assert agent.max_tokens == 150_000
    assert engine.max_tokens == 150_000
    assert engine.threshold_tokens == 425_000


def test_provider_output_token_budget_reaches_wire_and_compressor(tmp_path, monkeypatch):
    cfg = dict(TENSORFOLD)
    cfg["providers"]["tensorfold"]["output_token_budget"] = 150_000
    cfg["providers"]["tensorfold"].pop("request_overrides", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="deepseek-r1",
            api_key="fixture",
            base_url="http://127.0.0.1:9/v1",
            provider="custom",
            requested_provider="custom:tensorfold",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.max_tokens == 150_000
    assert agent.context_compressor.max_tokens == 150_000
    kw = agent._build_api_kwargs([{"role": "user", "content": "x"}], tools_for_api=[])
    assert kw.get("max_tokens") == 150_000


def test_output_token_budget_route_isolated_from_other_providers(tmp_path, monkeypatch):
    cfg = {
        "model": {"default": "other-model", "provider": "custom:other", "context_length": 1_000_000},
        "providers": {
            "tensorfold": {
                "api": "http://127.0.0.1:9/v1",
                "api_key": "k",
                "output_token_budget": 150_000,
            },
            "other": {"api": "http://127.0.0.1:8/v1", "api_key": "k"},
        },
        "compression": {"threshold": 0.5},
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="other-model",
            api_key="k",
            base_url="http://127.0.0.1:8/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.max_tokens is None
    assert agent.context_compressor.max_tokens is None


def test_invalid_output_token_budget_ignored(tmp_path, monkeypatch):
    cfg = {
        "model": {"default": "deepseek-r1", "context_length": 1_000_000},
        "providers": {
            "tensorfold": {
                "api": "http://127.0.0.1:9/v1",
                "api_key": "k",
                "output_token_budget": -5,
            },
        },
        "compression": {"threshold": 0.5},
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    assert agent.max_tokens is None


def test_extra_body_does_not_emit_top_level_max_tokens_via_profile_transport():
    transport = ChatCompletionsTransport()
    from providers import get_provider_profile

    kw = transport.build_kwargs(
        "deepseek-r1",
        [{"role": "user", "content": "x"}],
        provider_profile=get_provider_profile("custom"),
        base_url="http://127.0.0.1:9/v1",
        request_overrides={"extra_body": {"max_tokens": 150_000}},
        max_tokens_param_fn=lambda v: {"max_tokens": v},
    )
    assert "max_tokens" not in kw
    assert kw.get("extra_body", {}).get("max_tokens") == 150_000
