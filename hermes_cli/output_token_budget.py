"""Opt-in per-route output token budget (wire ceiling + context-engine reserve).

Not a revival of ``model.max_tokens``, ``HERMES_MAX_TOKENS``, or provider ``max_output_tokens``.
Only ``providers.<id>.output_token_budget`` (or per-model override) on a matching route.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from hermes_cli.config_providers import _entries_for_route, _route_model_cfgs

logger = logging.getLogger(__name__)

_OUTPUT_BUDGET_KEY = "output_token_budget"


class OutputTokenBudgetAlignmentError(RuntimeError):
    """Configured ``output_token_budget`` cannot be applied on both wire and context engine."""


def coerce_output_token_budget(raw: Any) -> Optional[int]:
    """Positive int budget, or None when unset/invalid."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, float):
        if math.isnan(raw) or math.isinf(raw):
            return None
        if not raw.is_integer():
            return None
        raw = int(raw)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_route_output_token_budget(
    config: dict[str, Any] | None,
    *,
    provider: str = "",
    base_url: str = "",
    model: str = "",
    requested_provider: str = "",
    context_length: Optional[int] = None,
) -> Optional[int]:
    """Budget for the active custom route (base_url match only); None when unset."""
    if not isinstance(config, dict) or not base_url or not model:
        return None
    from hermes_cli.config import get_compatible_custom_providers

    try:
        custom_providers = get_compatible_custom_providers(config)
    except Exception:
        return None

    raw: Any = None
    for model_cfg in _route_model_cfgs(model, base_url, custom_providers, config):
        if _OUTPUT_BUDGET_KEY in model_cfg:
            raw = model_cfg.get(_OUTPUT_BUDGET_KEY)
            break
    if raw is None:
        for entry in _entries_for_route(base_url, custom_providers, config):
            if _OUTPUT_BUDGET_KEY in entry:
                raw = entry.get(_OUTPUT_BUDGET_KEY)
                break

    budget = coerce_output_token_budget(raw)
    if budget is None:
        if raw is not None:
            logger.warning(
                "Invalid output_token_budget=%r for model %r at %s — ignored",
                raw, model, base_url,
            )
        return None

    if isinstance(context_length, int) and context_length > 0 and budget >= context_length:
        logger.warning(
            "output_token_budget=%s for %s is not below context_length=%s — ignored",
            budget, model, context_length,
        )
        return None
    return budget


def apply_route_output_token_budget(agent: Any, config: Optional[dict] = None) -> None:
    """Set ``agent.max_tokens`` from config when the caller did not pin a budget."""
    if getattr(agent, "_max_tokens_user_pinned", False):
        return
    if getattr(agent, "max_tokens", None) is not None:
        return
    try:
        from hermes_cli.config import load_config_readonly

        cfg = config if isinstance(config, dict) else load_config_readonly()
    except Exception:
        cfg = config
    if not isinstance(cfg, dict):
        return

    ctx_hint = getattr(agent, "_config_context_length", None)
    if not isinstance(ctx_hint, int):
        ctx_hint = getattr(getattr(agent, "context_compressor", None), "context_length", None)

    budget = resolve_route_output_token_budget(
        cfg,
        provider=str(getattr(agent, "provider", "") or ""),
        base_url=str(getattr(agent, "base_url", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
        requested_provider=str(getattr(agent, "requested_provider", "") or ""),
        context_length=ctx_hint if isinstance(ctx_hint, int) else None,
    )
    if budget is None:
        return
    agent.max_tokens = budget
    agent._output_token_budget_from_config = True
