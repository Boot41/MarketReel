from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class RiskOutputChecker(BaseAgent):
    """Escalates if risk_flags in session state is a non-empty valid list."""

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        flags = context.session.state.get("risk_flags")
        if isinstance(flags, list) and len(flags) > 0:
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name)


class ValuationOutputChecker(BaseAgent):
    """Escalates if valuation_result in session state has all required fields."""

    _REQUIRED = frozenset(
        {
            "mg_estimate_usd",
            "confidence_interval_low_usd",
            "confidence_interval_high_usd",
            "theatrical_projection_usd",
            "vod_projection_usd",
        }
    )

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        result = context.session.state.get("valuation_result")
        if isinstance(result, dict) and self._REQUIRED.issubset(result.keys()):
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name)


class StrategyOutputChecker(BaseAgent):
    """Escalates if strategy_result in session state has all required fields."""

    _REQUIRED = frozenset(
        {
            "release_mode",
            "release_window_days",
            "marketing_spend_usd",
            "platform_priority",
            "roi_scenarios",
        }
    )

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        result = context.session.state.get("strategy_result")
        if isinstance(result, dict) and self._REQUIRED.issubset(result.keys()):
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name)
