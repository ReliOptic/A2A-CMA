"""Scenario layer for A2A-CMA v1 (Amazon Buy-for-Me grounding)."""

from scenarios.schema import Scenario, SpendAuthorization, UserPersona
from scenarios.loader import load_scenarios, get_scenario, attach_product
from scenarios.prompt import render_buyer_scenario_clause

__all__ = [
    "Scenario",
    "SpendAuthorization",
    "UserPersona",
    "load_scenarios",
    "get_scenario",
    "attach_product",
    "render_buyer_scenario_clause",
]
