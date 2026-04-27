"""Load scenarios from JSON and resolve their product references."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scenarios.schema import Scenario


PathLike = Union[str, Path]


def load_scenarios(path: PathLike = "dataset/scenarios.json") -> List[Scenario]:
    """Load every scenario in the file. No product attachment yet."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of scenarios.")
    return [Scenario.from_dict(item) for item in raw]


def get_scenario(
    scenario_id: str,
    path: PathLike = "dataset/scenarios.json",
) -> Scenario:
    for s in load_scenarios(path):
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Scenario id {scenario_id!r} not found in {path}.")


def attach_product(
    scenario: Scenario,
    products_path: PathLike = "dataset/products.json",
) -> Scenario:
    """Resolve `scenario.product_id` against products.json in place."""
    products = json.loads(Path(products_path).read_text(encoding="utf-8"))
    matches = [p for p in products if p.get("id") == scenario.product_id]
    if not matches:
        raise KeyError(
            f"Scenario {scenario.scenario_id} references product_id="
            f"{scenario.product_id} which is not in {products_path}."
        )
    scenario.product = matches[0]
    return scenario


def load_active_scenarios(
    scenarios_path: PathLike = "dataset/scenarios.json",
    products_path: PathLike = "dataset/products.json",
) -> List[Scenario]:
    """Load all v1-active scenarios with products already attached."""
    out: List[Scenario] = []
    for s in load_scenarios(scenarios_path):
        if not s.v1_active:
            continue
        attach_product(s, products_path)
        out.append(s)
    return out
