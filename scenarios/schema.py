"""Typed schema for A2A-CMA scenarios.

Kept dependency-free (stdlib dataclasses only) so the loader works in the
existing python>=3.9 conda env without adding pydantic.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class UserPersona:
    summary: str
    price_sensitivity: str = "medium"   # "low" | "medium" | "high"
    urgency: str = "medium"             # "low" | "medium" | "high"


@dataclass
class SpendAuthorization:
    """AP2-flavoured mandate the buyer agent operates under.

    `hard_cap` is the absolute ceiling the payment layer would reject.
    `per_item_cap` is the cap for a single line item.
    `requires_confirm_above` is the threshold above which a real Buy-for-Me
    flow would interrupt for human confirmation; in v1 we treat it as a soft
    warning that the buyer prompt is told about.
    """
    hard_cap: float
    per_item_cap: Optional[float] = None
    requires_confirm_above: Optional[float] = None
    currency: str = "USD"
    mandate_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.per_item_cap is None:
            self.per_item_cap = self.hard_cap


@dataclass
class Scenario:
    scenario_id: str
    service_template: str               # e.g. "amazon_buy_for_me"
    description: str
    product_id: int
    user_persona: UserPersona
    spend_authorization: SpendAuthorization
    seller_archetype: str = "honest_retailer"
    expected_risks: List[str] = field(default_factory=list)
    v1_active: bool = True

    # Populated by `attach_product` after loading.
    product: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Scenario":
        persona = UserPersona(**raw["user_persona"])
        auth = SpendAuthorization(**raw["spend_authorization"])
        return cls(
            scenario_id=raw["scenario_id"],
            service_template=raw["service_template"],
            description=raw["description"],
            product_id=int(raw["product_id"]),
            user_persona=persona,
            spend_authorization=auth,
            seller_archetype=raw.get("seller_archetype", "honest_retailer"),
            expected_risks=list(raw.get("expected_risks", [])),
            v1_active=bool(raw.get("v1_active", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Strip the resolved product blob from serialisation; it's loaded on demand.
        d.pop("product", None)
        return d
