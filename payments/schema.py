"""Mandate / token schema for the AP2-style payment layer.

Mirrors the stdlib-only style of `scenarios/schema.py`. We model just enough of
Google's Agent Payments Protocol (AP2) and Stripe/OpenAI's Agentic Commerce
Protocol (ACP) to make the safety story testable end-to-end:

- `IntentMandate` is the user-signed authorisation the agent operates under.
- `CartMandate` binds a concrete cart to that intent before settlement.
- `SharedPaymentToken` is the ACP-style ephemeral token the gateway issues
  once it has decided the cart is in-policy.

No real cryptography happens here; the gateway is a deterministic policy
engine, not a payment processor.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class IntentMandate:
    mandate_id: str
    currency: str
    hard_cap: float
    per_item_cap: float
    requires_confirm_above: Optional[float] = None
    allowed_categories: List[str] = field(default_factory=list)
    issued_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CartMandate:
    cart_id: str
    intent_mandate_id: str
    line_items: List[Dict[str, Any]]
    total: float

    @property
    def total_matches_line_items(self) -> bool:
        """True iff `total` equals the sum of unit_price * qty across items.

        Uses a small tolerance for float arithmetic so a cent-level rounding
        artefact doesn't trip the check.
        """
        computed = sum(
            float(item["unit_price"]) * float(item["qty"])
            for item in self.line_items
        )
        return abs(computed - float(self.total)) < 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SharedPaymentToken:
    token_id: str
    mandate_id: str
    cart_id: Optional[str]
    amount: float
    currency: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuthorizationOutcome(str, Enum):
    APPROVED = "APPROVED"
    NEEDS_HUMAN_CONFIRM = "NEEDS_HUMAN_CONFIRM"
    DECLINED_OVER_HARD_CAP = "DECLINED_OVER_HARD_CAP"
    DECLINED_OVER_PER_ITEM_CAP = "DECLINED_OVER_PER_ITEM_CAP"
    DECLINED_CATEGORY_NOT_ALLOWED = "DECLINED_CATEGORY_NOT_ALLOWED"


@dataclass
class AuthorizationDecision:
    outcome: AuthorizationOutcome
    reason: str
    token: Optional[SharedPaymentToken] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "token": self.token.to_dict() if self.token is not None else None,
        }
