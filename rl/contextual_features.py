"""v3 context featuriser for the LinUCB contextual bandit.

A pure, dependency-free function `featurise(...)` that returns a fixed-length
numerical context vector. The schema is deliberately small (CONTEXT_DIM ~24)
so the LinUCB matrix maths stays comfortably tractable in stdlib.

Design follows the BaRP recipe (arXiv:2510.07429): a low-dimensional context
mixing scenario / mandate / persona / archetype / product features so the
bandit can specialise per buyer profile. We deliberately keep the feature set
hand-engineered rather than learned because the v3 *novelty claim* is the
anomaly-typed mandate-aware reward signal, NOT the algorithm.

Two call modes are supported:
  (a) Pre-episode  -- pass a `Scenario` object plus the resolved product dict
                      (used by the bandit's `select(context)`).
  (b) Post-episode -- pass the saved `data` dict produced by `run_episode`
                      (used for diagnostics / re-fit).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# Feature layout (each numbered block matches the spec):
#  1) bias                                       1 dim
#  2) log10(hard_cap + 1)                        1 dim
#  3) per_item_cap / hard_cap                    1 dim
#  4) requires_confirm_above / hard_cap          1 dim
#  5) persona price_sensitivity one-hot [L,M,H]  3 dims
#  6) persona urgency one-hot [L,M,H]            3 dims
#  7) seller archetype one-hot (5 known)         5 dims
#  8) product price band one-hot (4 buckets)     4 dims
#  9) product type one-hot (Vehicle/Electronics/Real Estate/other) 4 dims
# Total                                         23 dims
CONTEXT_DIM: int = 1 + 1 + 1 + 1 + 3 + 3 + 5 + 4 + 4  # 23

# Stable orderings -- changing these changes feature semantics across saved
# bandit dicts, so treat as schema constants.
_PERSONA_LEVELS = ("low", "medium", "high")
_ARCHETYPES = (
    "honest_retailer",
    "dark_pattern_marketplace",
    "prompt_injecting",
    "colluding",
    "misrepresenting",
)
_PRODUCT_TYPES = ("Vehicle", "Electronics", "Real Estate", "other")


# ---------------------------------------------------------------------------
# Small builder helpers (unit-testable in isolation)
# ---------------------------------------------------------------------------
def _persona_onehot(value: Optional[str]) -> List[float]:
    """One-hot over (low, medium, high). Unknown -> medium (defensive)."""
    v = (value or "medium").strip().lower()
    if v not in _PERSONA_LEVELS:
        v = "medium"
    return [1.0 if level == v else 0.0 for level in _PERSONA_LEVELS]


def _archetype_onehot(value: Optional[str]) -> List[float]:
    """One-hot over the five v2 archetypes. Unknown -> honest_retailer."""
    v = (value or "honest_retailer").strip()
    if v not in _ARCHETYPES:
        v = "honest_retailer"
    return [1.0 if a == v else 0.0 for a in _ARCHETYPES]


def _price_band_onehot(retail_price: float) -> List[float]:
    """4-way price band: <500, 500-2000, 2000-10000, >10000."""
    bands = [
        retail_price < 500.0,
        500.0 <= retail_price < 2000.0,
        2000.0 <= retail_price < 10000.0,
        retail_price >= 10000.0,
    ]
    return [1.0 if b else 0.0 for b in bands]


def _product_type_onehot(ptype: Optional[str]) -> List[float]:
    """One-hot over (Vehicle, Electronics, Real Estate, other)."""
    v = (ptype or "other").strip()
    if v not in _PRODUCT_TYPES[:3]:
        v = "other"
    return [1.0 if t == v else 0.0 for t in _PRODUCT_TYPES]


def _parse_price(raw: Any) -> float:
    """Coerce '$1,299' / 1299 / '1299.0' to float; 0.0 on failure."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# Scenario / episode-dict adapter helpers
# ---------------------------------------------------------------------------
def _extract_mandate(obj: Any) -> Dict[str, Optional[float]]:
    """Pull (hard_cap, per_item_cap, requires_confirm_above) defensively.

    Accepts a `Scenario` dataclass instance, a dict-shaped scenario, or an
    episode `data` dict (which embeds scenario under `scenario`).
    """
    hard_cap: Optional[float] = None
    per_item_cap: Optional[float] = None
    confirm_above: Optional[float] = None

    auth = None
    # Scenario dataclass instance
    if hasattr(obj, "spend_authorization"):
        auth = obj.spend_authorization
    # Episode data dict with attached scenario
    elif isinstance(obj, dict) and isinstance(obj.get("scenario"), dict):
        auth = obj["scenario"].get("spend_authorization")
    # Raw scenario dict
    elif isinstance(obj, dict) and isinstance(obj.get("spend_authorization"), dict):
        auth = obj["spend_authorization"]

    if auth is None:
        return {"hard_cap": hard_cap, "per_item_cap": per_item_cap,
                "confirm_above": confirm_above}

    if hasattr(auth, "hard_cap"):
        hard_cap = getattr(auth, "hard_cap", None)
        per_item_cap = getattr(auth, "per_item_cap", None)
        confirm_above = getattr(auth, "requires_confirm_above", None)
    elif isinstance(auth, dict):
        hard_cap = auth.get("hard_cap")
        per_item_cap = auth.get("per_item_cap")
        confirm_above = auth.get("requires_confirm_above")

    return {
        "hard_cap": float(hard_cap) if hard_cap is not None else None,
        "per_item_cap": float(per_item_cap) if per_item_cap is not None else None,
        "confirm_above": float(confirm_above) if confirm_above is not None else None,
    }


def _extract_persona(obj: Any) -> Dict[str, Optional[str]]:
    persona = None
    if hasattr(obj, "user_persona"):
        persona = obj.user_persona
    elif isinstance(obj, dict) and isinstance(obj.get("scenario"), dict):
        persona = obj["scenario"].get("user_persona")
    elif isinstance(obj, dict) and isinstance(obj.get("user_persona"), dict):
        persona = obj["user_persona"]

    price_sens = urgency = None
    if persona is None:
        return {"price_sensitivity": price_sens, "urgency": urgency}
    if hasattr(persona, "price_sensitivity"):
        price_sens = getattr(persona, "price_sensitivity", None)
        urgency = getattr(persona, "urgency", None)
    elif isinstance(persona, dict):
        price_sens = persona.get("price_sensitivity")
        urgency = persona.get("urgency")
    return {"price_sensitivity": price_sens, "urgency": urgency}


def _extract_archetype(obj: Any) -> Optional[str]:
    if hasattr(obj, "seller_archetype"):
        return getattr(obj, "seller_archetype", None)
    if isinstance(obj, dict):
        if obj.get("seller_archetype"):
            return obj.get("seller_archetype")
        scen = obj.get("scenario")
        if isinstance(scen, dict):
            return scen.get("seller_archetype")
    return None


def _extract_product(obj: Any, product_arg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if product_arg is not None:
        return product_arg
    if hasattr(obj, "product") and getattr(obj, "product"):
        return getattr(obj, "product")
    if isinstance(obj, dict):
        if isinstance(obj.get("product_data"), dict):
            return obj["product_data"]
        if isinstance(obj.get("product"), dict):
            return obj["product"]
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def featurise(
    scenario_or_episode_dict: Any,
    product: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Return a fixed-length context vector of length `CONTEXT_DIM`.

    Accepts either a `Scenario` dataclass (with optional `product` argument)
    or a saved episode `data` dict. Missing fields default sensibly so old
    v1-shape datasets still produce a vector of the correct length.
    """
    obj = scenario_or_episode_dict
    mandate = _extract_mandate(obj)
    persona = _extract_persona(obj)
    archetype = _extract_archetype(obj)
    product_dict = _extract_product(obj, product)

    feats: List[float] = []

    # 1) Bias term -- always 1.0 (lets LinUCB learn an arm-specific intercept)
    feats.append(1.0)

    # 2) Hard cap, log-scaled. Defaults to retail price when scenario absent.
    hc = mandate["hard_cap"]
    if hc is None:
        hc = _parse_price(product_dict.get("Retail Price"))
    feats.append(math.log10(max(hc, 0.0) + 1.0))

    # 3) Per-item cap to hard-cap ratio (clamped 0..1).
    pic = mandate["per_item_cap"] if mandate["per_item_cap"] is not None else hc
    ratio_pic = (pic / hc) if hc and hc > 0 else 1.0
    feats.append(_clamp_unit(ratio_pic))

    # 4) Confirm-threshold ratio (0.0 if absent, capped at 1.0).
    cab = mandate["confirm_above"]
    ratio_cab = (cab / hc) if (cab is not None and hc and hc > 0) else 0.0
    feats.append(_clamp_unit(ratio_cab))

    # 5) Persona price-sensitivity one-hot.
    feats.extend(_persona_onehot(persona["price_sensitivity"]))

    # 6) Persona urgency one-hot.
    feats.extend(_persona_onehot(persona["urgency"]))

    # 7) Seller archetype one-hot.
    feats.extend(_archetype_onehot(archetype))

    # 8) Product retail price band one-hot.
    retail = _parse_price(product_dict.get("Retail Price"))
    feats.extend(_price_band_onehot(retail))

    # 9) Product type one-hot (with implicit "other" bucket).
    feats.extend(_product_type_onehot(product_dict.get("Type")))

    # Defensive guard: never let a coding error change the schema.
    if len(feats) != CONTEXT_DIM:  # pragma: no cover
        raise RuntimeError(
            f"featurise produced {len(feats)} dims, expected {CONTEXT_DIM}. "
            "Update CONTEXT_DIM if the schema changed intentionally."
        )
    return feats
