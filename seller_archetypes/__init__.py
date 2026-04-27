"""Seller archetype registry.

Each archetype module exposes a `render(product) -> str` function that
returns the system-prompt content for that seller persona. The dispatch
table below is the single source of truth that `Conversation.format_seller_prompt`
and the v2 test suite consult.

Archetype catalogue (see individual modules for the cited prior art):
  - honest_retailer            : v1 baseline; verbatim move of the original
                                 inlined seller prompt.
  - dark_pattern_marketplace   : TrickyArena / DECEPTICON tactics.
  - prompt_injecting           : WASP / WebInject style indirect injection.
  - colluding                  : Motwani 2402.07510 / Lin 2410.00031.
  - misrepresenting            : false feature/warranty/model claims.
"""

from typing import Callable, Dict, List

from seller_archetypes import (
    colluding,
    dark_pattern_marketplace,
    honest_retailer,
    misrepresenting,
    prompt_injecting,
)


# Single source of truth: archetype name -> render(product) -> str.
ARCHETYPES: Dict[str, Callable[[Dict], str]] = {
    "honest_retailer": honest_retailer.render,
    "dark_pattern_marketplace": dark_pattern_marketplace.render,
    "prompt_injecting": prompt_injecting.render,
    "colluding": colluding.render,
    "misrepresenting": misrepresenting.render,
}


def list_archetypes() -> List[str]:
    """Return the canonical archetype names in registration order."""
    return list(ARCHETYPES.keys())


def render_seller_system_prompt(archetype: str, product: Dict) -> str:
    """Render the seller system prompt for `archetype`.

    Falls back to `honest_retailer` (the v1 baseline) when `archetype` is
    None, empty, or unknown. This keeps the negotiation loop forward-
    compatible with future archetype names that haven't been registered yet.
    """
    if not archetype or archetype not in ARCHETYPES:
        return ARCHETYPES["honest_retailer"](product)
    return ARCHETYPES[archetype](product)


__all__ = [
    "ARCHETYPES",
    "list_archetypes",
    "render_seller_system_prompt",
]
