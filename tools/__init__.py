"""A2A-CMA developer tooling.

This package hosts stdlib-only helper scripts used to bootstrap analyses
when real episode data is unavailable (e.g. CI, fresh checkouts, the v2
notebook running for the first time). Nothing in here may import
anything outside the standard library so it stays usable in the most
constrained environments.

Submodules are imported lazily (on first attribute access) so that
``python -m tools.generate_synthetic_episodes`` does not double-import the
script and trigger a ``RuntimeWarning``.
"""

from typing import Any

__all__ = ["generate"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute hook
    if name == "generate":
        from tools.generate_synthetic_episodes import generate as _generate
        return _generate
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
