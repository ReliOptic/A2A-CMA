"""LinUCB contextual bandit (stdlib-only) for v3 prompt selection.

Implements the standard disjoint-arm LinUCB update:

    A_a   <- A_a   + x x^T
    b_a   <- b_a   + r * x
    theta_a = A_a^{-1} b_a
    score_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

This is the algorithm cited in "Online Multi-LLM Selection via Contextual
Bandits" (arXiv:2506.17670) and is the BaRP-style upgrade path
(arXiv:2510.07429) for the previous stateless `SoftmaxPolicy`. The v3
*novelty claim* is the anomaly-typed mandate-aware reward signal -- this
class is intentionally a textbook LinUCB so the contribution remains crisp.

No numpy: matrix inverse is implemented via Gauss-Jordan elimination on a
small (CONTEXT_DIM x CONTEXT_DIM) dense matrix (~23x23 in practice).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tiny matrix helpers (stdlib-only)
# ---------------------------------------------------------------------------
def _identity(n: int, scale: float = 1.0) -> List[List[float]]:
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def _zeros(n: int) -> List[float]:
    return [0.0 for _ in range(n)]


def _matvec(m: List[List[float]], v: List[float]) -> List[float]:
    n = len(m)
    out = [0.0] * n
    for i in range(n):
        row = m[i]
        s = 0.0
        for j in range(n):
            s += row[j] * v[j]
        out[i] = s
    return out


def _dot(a: List[float], b: List[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _outer_add_inplace(m: List[List[float]], v: List[float]) -> None:
    """m += v v^T  (in-place)."""
    n = len(v)
    for i in range(n):
        vi = v[i]
        row = m[i]
        for j in range(n):
            row[j] += vi * v[j]


def _invert(matrix: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inverse with a one-shot ridge bump on near-singular pivots.

    Numerical robustness: if any pivot |p| < 1e-12 we add `ridge*1e-3` to the
    diagonal of a fresh copy and retry once. Still singular -> raise.
    """
    return _invert_impl(matrix, retry=True)


def _invert_impl(matrix: List[List[float]], retry: bool) -> List[List[float]]:
    n = len(matrix)
    # Augmented [A | I]
    aug: List[List[float]] = []
    for i in range(n):
        row = list(matrix[i]) + [1.0 if j == i else 0.0 for j in range(n)]
        aug.append(row)

    for col in range(n):
        # Partial pivoting: find the row with the largest |aug[r][col]|.
        pivot_row = col
        max_abs = abs(aug[col][col])
        for r in range(col + 1, n):
            if abs(aug[r][col]) > max_abs:
                max_abs = abs(aug[r][col])
                pivot_row = r
        if max_abs < 1e-12:
            if retry:
                # Bump diagonal by a small ridge and retry once.
                bumped = [row[:] for row in matrix]
                bump = 1e-3
                for i in range(n):
                    bumped[i][i] += bump
                return _invert_impl(bumped, retry=False)
            raise ValueError(
                "LinUCB matrix inverse failed: matrix is singular even after "
                "ridge bump. This indicates a degenerate context distribution."
            )
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalise pivot row.
        pivot = aug[col][col]
        inv_pivot = 1.0 / pivot
        for j in range(2 * n):
            aug[col][j] *= inv_pivot

        # Eliminate other rows.
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]

    # Right half is the inverse.
    inv = [row[n:] for row in aug]
    return inv


# ---------------------------------------------------------------------------
# LinUCB policy
# ---------------------------------------------------------------------------
class LinUCBPolicy:
    """Disjoint-arm LinUCB over `num_actions` arms in `context_dim` features."""

    def __init__(
        self,
        num_actions: int,
        context_dim: int,
        alpha: float = 1.0,
        ridge: float = 1.0,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")
        if context_dim <= 0:
            raise ValueError("context_dim must be > 0")
        self.num_actions = int(num_actions)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        # Per-arm A: identity * ridge (D x D).
        self.A: List[List[List[float]]] = [
            _identity(self.context_dim, scale=self.ridge)
            for _ in range(self.num_actions)
        ]
        # Per-arm b: D-dim zero.
        self.b: List[List[float]] = [
            _zeros(self.context_dim) for _ in range(self.num_actions)
        ]

    # -----------------------------------------------------------------
    def _theta(self, a: int) -> List[float]:
        """Ridge-regression weights theta_a = A_a^{-1} b_a."""
        A_inv = _invert(self.A[a])
        return _matvec(A_inv, self.b[a])

    # -----------------------------------------------------------------
    def select(self, context: List[float]) -> Tuple[int, float]:
        """Return (best_arm_index, ucb_score).

        score_a = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)
        Ties are broken in favour of the lower index (deterministic).
        """
        if len(context) != self.context_dim:
            raise ValueError(
                f"context has {len(context)} dims, expected {self.context_dim}"
            )
        best_arm = 0
        best_score = -float("inf")
        for a in range(self.num_actions):
            A_inv = _invert(self.A[a])
            theta = _matvec(A_inv, self.b[a])
            mean = _dot(theta, context)
            # x^T A^{-1} x  -- non-negative for SPD A_inv; clamp tiny negatives.
            quad = _dot(context, _matvec(A_inv, context))
            if quad < 0.0:
                quad = 0.0
            score = mean + self.alpha * math.sqrt(quad)
            if not math.isfinite(score):  # pragma: no cover -- defensive
                continue
            if score > best_score:
                best_score = score
                best_arm = a
        return best_arm, best_score

    # -----------------------------------------------------------------
    def update(self, action_idx: int, context: List[float], reward: float) -> None:
        """In-place sufficient-statistics update for the chosen arm."""
        if not (0 <= action_idx < self.num_actions):
            raise ValueError(f"action_idx {action_idx} out of range")
        if len(context) != self.context_dim:
            raise ValueError(
                f"context has {len(context)} dims, expected {self.context_dim}"
            )
        _outer_add_inplace(self.A[action_idx], context)
        b = self.b[action_idx]
        for i in range(self.context_dim):
            b[i] += reward * context[i]

    # -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "ridge": self.ridge,
            "A": [[row[:] for row in mat] for mat in self.A],
            "b": [vec[:] for vec in self.b],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinUCBPolicy":
        obj = cls(
            num_actions=int(d["num_actions"]),
            context_dim=int(d["context_dim"]),
            alpha=float(d.get("alpha", 1.0)),
            ridge=float(d.get("ridge", 1.0)),
        )
        obj.A = [[list(row) for row in mat] for mat in d["A"]]
        obj.b = [list(vec) for vec in d["b"]]
        return obj


# ---------------------------------------------------------------------------
# Optional helper -- softmax over UCB scores for epsilon-style exploration.
# ---------------------------------------------------------------------------
def softmax_select_from_ucb(
    scores: List[float],
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
) -> int:
    """Sample an arm index from softmax(scores / temperature).

    Useful when the caller wants a Boltzmann-style exploration on top of UCB
    rather than the deterministic argmax. Not used by the default training
    loop but exported because the spec called it out as a nice-to-have.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    rng = rng or random
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    if total <= 0:  # pragma: no cover -- numerical guard
        return int(max(range(len(scores)), key=lambda i: scores[i]))
    probs = [e / total for e in exps]
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1
