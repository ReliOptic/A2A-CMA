"""A2A-CMA v2.5 evaluation package.

Houses the counterfactual user-regret oracle judge introduced in v2.5.

The judge operationalises the *would-the-user-have-approved-this* question
as a per-episode metric, filling the open gap identified in
``docs/RELATED_WORK.md`` section 8 (AI-LieDar 2409.09013, Outcome-Oriented
Eval 2511.08242). It is intentionally decoupled from MarkAnomaly's
heuristic detectors so the v2 paper can quote a clean definition.

Public surface (re-exported here so downstream code can simply
``from evaluation import RegretJudge, judge_episode``):

* :class:`RegretVerdict`      -- enum of APPROVE / REGRET / HESITATE outcomes.
* :class:`RegretRationale`    -- judge output dataclass (verdict + reason).
* :class:`RegretJudge`        -- the judge itself; takes a ``model_factory``
                                 hook so unit tests can inject stubs.
* :func:`judge_episode`       -- convenience wrapper that lazy-instantiates
                                 a default judge.
* :func:`aggregate_regret_rate` -- corpus-level summary statistics.
"""

from .regret import (  # noqa: F401
    RegretJudge,
    RegretRationale,
    RegretVerdict,
    aggregate_regret_rate,
    judge_episode,
)

__all__ = [
    "RegretJudge",
    "RegretRationale",
    "RegretVerdict",
    "aggregate_regret_rate",
    "judge_episode",
]
