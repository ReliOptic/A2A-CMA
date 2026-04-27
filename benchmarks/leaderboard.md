# A2A-CMA Leaderboard — v1

This is the v1 leaderboard for A2A-CMA, the safety benchmark for agentic
commerce buyer agents. Submissions are ranked by **`regret_rate`** (lower
is better — fewer episodes the user-regret oracle marked `WOULD_REGRET`).
Ties are broken by **`mean_reward`** (higher is better). All metrics are
defined in [`SUBMISSION.md`](SUBMISSION.md).

The two rows below are reference baselines drawn from internal runs, not
external submissions. They exist to give first-time submitters a sense of
what "bad" and "less bad" look like on this benchmark, and to anchor the
leaderboard at launch.

| Rank | Team | Agent | regret_rate ↓ | injection_compliance ↓ | dark_pattern ↓ | collusion ↓ | overpayment ↓ | mean_reward ↑ | submission |
|---|---|---|---|---|---|---|---|---|---|
| 1 | a2a-cma-internal *(reference, not a submission)* | v1-bandit-best | 0.30 | 0.04 | 0.13 | 0.07 | 0.18 | -0.55 | (internal) |
| 2 | a2a-cma-internal *(reference, not a submission)* | honest-baseline | 0.45 | 0.16 | 0.22 | 0.18 | 0.26 | -1.10 | (internal) |

**Reference row notes.**

- *honest-baseline* — `gpt-4o-mini` running the original v1 buyer prompt
  with no scenario-aware mitigation. This is the "out of the box" number
  for an agent that knows nothing about adversarial sellers.
- *v1-bandit-best* — the best softmax-bandit prompt selected by the
  original `rl/` work. Improvement over the baseline is concentrated in
  the injection and collusion susceptibility columns; the dark-pattern
  column moves less because the bandit's action space did not include a
  dark-pattern-specific prompt variant in v1.

## How to submit

Read [`SUBMISSION.md`](SUBMISSION.md) for the full submission guide, the
required JSON schema, and the review SLA. The short version: run
`python -m a2a_cma_cli run-all` followed by `python -m a2a_cma_cli report`
against the v1 scenario set, then open a PR adding your generated
`report.json` under `benchmarks/submissions/`. Reviewers will verify the
dataset hash, replay the anomaly detectors against your raw episode JSONs,
and append a row to this table once the checks pass.

A worked example of the submission JSON schema lives at
[`_example_submission.json`](_example_submission.json).
