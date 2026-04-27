# Contributing to A2A-CMA

Thank you for considering a contribution. This guide describes the kinds of
contributions we accept, how to get a local environment running, and the
coding conventions the project uses. It is intentionally short — read it
once, then refer back to the relevant section.

## About this benchmark

A2A-CMA is positioned as a **safety benchmark for agentic commerce** — the
class of products represented by Amazon Buy-for-Me, OpenAI Operator,
Anthropic Computer Use, and the AP2 / ACP payment-mandate stack. It is
explicitly *not* a capability benchmark: we do not measure whether an agent
can complete a purchase, we measure whether it does so without harming the
human who issued the mandate (overpayment, mandate breach, falling for an
adversarial seller, etc.).

For the long-form positioning see [`docs/CONCEPT.md`](docs/CONCEPT.md). For
the citation backbone (the 14 papers and protocol documents we ground
against) see [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## What kinds of contributions are welcome

The four contribution shapes we have a clear review path for:

### 1. New scenarios

Edit [`dataset/scenarios.json`](dataset/scenarios.json). Each scenario is
typed by [`scenarios/schema.py`](scenarios/schema.py). A new scenario must:

- Reference a `product_id` that is already present in
  [`dataset/products.json`](dataset/products.json). We do not accept
  product additions and scenario additions in the same PR — split them.
- Include a non-empty `user_persona.summary` describing who the buyer is
  (price sensitivity, urgency, prior experience). The persona is what the
  user-regret oracle re-reads in the morning.
- Set `seller_archetype` to one of the registered archetype names (see
  [`seller_archetypes/`](seller_archetypes/)). PRs that introduce a new
  archetype name without also adding the implementation will be closed.
- Include an `expected_risks` list drawn from the anomaly fields defined
  in `MarkAnomaly.py`. This is documentation, not an oracle: the test is
  what the agent actually does, not what we expected.
- Set `v1_active` explicitly. Scenarios used by the published v1
  leaderboard must remain stable — once a scenario is `v1_active=true`
  and a release is cut, it cannot be edited without bumping the
  benchmark version.

### 2. New seller archetypes

Add a module under [`seller_archetypes/`](seller_archetypes/). Required:

- A `render(product) -> str` function that returns the seller's product
  pitch (deterministic given `product`; no LLM calls inside `render`).
- A comment-cited prior-art reference drawn from
  [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md). If your tactic is not
  already cited there, add the entry in the same PR. Archetypes without
  a citation are not academic contributions and we will ask for one.
- The wholesale-floor invariant must be preserved: the seller may bluff,
  anchor, mislead, or inject, but it may not accept a settlement below
  its wholesale price. This is what keeps episodes economically grounded.
- The one-message-only constraint must be preserved: the seller speaks
  once per turn. Multi-message tactics belong in a future archetype API,
  not in single-archetype workarounds.
- At least one corresponding update to `MarkAnomaly.py`: either a new
  detector function, an addition to `INJECTION_MARKERS` /
  `COLLUSION_MARKERS`, or a new boolean anomaly key. Without a detector
  the archetype produces no measurable signal.

### 3. New anomaly detectors

Edit [`MarkAnomaly.py`](MarkAnomaly.py). Required:

- Defensive about missing keys. Anomaly detection runs against historical
  episodes that may pre-date the new field; a `KeyError` on an old file
  is a bug. Default to `False` for booleans and `0` for numerics.
- Accompanied by a smoke test under `tests/`. The test must construct a
  synthetic episode dict (no LLM calls) and assert the detector returns
  the expected value for at least one positive and one negative case.
- The new field name should be documented in the release notes and, if
  it becomes a headline metric, added to
  [`benchmarks/SUBMISSION.md`](benchmarks/SUBMISSION.md).

### 4. Leaderboard submissions

If you have run a buyer agent against the v1 scenario set and want to
appear on the public leaderboard, follow
[`benchmarks/SUBMISSION.md`](benchmarks/SUBMISSION.md). We do not require
you to share the agent's source code, but we do require the raw episode
JSONs so a reviewer can replay the dataset hash check and spot-check the
anomaly counts.

## Development workflow

```
git clone https://github.com/<your-fork>/A2A-CMA.git
cd A2A-CMA
pip install -r requirements.txt
python3 -m unittest discover tests
python -m a2a_cma_cli list-scenarios
```

The unit-test suite is stdlib-only and runs in well under a minute. It is
the right first signal that your environment works. The CLI smoke
(`list-scenarios`) confirms the dataset loader can read your edits.

When opening a PR:

- Target the active development branch named in the README. Do not open
  PRs against tagged release branches.
- Include a one-line summary of the change and a paragraph explaining
  why. For scenario or archetype PRs, link the prior-art entry.
- Keep the diff focused. One scenario, one archetype, or one detector
  per PR makes review tractable.

## Coding conventions

- **Stdlib first.** The core engine, anomaly detection, scenario loader,
  and CLI all run on the Python standard library. New pip dependencies
  require a one-paragraph justification in the PR.
- **Lazy imports for heavy deps.** `numpy`, `pandas`, OpenAI client, and
  any model-provider SDK are imported inside the function that needs
  them, never at module top. The pattern is in `MarkAnomaly.py` and
  `evaluation/regret.py`.
- **Type hints where they add clarity.** Function signatures, dataclass
  fields, and public return types should be typed. Internal helpers that
  are obvious from a five-line body do not need annotations.
- **No comments restating the obvious.** Comments should explain *why*
  a value was chosen, *why* a branch exists, or cite the paper that
  motivated a heuristic. Comments that paraphrase the next line of code
  should be deleted.
- **`INJECTION_MARKERS` and `COLLUSION_MARKERS` are append-only across
  releases.** A submission generated against v1 must continue to score
  identically on v1 forever, even after the markers list grows for v2.
  If you need to remove a marker, bump the benchmark version.

## Citing prior art

Every new seller archetype and every new anomaly detector must include a
comment citing an entry in [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).
Examples already in the codebase:

- `seller_archetypes/prompt_injecting.py` cites WASP (arXiv:2504.18575).
- `seller_archetypes/colluding.py` cites the CSA AP2 MAESTRO threat
  model (Cloud Security Alliance, Oct 2025).
- The `regret_flagged` detector in `MarkAnomaly.py` cites AI-LieDar
  (arXiv:2409.09013) and Outcome-Oriented Eval (arXiv:2511.08242) via
  `evaluation/regret.py`.

If your contribution operationalises a tactic that is not yet in the
related-work file, add the entry under the appropriate section in the
same PR. We would rather grow the bibliography than accept un-grounded
heuristics.

## What we won't accept (yet)

- Changes that break baseline back-compat. If a v1 submission would no
  longer parse against your change, it is a breaking change and needs
  a version bump.
- New pip dependencies without strong justification. The default answer
  is "do it with stdlib".
- Scenarios using a `seller_archetype` value that is not in the
  registered set. Add the archetype first, in a separate PR.
- Leaderboard submissions without raw episode JSONs. We cannot verify
  aggregate metrics from aggregates alone; the dataset hash check and
  the spot-check both need the underlying episodes.
- PRs that touch more than one of: scenarios, archetypes, detectors,
  CLI, evaluation. Split them. One concern per PR keeps review honest.

If your contribution does not fit any of the four shapes above, open a
discussion issue first so we can scope it together.
