"""A2A-CMA developer-toolkit CLI (v4).

The CLI is the primary entry point for external developers who want to
stress-test a shopping agent against the v1->v3 safety scenarios in a
single command. Three subcommands are exposed:

* ``list-scenarios`` -- print the active scenarios (with optional archetype
  filter) so a developer can pick a target without opening the JSON file.
* ``run`` / ``run-all`` -- drive ``Conversation.run_negotiation`` over one
  or every active scenario, persist the resulting episode JSONs.
* ``report`` -- aggregate anomalies / gateway interventions / regret
  verdicts / RL reward signal across an output directory.

The package is intentionally stdlib-only; all heavy lifting lives in the
existing ``Conversation``, ``MarkAnomaly``, ``payments``, ``scenarios``,
``seller_archetypes``, ``evaluation`` and ``rl`` modules.
"""

from a2a_cma_cli.cli import main

__all__ = ["main"]
