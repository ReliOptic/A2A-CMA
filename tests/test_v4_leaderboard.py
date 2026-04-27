"""Smoke tests for the v4 contributor + leaderboard scaffolding.

Stdlib unittest only. No network, no LLM imports. These tests verify the
shape of the documentation and the example submission JSON; they do not
attempt to reproduce any metric.
"""

import json
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
LEADERBOARD = BENCHMARKS_DIR / "leaderboard.md"
SUBMISSION_DOC = BENCHMARKS_DIR / "SUBMISSION.md"
EXAMPLE_JSON = BENCHMARKS_DIR / "_example_submission.json"
SUBMISSIONS_GITKEEP = BENCHMARKS_DIR / "submissions" / ".gitkeep"
SCENARIOS_JSON = REPO_ROOT / "dataset" / "scenarios.json"


REQUIRED_SUBMISSION_FIELDS = (
    "submission_id",
    "team",
    "agent_name",
    "agent_description",
    "agent_model_card_url",
    "benchmark_version",
    "eval_config",
    "aggregate_metrics",
    "per_scenario_metrics",
    "raw_episodes_url",
    "submitted_at",
)

HEADLINE_METRIC_KEYS = (
    "regret_rate",
    "injection_compliance_rate",
    "dark_pattern_susceptibility",
    "collusion_susceptibility",
    "overpayment_rate",
    "out_of_budget_rate",
    "gateway_intervention_rate",
    "mean_reward",
)


class ContributingDocTests(unittest.TestCase):
    def test_contributing_exists_with_required_sections(self):
        self.assertTrue(CONTRIBUTING.exists(), "CONTRIBUTING.md missing")
        text = CONTRIBUTING.read_text(encoding="utf-8")
        for header in (
            "What kinds of contributions are welcome",
            "Development workflow",
            "Citing prior art",
        ):
            self.assertIn(
                header,
                text,
                f"CONTRIBUTING.md missing required section header: {header!r}",
            )


class LeaderboardDocTests(unittest.TestCase):
    def test_leaderboard_exists_and_has_reference_rows(self):
        self.assertTrue(LEADERBOARD.exists(), "benchmarks/leaderboard.md missing")
        text = LEADERBOARD.read_text(encoding="utf-8")
        self.assertIn(
            "regret_rate",
            text,
            "leaderboard.md missing the regret_rate column header",
        )

        # Count table data rows: lines starting with `|` that come *after* the
        # markdown table separator (the line of `|---|---|...` dashes).
        lines = text.splitlines()
        data_rows = []
        seen_separator = False
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                seen_separator = False
                continue
            if not seen_separator:
                # Detect the separator row, which is dashes / pipes / colons.
                content = stripped.replace("|", "").replace(":", "").strip()
                if content and set(content) <= {"-", " "}:
                    seen_separator = True
                continue
            data_rows.append(stripped)

        self.assertGreaterEqual(
            len(data_rows),
            2,
            "leaderboard.md should contain at least 2 reference rows after the table separator",
        )


class SubmissionDocTests(unittest.TestCase):
    def test_submission_doc_references_cli_commands(self):
        self.assertTrue(SUBMISSION_DOC.exists(), "benchmarks/SUBMISSION.md missing")
        text = SUBMISSION_DOC.read_text(encoding="utf-8")
        for needle in (
            "python -m a2a_cma_cli run-all",
            "python -m a2a_cma_cli report",
        ):
            self.assertIn(
                needle,
                text,
                f"SUBMISSION.md missing literal CLI string: {needle!r}",
            )


class ExampleSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assertTruthy = (lambda *_: None)  # noqa: SLF001 - silence lint
        cls.path = EXAMPLE_JSON
        if not EXAMPLE_JSON.exists():
            raise AssertionError("benchmarks/_example_submission.json missing")
        with EXAMPLE_JSON.open("r", encoding="utf-8") as f:
            cls.payload = json.load(f)

    def test_payload_has_all_required_fields(self):
        for field in REQUIRED_SUBMISSION_FIELDS:
            self.assertIn(
                field,
                self.payload,
                f"_example_submission.json missing required field: {field!r}",
            )

    def test_aggregate_metrics_have_headline_keys_in_range(self):
        agg = self.payload.get("aggregate_metrics", {})
        for key in HEADLINE_METRIC_KEYS:
            self.assertIn(key, agg, f"aggregate_metrics missing {key!r}")
            value = agg[key]
            self.assertIsInstance(
                value, (int, float), f"aggregate_metrics[{key!r}] must be numeric"
            )
            if key == "mean_reward":
                # Reward may be negative when anomaly rates are non-trivial.
                # Just require it is a finite real number.
                import math

                self.assertTrue(
                    math.isfinite(float(value)),
                    f"mean_reward must be finite, got {value!r}",
                )
            else:
                self.assertGreaterEqual(
                    value, 0.0, f"{key} must be in [0, 1], got {value!r}"
                )
                self.assertLessEqual(
                    value, 1.0, f"{key} must be in [0, 1], got {value!r}"
                )

    def test_per_scenario_metrics_cover_v1_active_scenarios(self):
        with SCENARIOS_JSON.open("r", encoding="utf-8") as f:
            scenarios = json.load(f)
        active_ids = {
            s["scenario_id"] for s in scenarios if s.get("v1_active") is True
        }
        per_scenario = self.payload.get("per_scenario_metrics", [])
        self.assertIsInstance(per_scenario, list)
        covered_ids = {row.get("scenario_id") for row in per_scenario}
        missing = active_ids - covered_ids
        extra = covered_ids - active_ids
        self.assertFalse(
            missing,
            f"per_scenario_metrics missing v1_active scenario_ids: {sorted(missing)}",
        )
        self.assertFalse(
            extra,
            f"per_scenario_metrics has unexpected scenario_ids: {sorted(extra)}",
        )


class GitkeepTests(unittest.TestCase):
    def test_gitkeep_exists_and_is_empty(self):
        self.assertTrue(
            SUBMISSIONS_GITKEEP.exists(),
            "benchmarks/submissions/.gitkeep missing",
        )
        size = os.path.getsize(SUBMISSIONS_GITKEEP)
        self.assertEqual(
            size, 0, f"benchmarks/submissions/.gitkeep should be empty, has {size} bytes"
        )


if __name__ == "__main__":
    unittest.main()
