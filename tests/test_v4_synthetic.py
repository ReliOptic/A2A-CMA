"""Smoke tests for the v4 synthetic-episode generator and v2 safety notebook.

Stdlib only. No API calls. No notebook execution. Mirrors the import-shim
idiom from the existing v2 tests so MarkAnomaly stays importable in stdlib-
only environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_language_model_stub() -> None:
    """Inject a no-network LanguageModel stub before any imports that need it."""
    if "LanguageModel" in sys.modules:
        return

    class _StubLanguageModel:
        def __init__(self, model_name: str = "stub") -> None:
            self.model_name = model_name

        def get_response(self, prompt, temperature=0.7, max_tokens=1000):
            return ""

        def get_chat_response(self, messages, temperature=0.7, max_tokens=1000):
            return ""

    stub_module = types.ModuleType("LanguageModel")
    stub_module.LanguageModel = _StubLanguageModel
    sys.modules["LanguageModel"] = stub_module


_install_language_model_stub()


from MarkAnomaly import (  # noqa: E402
    COLLUSION_MARKERS,
    INJECTION_MARKERS,
    PostDataProcessor,
)
from tools.generate_synthetic_episodes import generate  # noqa: E402


REQUIRED_KEYS = {
    "scenario",
    "seller_archetype",
    "payment_gateway",
    "seller_price_offers",
    "negotiation_result",
    "conversation_history",
}


class _GeneratorFixture(unittest.TestCase):
    """Mixin that materialises a synthetic corpus once per TestCase."""

    n_per_scenario = 2
    seed = 0

    @classmethod
    def setUpClass(cls) -> None:  # noqa: D401
        cls._tmp = tempfile.mkdtemp(prefix="a2a_cma_v4_")
        cls.paths = generate(cls._tmp, n_per_scenario=cls.n_per_scenario, seed=cls.seed)

    @classmethod
    def tearDownClass(cls) -> None:  # noqa: D401
        shutil.rmtree(cls._tmp, ignore_errors=True)


class GenerateProducesExpectedFileCount(_GeneratorFixture):
    def test_eighteen_files_two_per_scenario(self) -> None:
        # 9 active scenarios (asserted from dataset/scenarios.json) * 2 = 18.
        with open(REPO_ROOT / "dataset" / "scenarios.json") as f:
            active_count = len(json.load(f))
        self.assertEqual(active_count, 9, "scenarios.json should have 9 entries")
        self.assertEqual(len(self.paths), 18)

    def test_every_path_is_loadable_json(self) -> None:
        for path in self.paths:
            with self.subTest(path=path):
                with open(path) as f:
                    data = json.load(f)
                self.assertIsInstance(data, dict)


class GeneratedEpisodesHaveExpectedKeys(_GeneratorFixture):
    def test_required_keys_present(self) -> None:
        for path in self.paths:
            with self.subTest(path=path):
                with open(path) as f:
                    data = json.load(f)
                missing = REQUIRED_KEYS - set(data.keys())
                self.assertFalse(missing, f"missing keys in {path}: {missing}")


class PromptInjectingEmbedsMarker(_GeneratorFixture):
    def test_at_least_one_seller_turn_carries_an_injection_marker(self) -> None:
        injecting_episodes = []
        for path in self.paths:
            with open(path) as f:
                data = json.load(f)
            if data.get("seller_archetype") == "prompt_injecting":
                injecting_episodes.append(data)

        self.assertGreater(len(injecting_episodes), 0)
        for ep in injecting_episodes:
            seller_msgs = [
                t.get("message", "")
                for t in ep.get("conversation_history", [])
                if isinstance(t, dict) and t.get("speaker") == "Seller"
            ]
            with self.subTest(scenario=ep.get("scenario", {}).get("scenario_id")):
                self.assertTrue(
                    any(marker in m for m in seller_msgs for marker in INJECTION_MARKERS),
                    f"no INJECTION_MARKERS found in seller turns: {seller_msgs!r}",
                )


class ColludingEmbedsMarker(_GeneratorFixture):
    def test_at_least_one_seller_turn_carries_a_collusion_marker(self) -> None:
        colluding_episodes = []
        for path in self.paths:
            with open(path) as f:
                data = json.load(f)
            if data.get("seller_archetype") == "colluding":
                colluding_episodes.append(data)

        self.assertGreater(len(colluding_episodes), 0)
        for ep in colluding_episodes:
            seller_msgs = [
                t.get("message", "")
                for t in ep.get("conversation_history", [])
                if isinstance(t, dict) and t.get("speaker") == "Seller"
            ]
            with self.subTest(scenario=ep.get("scenario", {}).get("scenario_id")):
                self.assertTrue(
                    any(marker in m for m in seller_msgs for marker in COLLUSION_MARKERS),
                    f"no COLLUSION_MARKERS found in seller turns: {seller_msgs!r}",
                )


class CalculateAnomaliesRunsOnEveryEpisode(_GeneratorFixture):
    EXPECTED_ANOMALY_KEYS = (
        "regret_flagged",
        "gateway_intervention_required",
        "accepted_injection_attempt",
        "fell_for_dark_pattern",
        "collusion_signal_detected",
        "paid_for_misrepresented_item",
        "overpayment",
    )

    def test_every_episode_yields_anomalies_dict(self) -> None:
        proc = PostDataProcessor()
        for path in self.paths:
            with open(path) as f:
                data = json.load(f)
            with self.subTest(path=path):
                anomalies = proc.calculate_anomalies(data)
                self.assertIsInstance(anomalies, dict)
                for key in self.EXPECTED_ANOMALY_KEYS:
                    self.assertIn(key, anomalies, f"missing anomaly key {key!r}")


class NotebookIsValidNbformat(unittest.TestCase):
    NB_PATH = REPO_ROOT / "data_postprocess" / "v2_safety_analysis.ipynb"

    def test_notebook_loads_and_has_cells(self) -> None:
        self.assertTrue(self.NB_PATH.exists(), f"missing notebook: {self.NB_PATH}")
        with self.NB_PATH.open() as f:
            nb = json.load(f)
        self.assertIn("cells", nb)
        cells = nb["cells"]
        self.assertIsInstance(cells, list)
        self.assertGreater(len(cells), 0)
        for cell in cells:
            self.assertIn("cell_type", cell)
            self.assertIn(cell["cell_type"], {"code", "markdown"})

    def test_at_least_one_code_cell_imports_markanomaly(self) -> None:
        with self.NB_PATH.open() as f:
            nb = json.load(f)
        importing = []
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            text = "".join(source) if isinstance(source, list) else str(source)
            if "MarkAnomaly" in text:
                importing.append(cell)
        self.assertGreater(
            len(importing), 0,
            "expected at least one code cell to import MarkAnomaly",
        )


class GeneratorIsDeterministic(unittest.TestCase):
    def test_same_seed_produces_byte_identical_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="a2a_cma_v4_a_") as a, \
                tempfile.TemporaryDirectory(prefix="a2a_cma_v4_b_") as b:
            paths_a = generate(a, n_per_scenario=2, seed=0)
            paths_b = generate(b, n_per_scenario=2, seed=0)
            self.assertEqual(len(paths_a), len(paths_b))

            def _hash(path: str) -> str:
                with open(path, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()

            # Compare by relative path so absolute tmpdir prefixes don't matter.
            rel_a = sorted(os.path.relpath(p, a) for p in paths_a)
            rel_b = sorted(os.path.relpath(p, b) for p in paths_b)
            self.assertEqual(rel_a, rel_b)
            for rel in rel_a:
                with self.subTest(rel=rel):
                    self.assertEqual(_hash(os.path.join(a, rel)),
                                     _hash(os.path.join(b, rel)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
