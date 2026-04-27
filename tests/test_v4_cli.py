"""Smoke tests for the v4 a2a-cma CLI.

Stdlib only. Reuses the LanguageModel-stub idiom from the existing v1.5 / v2
test suites so no real LLM calls are made and no API keys are required.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


# Make the repo root importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# LanguageModel stub. Drop-in for the real LM, scripted-reply aware so a
# Conversation can be driven through ``run_negotiation`` without ever touching
# the network.
# ---------------------------------------------------------------------------

def _install_language_model_stub():
    """Install a no-network LanguageModel stub before Conversation imports it."""
    if "LanguageModel" in sys.modules and getattr(
        sys.modules["LanguageModel"], "_a2a_cma_v4_stub", False
    ):
        return

    class _StubLanguageModel:
        # Class-level so individual test cases can configure the chat replies.
        chat_replies = []
        response_replies = []

        def __init__(self, model_name="stub"):
            self.model_name = model_name

        def get_response(self, prompt, temperature=0.7, max_tokens=1000):
            # Special-case the price-extraction prompt so the negotiation loop
            # picks up a deterministic price below the iPhone scenario's hard
            # cap (1000) and the negotiation-state prompt so it terminates.
            if "STRICLY ONLY return the price with $ symbol" in prompt:
                return "$900"
            if "ACCEPTANCE, REJECTION, or CONTINUE" in prompt:
                return "ACCEPTANCE"
            if _StubLanguageModel.response_replies:
                return _StubLanguageModel.response_replies.pop(0)
            # Default: a short canned message; sufficient for the buyer's
            # opening line.
            return "Hello, I'm interested in this product."

        def get_chat_response(self, messages, temperature=0.7, max_tokens=1000):
            if _StubLanguageModel.chat_replies:
                return _StubLanguageModel.chat_replies.pop(0)
            # Generic fallback alternating reply.
            return "Sure, I can offer it for $900."

    stub_module = types.ModuleType("LanguageModel")
    stub_module.LanguageModel = _StubLanguageModel
    stub_module._a2a_cma_v4_stub = True
    sys.modules["LanguageModel"] = stub_module


_install_language_model_stub()


from a2a_cma_cli import cli as cli_mod  # noqa: E402
import argparse  # noqa: E402


# ---------------------------------------------------------------------------
# 1. list-scenarios
# ---------------------------------------------------------------------------

class ListScenariosTest(unittest.TestCase):
    def setUp(self):
        os.chdir(REPO_ROOT)

    def _run(self, archetype=None):
        ns = argparse.Namespace(
            archetype=archetype,
            scenarios_path="dataset/scenarios.json",
        )
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_list_scenarios(ns)
        return rc, buf.getvalue()

    def test_no_filter_lists_all_nine_active_scenarios(self):
        rc, output = self._run()
        self.assertEqual(rc, 0)
        # Header + separator + 9 active rows.
        body_lines = [ln for ln in output.splitlines() if ln.strip()]
        # Subtract 2 for header + separator.
        self.assertEqual(len(body_lines) - 2, 9, msg=output)

    def test_archetype_filter_prompt_injecting_returns_one(self):
        rc, output = self._run(archetype="prompt_injecting")
        self.assertEqual(rc, 0)
        body_lines = [ln for ln in output.splitlines() if ln.strip()]
        self.assertEqual(len(body_lines) - 2, 1, msg=output)
        self.assertIn("bfm_oled_tv_injection_attempt", output)

    def test_unknown_archetype_returns_nonzero(self):
        rc, _ = self._run(archetype="does_not_exist")
        self.assertNotEqual(rc, 0)

    def test_missing_scenarios_file_returns_nonzero(self):
        ns = argparse.Namespace(
            archetype=None,
            scenarios_path=str(REPO_ROOT / "definitely_not_a_real_path.json"),
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_list_scenarios(ns)
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# 2. + 3. run / --no-gateway
# ---------------------------------------------------------------------------

class RunCommandTest(unittest.TestCase):
    def setUp(self):
        os.chdir(REPO_ROOT)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.outdir = self._tmp.name
        # Reset the canned replies so each test starts from a clean slate.
        stub = sys.modules["LanguageModel"].LanguageModel
        stub.chat_replies = []
        stub.response_replies = []

    def _run(self, no_gateway=False):
        ns = argparse.Namespace(
            scenario="bfm_iphone_strict_gift",
            buyer_model="stub",
            seller_model="stub",
            summary_model="stub",
            max_turns=4,
            output=self.outdir,
            repeats=1,
            no_gateway=no_gateway,
            scenarios_path="dataset/scenarios.json",
            products_path="dataset/products.json",
        )
        # Silence noisy stdout from Conversation.run_negotiation.
        with redirect_stdout(io.StringIO()):
            return cli_mod.cmd_run(ns)

    def test_writes_one_episode_with_required_keys(self):
        rc = self._run(no_gateway=False)
        self.assertEqual(rc, 0)
        files = [f for f in os.listdir(self.outdir) if f.endswith(".json")]
        self.assertEqual(len(files), 1, msg=files)
        with open(os.path.join(self.outdir, files[0]), "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("scenario", "seller_archetype", "payment_gateway", "last_authorization"):
            self.assertIn(key, data)
        # Sanity: scenario was the iPhone scenario, archetype defaults to honest_retailer
        self.assertEqual(data["scenario"]["scenario_id"], "bfm_iphone_strict_gift")
        self.assertEqual(data["seller_archetype"], "honest_retailer")
        # The auto-built gateway should be present and non-null.
        self.assertIsNotNone(data["payment_gateway"])

    def test_no_gateway_zeroes_payment_gateway_field(self):
        rc = self._run(no_gateway=True)
        self.assertEqual(rc, 0)
        files = [f for f in os.listdir(self.outdir) if f.endswith(".json")]
        self.assertEqual(len(files), 1, msg=files)
        with open(os.path.join(self.outdir, files[0]), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsNone(data["payment_gateway"])
        self.assertIsNone(data["last_authorization"])

    def test_unknown_scenario_returns_nonzero(self):
        ns = argparse.Namespace(
            scenario="this_does_not_exist",
            buyer_model="stub",
            seller_model="stub",
            summary_model="stub",
            max_turns=4,
            output=self.outdir,
            repeats=1,
            no_gateway=True,
            scenarios_path="dataset/scenarios.json",
            products_path="dataset/products.json",
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_run(ns)
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# 4. report
# ---------------------------------------------------------------------------

def _synthetic_episode(
    product_id: int,
    scenario_id: str = "synthetic_scenario",
    archetype: str = "honest_retailer",
    accepted: bool = True,
    final_price: float = 900.0,
    first_price: float = 1000.0,
    budget: float = 1000.0,
):
    return {
        "product_id": product_id,
        "experiment_num": 0,
        "product_data": {
            "id": product_id,
            "Product Name": "Synthetic Widget",
            "Type": "Widget",
            "Retail Price": f"${first_price:.0f}",
            "Wholesale Price": "$500",
            "Features": "synthetic",
        },
        "conversation_history": [
            {"speaker": "Buyer", "message": "hi"},
            {"speaker": "Seller", "message": f"price ${final_price:.0f}"},
            {"speaker": "Buyer", "message": "ok"},
        ],
        "seller_price_offers": [first_price, final_price],
        "budget": budget,
        "budget_scenario": "retail",
        "completed_turns": 1,
        "negotiation_completed": True,
        "negotiation_result": "accepted" if accepted else "rejected",
        "models": {"buyer": "stub", "seller": "stub", "summary": "stub"},
        "parameters": {"max_turns": 20},
        "scenario": {
            "scenario_id": scenario_id,
            "spend_authorization": {
                "hard_cap": budget,
                "per_item_cap": budget,
                "requires_confirm_above": budget * 0.95,
                "currency": "USD",
            },
            "seller_archetype": archetype,
        },
        "payment_gateway": None,
        "last_authorization": None,
        "gateway_decline_reason": None,
        "seller_archetype": archetype,
    }


class ReportCommandTest(unittest.TestCase):
    def setUp(self):
        os.chdir(REPO_ROOT)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.outdir = self._tmp.name

        # Two synthetic episodes: one clean accept, one rejected-under-budget.
        ep1 = _synthetic_episode(product_id=42)
        ep2 = _synthetic_episode(
            product_id=43,
            scenario_id="other_synth",
            archetype="dark_pattern_marketplace",
            accepted=False,
        )
        with open(os.path.join(self.outdir, "product_42_exp_0.json"), "w") as f:
            json.dump(ep1, f)
        with open(os.path.join(self.outdir, "product_43_exp_0.json"), "w") as f:
            json.dump(ep2, f)

    def test_no_regret_aggregates_and_writes_report_json(self):
        ns = argparse.Namespace(
            output_dir=self.outdir,
            judge_model="gpt-4o-mini",
            no_regret=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_report(ns)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Anomaly incidence", out)
        self.assertIn("Mandate metrics", out)
        # report.json must exist with consistent shape.
        report_path = os.path.join(self.outdir, "report.json")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["total_episodes"], 2)
        self.assertIn("anomaly_incidence", report)
        self.assertIn("mandate_metrics", report)
        self.assertIn("reward_summary", report)
        # --no-regret => no judge attempted, so no regret_summary
        self.assertIsNone(report["regret_summary"])

    def test_empty_directory_returns_nonzero(self):
        empty_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.rmdir(empty_dir))
        ns = argparse.Namespace(
            output_dir=empty_dir,
            judge_model="gpt-4o-mini",
            no_regret=True,
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_report(ns)
        self.assertEqual(rc, 1)

    def test_missing_directory_returns_nonzero(self):
        ns = argparse.Namespace(
            output_dir=str(REPO_ROOT / "definitely_not_a_dir_xyz"),
            judge_model="gpt-4o-mini",
            no_regret=True,
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_report(ns)
        self.assertEqual(rc, 2)

    def test_malformed_json_does_not_crash(self):
        # Drop a malformed JSON in alongside the synthetic episodes.
        with open(os.path.join(self.outdir, "broken.json"), "w") as f:
            f.write("{not json")
        ns = argparse.Namespace(
            output_dir=self.outdir,
            judge_model="gpt-4o-mini",
            no_regret=True,
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = cli_mod.cmd_report(ns)
        self.assertEqual(rc, 0)
        with open(os.path.join(self.outdir, "report.json")) as f:
            report = json.load(f)
        self.assertEqual(report["total_episodes"], 2)
        self.assertEqual(report["malformed_episodes"], 1)


# ---------------------------------------------------------------------------
# 5. python -m a2a_cma_cli list-scenarios via subprocess
# ---------------------------------------------------------------------------

class SubprocessSmokeTest(unittest.TestCase):
    def test_python_dash_m_list_scenarios_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "a2a_cma_cli", "list-scenarios"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr.decode())
        self.assertGreater(len(proc.stdout), 0)


if __name__ == "__main__":
    unittest.main()
