"""Smoke tests for v3 LinUCB contextual bandit + anomaly-typed reward.

Stdlib only. No API calls. Reuses the LanguageModel-stub idiom from
`tests/test_v1_5_payment_gateway.py` so any incidental Conversation imports
do not require network.
"""

from __future__ import annotations

import os
import random
import sys
import types
import unittest
from pathlib import Path

# Make the repo root importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_language_model_stub():
    if "LanguageModel" in sys.modules:
        return

    class _StubLanguageModel:
        def __init__(self, model_name="stub"):
            self.model_name = model_name

        def get_response(self, prompt, temperature=0.7, max_tokens=1000):
            return ""

        def get_chat_response(self, messages, temperature=0.7, max_tokens=1000):
            return ""

    stub_module = types.ModuleType("LanguageModel")
    stub_module.LanguageModel = _StubLanguageModel
    sys.modules["LanguageModel"] = stub_module


_install_language_model_stub()

from rl.contextual_bandit import (  # noqa: E402
    LinUCBPolicy,
    softmax_select_from_ucb,
    _invert,
)
from rl.contextual_features import (  # noqa: E402
    CONTEXT_DIM,
    _archetype_onehot,
    _persona_onehot,
    _price_band_onehot,
    _product_type_onehot,
    featurise,
)
from rl.policy import (  # noqa: E402
    DEFAULT_REWARD_COEFFS,
    reward_from_anomalies,
    reward_from_anomalies_with_breakdown,
)
from scenarios.schema import Scenario, SpendAuthorization, UserPersona  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
SAMPLE_PRODUCT = {
    "id": 4242,
    "Product Name": "Sample OLED TV",
    "Type": "Electronics",
    "Retail Price": "$1899",
    "Wholesale Price": "$1329",
    "Features": "OLED 4K HDR, 120Hz",
}


def _make_scenario(
    archetype: str = "honest_retailer",
    hard_cap: float = 1000.0,
    requires_confirm_above: float = 950.0,
    price_sens: str = "medium",
    urgency: str = "medium",
) -> Scenario:
    return Scenario(
        scenario_id=f"unit_{archetype}",
        service_template="amazon_buy_for_me",
        description="unit-test scenario",
        product_id=SAMPLE_PRODUCT["id"],
        user_persona=UserPersona(
            summary="unit", price_sensitivity=price_sens, urgency=urgency
        ),
        spend_authorization=SpendAuthorization(
            hard_cap=hard_cap,
            per_item_cap=hard_cap,
            requires_confirm_above=requires_confirm_above,
            mandate_id="ap2-unit",
        ),
        seller_archetype=archetype,
    )


# ---------------------------------------------------------------------------
# 1. featurise length parity for Scenario vs episode-dict inputs
# ---------------------------------------------------------------------------
class FeatureLengthTest(unittest.TestCase):
    def test_scenario_and_episode_dict_have_matching_length(self):
        scen = _make_scenario()
        from_scenario = featurise(scen, product=SAMPLE_PRODUCT)

        episode_data = {
            "product_data": SAMPLE_PRODUCT,
            "scenario": {
                "spend_authorization": {
                    "hard_cap": 1000.0,
                    "per_item_cap": 1000.0,
                    "requires_confirm_above": 950.0,
                },
                "user_persona": {
                    "price_sensitivity": "medium",
                    "urgency": "medium",
                },
                "seller_archetype": "honest_retailer",
            },
        }
        from_data = featurise(episode_data)
        self.assertEqual(len(from_scenario), CONTEXT_DIM)
        self.assertEqual(len(from_data), CONTEXT_DIM)
        self.assertEqual(len(from_scenario), len(from_data))


# ---------------------------------------------------------------------------
# 2. one-hots are well-formed
# ---------------------------------------------------------------------------
class OneHotShapeTest(unittest.TestCase):
    def test_persona_one_hot(self):
        for v in ("low", "medium", "high"):
            oh = _persona_onehot(v)
            self.assertEqual(len(oh), 3)
            self.assertAlmostEqual(sum(oh), 1.0)
            for x in oh:
                self.assertIn(x, (0.0, 1.0))
        # Unknown defaults to medium.
        oh = _persona_onehot("garbage")
        self.assertEqual(oh[1], 1.0)

    def test_archetype_one_hot(self):
        for v in (
            "honest_retailer",
            "dark_pattern_marketplace",
            "prompt_injecting",
            "colluding",
            "misrepresenting",
        ):
            oh = _archetype_onehot(v)
            self.assertEqual(len(oh), 5)
            self.assertAlmostEqual(sum(oh), 1.0)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in oh))

    def test_price_band_and_type_one_hot(self):
        bands = _price_band_onehot(1500.0)
        self.assertEqual(len(bands), 4)
        self.assertAlmostEqual(sum(bands), 1.0)
        # 500-2000 bucket should fire (index 1).
        self.assertEqual(bands[1], 1.0)

        for t in ("Vehicle", "Electronics", "Real Estate", "Mystery"):
            oh = _product_type_onehot(t)
            self.assertEqual(len(oh), 4)
            self.assertAlmostEqual(sum(oh), 1.0)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in oh))


# ---------------------------------------------------------------------------
# 3. LinUCB select on untrained model returns a valid arm + finite score
# ---------------------------------------------------------------------------
class LinUCBSmokeTest(unittest.TestCase):
    def test_untrained_select_is_valid(self):
        rng = random.Random(0)
        policy = LinUCBPolicy(num_actions=8, context_dim=CONTEXT_DIM, alpha=1.0, ridge=1.0)
        ctx = [rng.uniform(-1.0, 1.0) for _ in range(CONTEXT_DIM)]
        arm, score = policy.select(ctx)
        self.assertIsInstance(arm, int)
        self.assertGreaterEqual(arm, 0)
        self.assertLess(arm, 8)
        self.assertTrue(score == score)  # not NaN
        self.assertNotEqual(score, float("inf"))
        self.assertNotEqual(score, -float("inf"))

    def test_invert_identity(self):
        I = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
        inv = _invert(I)
        for i in range(5):
            for j in range(5):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(inv[i][j], expected, places=9)

    def test_invert_nontrivial(self):
        # 3x3 with known inverse (diag(2,3,4)).
        M = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        inv = _invert(M)
        self.assertAlmostEqual(inv[0][0], 0.5)
        self.assertAlmostEqual(inv[1][1], 1.0 / 3.0)
        self.assertAlmostEqual(inv[2][2], 0.25)


# ---------------------------------------------------------------------------
# 4. After biased updates, select prefers the rewarded arm
# ---------------------------------------------------------------------------
class LinUCBLearningTest(unittest.TestCase):
    def test_favoured_arm_is_picked_after_biased_updates(self):
        rng = random.Random(1234)
        K = 6
        D = CONTEXT_DIM
        policy = LinUCBPolicy(num_actions=K, context_dim=D, alpha=0.3, ridge=1.0)
        # Build a fixed context that we'll use both for training and for
        # the trial loop; the bandit only needs to learn that arm 3 is
        # the best on this neighbourhood.
        base_ctx = [rng.uniform(0.0, 1.0) for _ in range(D)]
        for _ in range(50):
            jittered = [c + rng.uniform(-0.05, 0.05) for c in base_ctx]
            policy.update(3, jittered, reward=2.0)
            for other in (0, 1, 2, 4, 5):
                policy.update(other, jittered, reward=-0.1)
        hits = 0
        trials = 20
        for _ in range(trials):
            jittered = [c + rng.uniform(-0.05, 0.05) for c in base_ctx]
            arm, _ = policy.select(jittered)
            if arm == 3:
                hits += 1
        self.assertGreaterEqual(
            hits / trials,
            0.7,
            msg=f"only {hits}/{trials} trials picked arm 3",
        )


# ---------------------------------------------------------------------------
# 5. to_dict / from_dict round-trip preserves select output
# ---------------------------------------------------------------------------
class LinUCBSerialiseTest(unittest.TestCase):
    def test_round_trip_preserves_select(self):
        rng = random.Random(42)
        D = CONTEXT_DIM
        policy = LinUCBPolicy(num_actions=4, context_dim=D, alpha=0.5, ridge=1.0)
        for _ in range(30):
            ctx = [rng.uniform(-1.0, 1.0) for _ in range(D)]
            a = rng.randrange(4)
            policy.update(a, ctx, reward=rng.uniform(-1.0, 1.0))
        fixed_ctx = [0.1 * i for i in range(D)]
        arm_before, score_before = policy.select(fixed_ctx)
        d = policy.to_dict()
        restored = LinUCBPolicy.from_dict(d)
        arm_after, score_after = restored.select(fixed_ctx)
        self.assertEqual(arm_before, arm_after)
        self.assertAlmostEqual(score_before, score_after, places=9)

    def test_softmax_select_helper(self):
        # Highest-score arm should win the most often under low temperature.
        scores = [0.1, 0.5, 2.5, 0.2]
        rng = random.Random(0)
        wins = [0, 0, 0, 0]
        for _ in range(500):
            i = softmax_select_from_ucb(scores, temperature=0.5, rng=rng)
            wins[i] += 1
        self.assertEqual(wins.index(max(wins)), 2)


# ---------------------------------------------------------------------------
# 6. reward_from_anomalies on a clean accepted episode -> >= +1.0
# ---------------------------------------------------------------------------
class CleanRewardTest(unittest.TestCase):
    def test_clean_accept_under_mandate_gets_positive_bonus(self):
        anomalies = {
            "overpayment": False,
            "out_of_budget": False,
            "offer_over_first": False,
            "fell_for_dark_pattern": False,
            "accepted_injection_attempt": False,
            "paid_for_misrepresented_item": False,
            "collusion_signal_detected": False,
            "collusion_succeeded": False,
            "gateway_blocked_overpayment": False,
        }
        data = {
            "negotiation_result": "accepted",
            "budget_scenario": "high",
            "completed_turns": 0,  # zero turns -> no turn penalty
        }
        r = reward_from_anomalies(anomalies, data)
        self.assertGreaterEqual(r, 1.0)


# ---------------------------------------------------------------------------
# 7. injection-attempt episode -> reward <= -3.0
# ---------------------------------------------------------------------------
class InjectionPenaltyTest(unittest.TestCase):
    def test_injection_attempt_dominant_penalty(self):
        anomalies = {
            "accepted_injection_attempt": True,
            "overpayment": False,
            "out_of_budget": False,
        }
        data = {
            "negotiation_result": "accepted",
            "budget_scenario": "high",
            "completed_turns": 5,
        }
        r = reward_from_anomalies(anomalies, data)
        self.assertLessEqual(r, -3.0)


# ---------------------------------------------------------------------------
# 8. collusion_succeeded + regret_flagged -> both contribute
# ---------------------------------------------------------------------------
class StackedPenaltiesTest(unittest.TestCase):
    def test_collusion_and_regret_are_both_summed(self):
        anomalies = {
            "collusion_succeeded": True,
            "collusion_signal_detected": True,
            "regret_flagged": True,
        }
        data = {
            "negotiation_result": "accepted",
            "budget_scenario": "high",
            "completed_turns": 0,
        }
        r, breakdown = reward_from_anomalies_with_breakdown(anomalies, data)
        # Both keys should be present.
        self.assertIn("collusion_succeeded", breakdown)
        self.assertIn("regret_flagged", breakdown)
        # And the elevated "_only" version must NOT fire when succeeded did.
        self.assertNotIn("collusion_signal_detected_only", breakdown)
        # Numerical: -3.0 + -2.0 = -5.0 (no other flags fired).
        self.assertAlmostEqual(
            r,
            DEFAULT_REWARD_COEFFS["collusion_succeeded"]
            + DEFAULT_REWARD_COEFFS["regret_flagged"],
            places=9,
        )


# ---------------------------------------------------------------------------
# 9. robust to missing keys (back-compat)
# ---------------------------------------------------------------------------
class BackCompatRewardTest(unittest.TestCase):
    def test_v1_shape_episode_returns_a_finite_number(self):
        # No v1.5 / v2 / v2.5 fields at all.
        anomalies = {
            "overpayment": False,
            "out_of_budget": False,
            "offer_over_first": False,
        }
        data = {
            "negotiation_result": "rejected",
            "budget_scenario": "low",
            "completed_turns": 7,
        }
        r = reward_from_anomalies(anomalies, data)
        self.assertEqual(r, r)  # not NaN
        self.assertTrue(isinstance(r, float))

    def test_legacy_overpayment_high_matches_v1_constants(self):
        # v1 was: -2.0 for overpayment@high, -1.0 if offer_over_first&accepted,
        # and -0.02 per turn.
        anomalies = {
            "overpayment": True,
            "out_of_budget": False,
            "offer_over_first": True,
        }
        data = {
            "negotiation_result": "accepted",
            "budget_scenario": "high",
            "completed_turns": 10,
        }
        r = reward_from_anomalies(anomalies, data)
        # -2.0 (overpayment) + -1.0 (offer_over_first&accepted) + -0.02*10
        self.assertAlmostEqual(r, -2.0 + -1.0 + -0.02 * 10, places=9)

    def test_legacy_out_of_budget_low_matches_v1_constants(self):
        anomalies = {
            "overpayment": False,
            "out_of_budget": True,
            "offer_over_first": False,
        }
        data = {
            "negotiation_result": "accepted",
            "budget_scenario": "low",
            "completed_turns": 4,
        }
        r = reward_from_anomalies(anomalies, data)
        self.assertAlmostEqual(r, -1.0 + -0.02 * 4, places=9)

    def test_legacy_deadlock_matches_v1_constants(self):
        anomalies = {
            "overpayment": False,
            "out_of_budget": False,
            "offer_over_first": False,
        }
        data = {
            "negotiation_result": "max_turns_reached",
            "budget_scenario": "high",
            "completed_turns": 20,
        }
        r = reward_from_anomalies(anomalies, data)
        # -1.0 (deadlock) + -0.02*20
        self.assertAlmostEqual(r, -1.0 + -0.02 * 20, places=9)


# ---------------------------------------------------------------------------
# 10. run_episode integration smoke (stub Conversation to avoid LM calls)
# ---------------------------------------------------------------------------
class RunEpisodeIntegrationTest(unittest.TestCase):
    def test_run_episode_uses_v3_reward_and_exposes_breakdown(self):
        # Patch RLConversation in rl.env to avoid real LM calls.
        os.chdir(REPO_ROOT)
        import rl.env as env_mod

        class _StubConv:
            def __init__(self, *args, **kwargs):
                self.product_data = kwargs.get("product_data") or args[0]
                self.budget = kwargs.get("budget", 1000.0)
                self.conversation_history = [
                    {"speaker": "Buyer", "message": "hi"},
                    {"speaker": "Seller", "message": "Best $900"},
                    {"speaker": "Buyer", "message": "deal"},
                ]
                self.seller_price_offers = [1000.0, 950.0, 900.0]
                self.completed_turns = 3
                self.negotiation_completed = True
                self.negotiation_result = "accepted"
                self.current_price_offer = 900.0
                self.budget_scenario = None

            def run_negotiation(self):
                return None

        original = env_mod.RLConversation
        env_mod.RLConversation = _StubConv
        try:
            reward, info, data = env_mod.run_episode(
                product=SAMPLE_PRODUCT,
                budget_scenario="high",
                action_idx=0,
                buyer_model="stub",
                seller_model="stub",
                summary_model="stub",
                max_turns=20,
            )
        finally:
            env_mod.RLConversation = original

        self.assertIn("reward_breakdown", info)
        self.assertIsInstance(info["reward_breakdown"], dict)
        self.assertAlmostEqual(reward, sum(info["reward_breakdown"].values()), places=9)


if __name__ == "__main__":
    unittest.main()
