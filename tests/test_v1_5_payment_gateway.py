"""Smoke tests for the v1.5 AP2-style PaymentGateway.

Covers the schema round-trip, gateway decision matrix, log fidelity, and the
Conversation integration path. The Conversation tests stub `LanguageModel`
*before* importing `Conversation` so no network calls are made and no API
keys are required.
"""

import os
import sys
import types
import unittest
from pathlib import Path

# Make the repo root importable when the test is invoked from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_language_model_stub():
    """Inject a no-network LanguageModel stub before Conversation imports it."""
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

from payments import (  # noqa: E402
    AuthorizationOutcome,
    IntentMandate,
    PaymentGateway,
    SharedPaymentToken,
    build_intent_mandate_from_scenario,
)
from scenarios.loader import get_scenario, attach_product  # noqa: E402


def _frozen_clock():
    """Deterministic ISO timestamp for reproducibility in assertions."""
    return "2026-04-27T00:00:00+00:00"


def _build_mandate(
    hard_cap=1000.0,
    per_item_cap=1000.0,
    requires_confirm_above=950.0,
    allowed_categories=None,
):
    return IntentMandate(
        mandate_id="ap2-mock-test",
        currency="USD",
        hard_cap=hard_cap,
        per_item_cap=per_item_cap,
        requires_confirm_above=requires_confirm_above,
        allowed_categories=allowed_categories or [],
        issued_at=_frozen_clock(),
    )


class IntentMandateRoundTripTest(unittest.TestCase):
    def test_to_dict_preserves_key_fields(self):
        mandate = _build_mandate(allowed_categories=["Phone"])
        as_dict = mandate.to_dict()
        self.assertEqual(as_dict["mandate_id"], "ap2-mock-test")
        self.assertEqual(as_dict["currency"], "USD")
        self.assertEqual(as_dict["hard_cap"], 1000.0)
        self.assertEqual(as_dict["per_item_cap"], 1000.0)
        self.assertEqual(as_dict["requires_confirm_above"], 950.0)
        self.assertEqual(as_dict["allowed_categories"], ["Phone"])
        self.assertEqual(as_dict["issued_at"], _frozen_clock())


class PaymentGatewayAuthorizeTest(unittest.TestCase):
    def test_approves_under_all_caps_and_issues_token(self):
        gateway = PaymentGateway(_build_mandate(), clock=_frozen_clock)
        decision = gateway.authorize(
            amount=500.0,
            line_items=[{"name": "Widget", "unit_price": 500.0, "qty": 1}],
        )
        self.assertEqual(decision.outcome, AuthorizationOutcome.APPROVED)
        self.assertIsInstance(decision.token, SharedPaymentToken)
        self.assertEqual(decision.token.amount, 500.0)
        self.assertEqual(decision.token.currency, "USD")
        self.assertEqual(decision.token.mandate_id, "ap2-mock-test")

    def test_needs_human_confirm_above_threshold_below_hard_cap(self):
        gateway = PaymentGateway(_build_mandate(), clock=_frozen_clock)
        decision = gateway.authorize(
            amount=975.0,
            line_items=[{"name": "Widget", "unit_price": 975.0, "qty": 1}],
        )
        self.assertEqual(decision.outcome, AuthorizationOutcome.NEEDS_HUMAN_CONFIRM)
        self.assertIsNone(decision.token)

    def test_declines_when_amount_exceeds_hard_cap(self):
        gateway = PaymentGateway(_build_mandate(), clock=_frozen_clock)
        decision = gateway.authorize(
            amount=1100.0,
            line_items=[{"name": "Widget", "unit_price": 1100.0, "qty": 1}],
        )
        # Per-item cap also fires here, but since per_item check runs before
        # hard_cap, simulate a hard_cap-only breach by using qty math that
        # keeps each line under the per-item cap.
        gateway2 = PaymentGateway(
            _build_mandate(hard_cap=1000.0, per_item_cap=600.0),
            clock=_frozen_clock,
        )
        decision2 = gateway2.authorize(
            amount=1100.0,
            line_items=[
                {"name": "A", "unit_price": 550.0, "qty": 1},
                {"name": "B", "unit_price": 550.0, "qty": 1},
            ],
        )
        self.assertEqual(decision2.outcome, AuthorizationOutcome.DECLINED_OVER_HARD_CAP)
        # And the single-line case at least surfaces a decline (either hard
        # cap or per-item cap depending on order; both are valid blocks).
        self.assertIn(
            decision.outcome,
            {
                AuthorizationOutcome.DECLINED_OVER_HARD_CAP,
                AuthorizationOutcome.DECLINED_OVER_PER_ITEM_CAP,
            },
        )

    def test_decision_log_records_every_call(self):
        gateway = PaymentGateway(_build_mandate(), clock=_frozen_clock)
        gateway.authorize(amount=100.0, line_items=[{"name": "x", "unit_price": 100.0, "qty": 1}])
        gateway.authorize(amount=975.0, line_items=[{"name": "y", "unit_price": 975.0, "qty": 1}])
        gateway.authorize(amount=2000.0, line_items=[{"name": "z", "unit_price": 2000.0, "qty": 1}])
        self.assertEqual(len(gateway.decision_log), 3)
        outcomes = [entry["outcome"] for entry in gateway.decision_log]
        self.assertEqual(
            outcomes,
            [
                AuthorizationOutcome.APPROVED.value,
                AuthorizationOutcome.NEEDS_HUMAN_CONFIRM.value,
                # 2000 trips per_item cap (1000) before hard_cap, so accept
                # either decline label here.
                gateway.decision_log[2]["outcome"],
            ],
        )
        self.assertIn(
            gateway.decision_log[2]["outcome"],
            {
                AuthorizationOutcome.DECLINED_OVER_HARD_CAP.value,
                AuthorizationOutcome.DECLINED_OVER_PER_ITEM_CAP.value,
            },
        )
        for entry in gateway.decision_log:
            self.assertEqual(entry["timestamp"], _frozen_clock())


class ConversationGatewayIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Resolve the bfm_iphone_strict_gift scenario with its product.
        os.chdir(REPO_ROOT)
        scenario = get_scenario("bfm_iphone_strict_gift")
        attach_product(scenario)
        self.scenario = scenario

    def test_gateway_declines_when_offer_exceeds_hard_cap(self):
        from Conversation import Conversation  # imported after stub install

        convo = Conversation(
            product_data=self.scenario.product,
            scenario=self.scenario,
        )
        # The gateway must auto-build from the scenario.
        self.assertIsNotNone(convo.gateway)
        self.assertEqual(convo.gateway.mandate.hard_cap, 1000.0)

        convo.conversation_history = [
            {"speaker": "Buyer", "message": "Hi, interested in this."},
            {"speaker": "Seller", "message": "Best I can do is $1100."},
            {"speaker": "Buyer", "message": "Deal, I'll take it."},
        ]
        convo.current_price_offer = 1100.0
        convo._finalize_acceptance()

        self.assertEqual(convo.negotiation_result, "gateway_declined")
        self.assertTrue(convo.negotiation_completed)
        self.assertIsNotNone(convo.last_authorization)
        self.assertIn(
            convo.gateway_decline_reason,
            {
                AuthorizationOutcome.DECLINED_OVER_HARD_CAP.value,
                AuthorizationOutcome.DECLINED_OVER_PER_ITEM_CAP.value,
            },
        )

    def test_gateway_needs_human_confirm_in_step_up_band(self):
        from Conversation import Conversation

        convo = Conversation(
            product_data=self.scenario.product,
            scenario=self.scenario,
        )
        # 975 is above requires_confirm_above (950) but under hard_cap (1000).
        convo.conversation_history = [
            {"speaker": "Buyer", "message": "Hi."},
            {"speaker": "Seller", "message": "$975 final."},
            {"speaker": "Buyer", "message": "Sounds good."},
        ]
        convo.current_price_offer = 975.0
        convo._finalize_acceptance()

        self.assertEqual(convo.negotiation_result, "needs_human_confirm")
        self.assertTrue(convo.negotiation_completed)
        self.assertEqual(
            convo.gateway_decline_reason,
            AuthorizationOutcome.NEEDS_HUMAN_CONFIRM.value,
        )


class ConversationBackwardCompatibilityTest(unittest.TestCase):
    def test_no_scenario_no_gateway_keeps_baseline_behaviour(self):
        from Conversation import Conversation

        product = {
            "id": 999,
            "Product Name": "Test Widget",
            "Retail Price": "$100",
            "Wholesale Price": "$60",
            "Features": "n/a",
        }
        convo = Conversation(product_data=product, budget=200)
        self.assertIsNone(convo.gateway)
        self.assertIsNone(convo.scenario)
        self.assertIsNone(convo.last_authorization)
        self.assertIsNone(convo.gateway_decline_reason)

        # Drive _finalize_acceptance directly to confirm the baseline path
        # ("accepted") is preserved when no gateway is wired in.
        convo.current_price_offer = 80.0
        convo._finalize_acceptance()
        self.assertEqual(convo.negotiation_result, "accepted")
        self.assertTrue(convo.negotiation_completed)


class IntegrationHelperTest(unittest.TestCase):
    def test_build_intent_mandate_from_scenario_uses_mandate_id(self):
        os.chdir(REPO_ROOT)
        scenario = get_scenario("bfm_iphone_strict_gift")
        mandate = build_intent_mandate_from_scenario(scenario)
        self.assertEqual(mandate.mandate_id, "ap2-mock-iphone-001")
        self.assertEqual(mandate.hard_cap, 1000.0)
        self.assertEqual(mandate.per_item_cap, 1000.0)
        self.assertEqual(mandate.requires_confirm_above, 950.0)
        self.assertEqual(mandate.currency, "USD")


if __name__ == "__main__":
    unittest.main()
