"""Glue between the v1 scenario layer and the v1.5 payment gateway."""

from datetime import datetime, timezone

from payments.schema import IntentMandate


def build_intent_mandate_from_scenario(scenario) -> IntentMandate:
    """Translate a `Scenario.spend_authorization` into an `IntentMandate`.

    Scenario-level spend authorisations already carry hard cap / per-item cap /
    confirm threshold semantics; here we just lift them into the mandate
    object the gateway operates on. If the scenario didn't pin a mandate id we
    derive a stable one from `scenario_id` so episodes stay reproducible.
    """
    auth = scenario.spend_authorization
    mandate_id = auth.mandate_id or f"ap2-mock-auto-{scenario.scenario_id}"
    return IntentMandate(
        mandate_id=mandate_id,
        currency=auth.currency,
        hard_cap=float(auth.hard_cap),
        per_item_cap=float(auth.per_item_cap if auth.per_item_cap is not None else auth.hard_cap),
        requires_confirm_above=(
            float(auth.requires_confirm_above)
            if auth.requires_confirm_above is not None
            else None
        ),
        allowed_categories=[],
        issued_at=datetime.now(tz=timezone.utc).isoformat(),
    )
