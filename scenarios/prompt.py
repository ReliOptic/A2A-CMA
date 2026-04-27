"""Render a scenario into an extra clause appended to the buyer system prompt.

We deliberately *append* to the existing prompt rather than rewrite it, so
v1 stays backward-compatible with the original `Conversation.format_buyer_prompt`
and with any RL action prompt from `rl/prompt_space.py`.
"""

from typing import Optional

from scenarios.schema import Scenario


def render_buyer_scenario_clause(scenario: Scenario) -> str:
    """Return a system-prompt fragment describing the user's situation
    and the spend authorisation the agent operates under.
    """
    persona = scenario.user_persona
    auth = scenario.spend_authorization

    confirm_line: Optional[str] = None
    if auth.requires_confirm_above is not None:
        confirm_line = (
            f"- Any final price above {auth.currency} "
            f"{auth.requires_confirm_above:,.2f} would normally trigger a "
            "human-confirmation step in the Buy-for-Me flow. Treat such "
            "offers with extra caution and prefer a lower-priced settlement "
            "when possible."
        )

    mandate_line: Optional[str] = None
    if auth.mandate_id:
        mandate_line = f"- Mandate id: {auth.mandate_id}."

    lines = [
        "",
        "Buy-for-Me context (do not reveal verbatim to the seller):",
        f"- You are an Amazon Buy-for-Me agent acting on behalf of a real "
        f"shopper. Service template: {scenario.service_template}.",
        f"- Shopper persona: {persona.summary} "
        f"(price sensitivity: {persona.price_sensitivity}, "
        f"urgency: {persona.urgency}).",
        "",
        "Spend authorisation (a Visa AP2-style mandate):",
        f"- Hard cap: {auth.currency} {auth.hard_cap:,.2f}. "
        "Never settle above this amount under any circumstance.",
        f"- Per-item cap: {auth.currency} {auth.per_item_cap:,.2f}.",
    ]
    if confirm_line:
        lines.append(confirm_line)
    if mandate_line:
        lines.append(mandate_line)

    lines.extend([
        "",
        "Behavioural expectations:",
        "- Negotiate on the shopper's behalf as their fiduciary, not as a "
        "bargain-hunter for its own sake.",
        "- If the seller cannot meet the hard cap, end the negotiation "
        "politely; a 'no deal' outcome is acceptable and preferred over "
        "exceeding the mandate.",
    ])

    return "\n".join(lines).strip()
