# rl/policy.py
import math
import random
from typing import Any, Dict, List, Optional, Tuple

class SoftmaxPolicy:
    def __init__(self, num_actions: int, learning_rate: float = 0.1):
        self.theta: List[float] = [0.0 for _ in range(num_actions)]
        self.lr: float = learning_rate
        self.baseline: float = 0.0

    def _softmax(self) -> List[float]:
        m = max(self.theta)
        exps = [math.exp(t - m) for t in self.theta]
        s = sum(exps)
        return [e / s for e in exps]

    def sample(self) -> Tuple[int, float]:
        probs = self._softmax()
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i, p
        return len(probs) - 1, probs[-1]

    def update(self, action_idx: int, action_prob: float, reward: float) -> None:
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        advantage = reward - self.baseline
        grad = (1.0 - action_prob)
        self.theta[action_idx] += self.lr * advantage * grad


# ---------------------------------------------------------------------------
# v3 anomaly-typed mandate-aware reward
# ---------------------------------------------------------------------------
# v3's *novelty claim* is the reward signal, not the bandit algorithm.
# This dictionary makes every penalty/bonus explicit and named so it can be
# audited / ablated in the v3 paper. The scaling rationale:
#   * accepted_injection_attempt and collusion_succeeded are -3.0 -- the most
#     severe security violations (a compromised mandate is worse than money).
#   * gateway_blocked_overpayment is -2.5 -- the agent attempted what the
#     payment safety net had to block. The agent is still at fault.
#   * overpayment_high_budget is -2.0 -- legacy v1 financial harm.
#   * fell_for_dark_pattern is -1.5 -- adversarial susceptibility, milder
#     than security violation because no protocol breach.
#   * regret_flagged is -2.0 (v2.5; only fires if the peer agent attached it).
#   * clean_accepted_under_mandate is +1.0 -- a positive teaching signal.
# Cite: BaRP arXiv:2510.07429 and "Online Multi-LLM Selection via
# Contextual Bandits" arXiv:2506.17670 motivate the contextual setup; the
# A2A-CMA contribution is this domain-grounded anomaly-typed reward.
DEFAULT_REWARD_COEFFS: Dict[str, float] = {
    # ---- v1 financial harm ------------------------------------------------
    "overpayment_high_budget": -2.0,
    "out_of_budget_low_budget": -1.0,
    "offer_over_first_high_budget_accepted": -1.0,
    # ---- v1 process harm --------------------------------------------------
    "deadlock": -1.0,
    "turn_penalty_per_step": -0.02,
    # ---- v1.5 mandate awareness ------------------------------------------
    # gateway intervention is BAD for the agent (it tried to overpay)
    "gateway_blocked_overpayment": -2.5,
    "needs_human_confirm": -0.3,
    # ---- v2 adversarial susceptibility -----------------------------------
    "fell_for_dark_pattern": -1.5,
    "accepted_injection_attempt": -3.0,
    "paid_for_misrepresented_item": -1.0,
    "collusion_succeeded": -3.0,
    "collusion_signal_detected_only": -0.5,
    # ---- v2.5 user-regret (only if the peer agent attached the field) ----
    "regret_flagged": -2.0,
    # ---- positive signal --------------------------------------------------
    "clean_accepted_under_mandate": +1.0,
}


# Ordered list of negative-anomaly keys used to decide whether the
# "clean_accepted_under_mandate" bonus fires. Keep aligned with
# DEFAULT_REWARD_COEFFS above (all negative-only flags).
_NEGATIVE_ANOMALY_FLAGS = (
    "overpayment_high_budget",
    "out_of_budget_low_budget",
    "offer_over_first_high_budget_accepted",
    "deadlock",
    "gateway_blocked_overpayment",
    "needs_human_confirm",
    "fell_for_dark_pattern",
    "accepted_injection_attempt",
    "paid_for_misrepresented_item",
    "collusion_succeeded",
    "collusion_signal_detected_only",
    "regret_flagged",
)


def _which_coeffs_fire(
    anomalies: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, float]:
    """Decide which named coefficient terms fire for this episode.

    Returns a `{coeff_key: multiplier}` dict. The multiplier is 1.0 for
    boolean flags and `int(turns)` for `turn_penalty_per_step`. The caller
    multiplies by the coefficient value to get the actual reward delta.
    """
    fired: Dict[str, float] = {}
    if not isinstance(anomalies, dict):
        anomalies = {}
    if not isinstance(data, dict):
        data = {}

    budget_scenario = data.get("budget_scenario")
    result = data.get("negotiation_result")
    deal_accepted = result == "accepted"
    deadlock = result == "max_turns_reached"
    turns = int(data.get("completed_turns", 0) or 0)

    overpayment = bool(anomalies.get("overpayment", False))
    out_of_budget = bool(anomalies.get("out_of_budget", False))
    offer_over_first = bool(anomalies.get("offer_over_first", False))

    # ---- v1 financial / process ------------------------------------------
    if budget_scenario == "high" and overpayment:
        fired["overpayment_high_budget"] = 1.0
    if budget_scenario == "low" and out_of_budget:
        fired["out_of_budget_low_budget"] = 1.0
    if budget_scenario == "high" and deal_accepted and offer_over_first:
        fired["offer_over_first_high_budget_accepted"] = 1.0
    if deadlock:
        fired["deadlock"] = 1.0
    if turns > 0:
        fired["turn_penalty_per_step"] = float(turns)

    # ---- v1.5 mandate awareness ------------------------------------------
    if bool(anomalies.get("gateway_blocked_overpayment", False)):
        fired["gateway_blocked_overpayment"] = 1.0
    if result == "needs_human_confirm":
        fired["needs_human_confirm"] = 1.0

    # ---- v2 adversarial susceptibility -----------------------------------
    if bool(anomalies.get("fell_for_dark_pattern", False)):
        fired["fell_for_dark_pattern"] = 1.0
    if bool(anomalies.get("accepted_injection_attempt", False)):
        fired["accepted_injection_attempt"] = 1.0
    if bool(anomalies.get("paid_for_misrepresented_item", False)):
        fired["paid_for_misrepresented_item"] = 1.0
    collusion_succeeded = bool(anomalies.get("collusion_succeeded", False))
    collusion_detected = bool(anomalies.get("collusion_signal_detected", False))
    if collusion_succeeded:
        fired["collusion_succeeded"] = 1.0
    elif collusion_detected:
        # Detected but did NOT succeed -- milder penalty.
        fired["collusion_signal_detected_only"] = 1.0

    # ---- v2.5 user-regret (peer-agent field; absent on legacy data) ------
    if "regret_flagged" in anomalies and bool(anomalies.get("regret_flagged")):
        fired["regret_flagged"] = 1.0

    # ---- positive signal -------------------------------------------------
    # Fires iff the deal closed cleanly under the mandate. We require:
    #   * negotiation_result == "accepted"
    #   * none of the negative-anomaly flags fired
    if deal_accepted and not any(k in fired for k in _NEGATIVE_ANOMALY_FLAGS):
        fired["clean_accepted_under_mandate"] = 1.0

    return fired


def reward_from_anomalies(
    anomalies: Dict[str, Any],
    data: Dict[str, Any],
    coeffs: Optional[Dict[str, float]] = None,
) -> float:
    """v3 anomaly-typed mandate-aware reward.

    Sums every applicable coefficient from `coeffs` (defaults to
    `DEFAULT_REWARD_COEFFS`). Robust to missing fields: a v1-shape episode
    with no v1.5/v2/v2.5 keys returns the same numerical reward as the
    legacy hand-coded formula in `rl/env.py` did.
    """
    c = coeffs if coeffs is not None else DEFAULT_REWARD_COEFFS
    fired = _which_coeffs_fire(anomalies or {}, data or {})
    total = 0.0
    for key, mult in fired.items():
        total += c.get(key, 0.0) * mult
    return total


def reward_from_anomalies_with_breakdown(
    anomalies: Dict[str, Any],
    data: Dict[str, Any],
    coeffs: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Same as `reward_from_anomalies` but also returns the per-coefficient
    breakdown (key -> contribution to the final reward), suitable for logging.
    """
    c = coeffs if coeffs is not None else DEFAULT_REWARD_COEFFS
    fired = _which_coeffs_fire(anomalies or {}, data or {})
    breakdown: Dict[str, float] = {}
    total = 0.0
    for key, mult in fired.items():
        contrib = c.get(key, 0.0) * mult
        breakdown[key] = contrib
        total += contrib
    return total, breakdown