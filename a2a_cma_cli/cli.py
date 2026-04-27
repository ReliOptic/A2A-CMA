"""argparse-based CLI for the A2A-CMA developer toolkit (v4).

Subcommands
-----------
``list-scenarios [--archetype X]``
    Print the active scenarios in ``dataset/scenarios.json`` as a fixed-width
    table. Optionally filtered by seller archetype.

``run --scenario SID [--buyer-model M] [--seller-model M] [--summary-model M]
       [--max-turns N] [--output DIR] [--repeats N] [--no-gateway]``
    Resolve the scenario, build a ``Conversation`` (with auto-built payment
    gateway unless ``--no-gateway``), run the negotiation ``--repeats`` times
    and persist each episode through ``Conversation.save_conversation``.

``run-all [--archetype X] [--buyer-model M] [--seller-model M]
          [--output DIR] [--repeats N]``
    Iterate every active scenario (optionally archetype-filtered) and call
    ``run`` for each. Skips scenarios already at ``--repeats`` in the output
    directory so the command is resumable.

``report DIR [--judge-model M] [--no-regret]``
    Walk ``DIR`` for ``*.json`` episode files; per file recompute
    ``MarkAnomaly.PostDataProcessor.calculate_anomalies``; optionally invoke
    ``evaluation.judge_episode`` for counterfactual user-regret verdicts;
    aggregate to stdout AND emit a machine-readable ``report.json``.

Each subcommand is implemented as ``cmd_<name>(args) -> int`` so unit tests
can drive them without a subprocess. ``main(argv=None) -> int`` wires
``argparse`` to dispatch.

Hard constraints
----------------
* Stdlib only.
* Every error path returns a non-zero exit code with a one-line stderr
  message; no stack traces leak to the user.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Ensure the repo root is on sys.path when invoked as ``python -m a2a_cma_cli``
# from a checkout (rather than installed). This mirrors the idiom the existing
# tests use so ``Conversation`` / ``MarkAnomaly`` etc. import cleanly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers (small, no side-effects, individually testable).
# ---------------------------------------------------------------------------

def _eprint(msg: str) -> None:
    """Single source of truth for user-facing error output."""
    print(msg, file=sys.stderr)


def _load_active_scenarios(scenarios_path: str, products_path: str):
    """Wrap ``scenarios.loader.load_active_scenarios`` so callers can be
    defensive about missing files without doing the try/except themselves.
    """
    from scenarios.loader import load_active_scenarios

    return load_active_scenarios(
        scenarios_path=scenarios_path,
        products_path=products_path,
    )


def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    """Stdlib-only fixed-width pretty printer."""
    all_rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
    sep = " | "
    out_lines = [sep.join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))]
    out_lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        out_lines.append(sep.join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out_lines)


def _existing_repeat_count(output_dir: str, product_id: int) -> int:
    """Count how many ``product_<id>_exp_<n>.json`` files already exist."""
    if not os.path.isdir(output_dir):
        return 0
    prefix = f"product_{product_id}_exp_"
    n = 0
    for name in os.listdir(output_dir):
        if name.startswith(prefix) and name.endswith(".json"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Command: list-scenarios
# ---------------------------------------------------------------------------

def cmd_list_scenarios(args: argparse.Namespace) -> int:
    scenarios_path = getattr(args, "scenarios_path", "dataset/scenarios.json")
    try:
        raw = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        _eprint(f"error: scenarios file not found: {scenarios_path}")
        return 2
    except json.JSONDecodeError as exc:
        _eprint(f"error: invalid JSON in {scenarios_path}: {exc}")
        return 2
    if not isinstance(raw, list):
        _eprint(f"error: {scenarios_path} must be a JSON list")
        return 2

    archetype_filter = getattr(args, "archetype", None)
    rows: List[List[str]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        if not s.get("v1_active", True):
            continue
        seller = s.get("seller_archetype", "honest_retailer")
        if archetype_filter and seller != archetype_filter:
            continue
        sid = str(s.get("scenario_id", "?"))
        product_id = s.get("product_id", "?")
        try:
            hard_cap = s["spend_authorization"]["hard_cap"]
        except (KeyError, TypeError):
            hard_cap = "?"
        risks = s.get("expected_risks", []) or []
        rows.append([
            sid,
            f"product#{product_id}",
            str(hard_cap),
            seller,
            ",".join(str(r) for r in risks),
        ])

    if not rows:
        if archetype_filter:
            _eprint(f"error: no active scenarios match archetype={archetype_filter!r}")
        else:
            _eprint("error: no active scenarios found")
        return 1

    headers = ["scenario_id", "product", "hard_cap", "archetype", "expected_risks"]
    print(_format_table(rows, headers))
    return 0


# ---------------------------------------------------------------------------
# Command: run (single scenario, --repeats episodes)
# ---------------------------------------------------------------------------

def _build_conversation(
    scenario,
    buyer_model: str,
    seller_model: str,
    summary_model: str,
    max_turns: int,
    experiment_num: int,
    no_gateway: bool,
):
    """Construct a ``Conversation`` with or without the AP2 gateway."""
    # Imported here so the CLI loads even if Conversation has heavy deps
    # (e.g. when only running list-scenarios in a stdlib-only env).
    from Conversation import Conversation

    kwargs: Dict[str, Any] = dict(
        product_data=scenario.product,
        buyer_model=buyer_model,
        seller_model=seller_model,
        summary_model=summary_model,
        max_turns=max_turns,
        experiment_num=experiment_num,
        scenario=scenario,
    )
    if no_gateway:
        # Explicitly override the auto-build path so the safety layer is
        # bypassed -- used for ablation runs.
        kwargs["gateway"] = None
        convo = Conversation(**kwargs)
        # The Conversation constructor only auto-builds the gateway when both
        # ``gateway is None`` and ``scenario is not None``. To honour
        # --no-gateway we explicitly null it out *after* construction.
        convo.gateway = None
        return convo
    return Conversation(**kwargs)


def cmd_run(args: argparse.Namespace) -> int:
    scenarios_path = getattr(args, "scenarios_path", "dataset/scenarios.json")
    products_path = getattr(args, "products_path", "dataset/products.json")
    sid = args.scenario
    output_dir = args.output
    repeats = max(1, int(args.repeats))

    try:
        from scenarios.loader import get_scenario, attach_product
        scenario = get_scenario(sid, path=scenarios_path)
        attach_product(scenario, products_path=products_path)
    except FileNotFoundError as exc:
        _eprint(f"error: missing scenario/product file: {exc}")
        return 2
    except KeyError as exc:
        _eprint(f"error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - convert to clean error
        _eprint(f"error: failed to resolve scenario {sid!r}: {exc}")
        return 2

    os.makedirs(output_dir, exist_ok=True)

    already = _existing_repeat_count(output_dir, scenario.product_id)
    to_run = max(0, repeats - already)
    if to_run == 0:
        print(f"[run] {sid}: already at {already} repeats in {output_dir}; nothing to do")
        return 0

    failures = 0
    for i in range(already, already + to_run):
        try:
            convo = _build_conversation(
                scenario=scenario,
                buyer_model=args.buyer_model,
                seller_model=args.seller_model,
                summary_model=args.summary_model,
                max_turns=args.max_turns,
                experiment_num=i,
                no_gateway=args.no_gateway,
            )
            convo.run_negotiation()
            convo.save_conversation(output_dir)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _eprint(f"error: episode {sid}#{i} failed: {exc}")
            if os.environ.get("A2A_CMA_CLI_DEBUG"):
                traceback.print_exc()
    if failures and failures == to_run:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Command: run-all
# ---------------------------------------------------------------------------

def cmd_run_all(args: argparse.Namespace) -> int:
    scenarios_path = getattr(args, "scenarios_path", "dataset/scenarios.json")
    products_path = getattr(args, "products_path", "dataset/products.json")
    try:
        scenarios = _load_active_scenarios(scenarios_path, products_path)
    except FileNotFoundError as exc:
        _eprint(f"error: missing scenario/product file: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        _eprint(f"error: failed to load scenarios: {exc}")
        return 2

    if args.archetype:
        scenarios = [s for s in scenarios if s.seller_archetype == args.archetype]
    if not scenarios:
        _eprint("error: no scenarios match the supplied filters")
        return 1

    overall_rc = 0
    for scenario in scenarios:
        # Build a per-scenario args namespace and reuse cmd_run for one
        # consistent code path (idempotency, error handling, etc.).
        sub = argparse.Namespace(
            scenario=scenario.scenario_id,
            buyer_model=args.buyer_model,
            seller_model=args.seller_model,
            summary_model=args.summary_model,
            max_turns=args.max_turns,
            output=args.output,
            repeats=args.repeats,
            no_gateway=args.no_gateway,
            scenarios_path=scenarios_path,
            products_path=products_path,
        )
        rc = cmd_run(sub)
        if rc != 0:
            overall_rc = rc
    return overall_rc


# ---------------------------------------------------------------------------
# Command: report
# ---------------------------------------------------------------------------

# Anomaly keys we surface in the incidence table. Kept aligned with
# MarkAnomaly.calculate_anomalies's boolean output keys; numeric keys like
# ``bargaining_rate`` are intentionally excluded.
_ANOMALY_KEYS: Tuple[str, ...] = (
    "overpayment",
    "out_of_budget",
    "out_of_wholesale",
    "irrational_refuse",
    "deadlock",
    "gateway_intervention_required",
    "gateway_blocked_overpayment",
    "fell_for_dark_pattern",
    "accepted_injection_attempt",
    "paid_for_misrepresented_item",
    "collusion_signal_detected",
    "collusion_succeeded",
    "regret_flagged",
)


def _iter_episode_files(output_dir: str) -> List[str]:
    """Return every .json file under ``output_dir`` except ``report.json``."""
    out: List[str] = []
    for root, _, files in os.walk(output_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            if name == "report.json":
                continue
            out.append(os.path.join(root, name))
    out.sort()
    return out


def _safe_load_episode(path: str) -> Optional[Dict[str, Any]]:
    """Load a single episode JSON; return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _scenario_id_of(episode: Dict[str, Any]) -> str:
    sc = episode.get("scenario")
    if isinstance(sc, dict) and sc.get("scenario_id"):
        return str(sc["scenario_id"])
    return "(unknown)"


def _archetype_of(episode: Dict[str, Any]) -> str:
    arche = episode.get("seller_archetype")
    if not arche:
        sc = episode.get("scenario")
        if isinstance(sc, dict):
            arche = sc.get("seller_archetype")
    return str(arche or "(unknown)")


def cmd_report(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        _eprint(f"error: output directory not found: {output_dir}")
        return 2

    files = _iter_episode_files(output_dir)
    if not files:
        _eprint(f"error: no episode JSON files found under {output_dir}")
        return 1

    # Lazy imports keep the CLI fast for trivial commands and avoid pulling
    # the evaluation oracle until --judge is actually used.
    from MarkAnomaly import PostDataProcessor, attach_regret_verdict
    from rl.policy import reward_from_anomalies

    processor = PostDataProcessor()

    total = 0
    by_scenario: Counter = Counter()
    by_archetype: Counter = Counter()
    anomaly_counts: Counter = Counter({k: 0 for k in _ANOMALY_KEYS})
    intervention_count = 0
    gateway_blocked_count = 0
    rewards: List[float] = []
    regret_verdicts: List[Any] = []
    malformed: List[str] = []

    judge = None
    if not args.no_regret:
        try:
            from evaluation import RegretJudge
            judge = RegretJudge(model_name=args.judge_model)
        except Exception as exc:  # noqa: BLE001
            _eprint(f"warning: could not initialise regret judge ({exc}); skipping")
            judge = None

    for path in files:
        episode = _safe_load_episode(path)
        if episode is None:
            malformed.append(path)
            continue
        # Optional regret verdict; attach BEFORE recomputing anomalies so the
        # ``regret_flagged`` flag picks it up downstream.
        if judge is not None:
            try:
                from evaluation import judge_episode
                rationale = judge_episode(episode, judge=judge)
                attach_regret_verdict(episode, rationale)
                regret_verdicts.append(rationale.verdict)
            except Exception as exc:  # noqa: BLE001
                _eprint(f"warning: regret judge failed on {path}: {exc}")

        try:
            anomalies = processor.calculate_anomalies(episode)
        except Exception as exc:  # noqa: BLE001
            _eprint(f"warning: anomaly calc failed on {path}: {exc}")
            malformed.append(path)
            continue

        total += 1
        by_scenario[_scenario_id_of(episode)] += 1
        by_archetype[_archetype_of(episode)] += 1

        for key in _ANOMALY_KEYS:
            if bool(anomalies.get(key, False)):
                anomaly_counts[key] += 1
        if bool(anomalies.get("gateway_intervention_required", False)):
            intervention_count += 1
        if bool(anomalies.get("gateway_blocked_overpayment", False)):
            gateway_blocked_count += 1

        try:
            r = reward_from_anomalies(anomalies, episode)
        except Exception:  # noqa: BLE001
            r = 0.0
        rewards.append(float(r))

    if total == 0:
        _eprint(f"error: no usable episodes in {output_dir} ({len(malformed)} malformed)")
        return 1

    # ---------- aggregate ---------------------------------------------------
    incidence = {
        k: (anomaly_counts[k] / total) for k in _ANOMALY_KEYS
    }
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mandate_metrics = {
        "gateway_intervention_rate": intervention_count / total,
        "gateway_blocked_overpayment_rate": gateway_blocked_count / total,
    }

    regret_summary: Optional[Dict[str, float]] = None
    if regret_verdicts:
        from evaluation import aggregate_regret_rate
        regret_summary = aggregate_regret_rate(regret_verdicts)

    report: Dict[str, Any] = {
        "output_dir": os.path.abspath(output_dir),
        "total_episodes": total,
        "malformed_episodes": len(malformed),
        "by_scenario": dict(by_scenario),
        "by_archetype": dict(by_archetype),
        "anomaly_incidence": incidence,
        "mandate_metrics": mandate_metrics,
        "reward_summary": {
            "mean": mean_reward,
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
            "n": len(rewards),
        },
        "regret_summary": regret_summary,
    }

    # ---------- stdout summary ---------------------------------------------
    print(f"[report] {output_dir}: {total} episodes ({len(malformed)} malformed)")
    if by_scenario:
        print("\nBy scenario:")
        for sid, n in sorted(by_scenario.items()):
            print(f"  {sid}: {n}")
    if by_archetype:
        print("\nBy archetype:")
        for a, n in sorted(by_archetype.items()):
            print(f"  {a}: {n}")
    print("\nAnomaly incidence (% of episodes flagged):")
    incidence_rows = [
        [k, f"{anomaly_counts[k]}/{total}", f"{incidence[k] * 100:.1f}%"]
        for k in _ANOMALY_KEYS
    ]
    print(_format_table(incidence_rows, ["anomaly", "count", "pct"]))
    print("\nMandate metrics:")
    print(f"  gateway_intervention_rate: {mandate_metrics['gateway_intervention_rate'] * 100:.1f}%")
    print(f"  gateway_blocked_overpayment_rate: {mandate_metrics['gateway_blocked_overpayment_rate'] * 100:.1f}%")
    print("\nReward summary (rl.policy.reward_from_anomalies):")
    print(f"  mean={mean_reward:.3f}  min={min(rewards):.3f}  max={max(rewards):.3f}  n={len(rewards)}")
    if regret_summary is not None:
        print("\nRegret summary (evaluation.aggregate_regret_rate):")
        for k, v in regret_summary.items():
            print(f"  {k}: {v}")

    # ---------- machine-readable report.json --------------------------------
    report_path = os.path.join(output_dir, "report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nWrote {report_path}")
    except OSError as exc:
        _eprint(f"warning: could not write {report_path}: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a2a-cma",
        description=(
            "A2A-CMA developer toolkit: stress-test a shopping agent "
            "against the v1->v3 safety scenarios."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list-scenarios -------------------------------------------------------
    p_list = sub.add_parser("list-scenarios", help="List active scenarios")
    p_list.add_argument("--archetype", default=None,
                        help="Filter by seller archetype (e.g. prompt_injecting)")
    p_list.add_argument("--scenarios-path", dest="scenarios_path",
                        default="dataset/scenarios.json")
    p_list.set_defaults(func=cmd_list_scenarios)

    # run ------------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a single scenario --repeats times")
    p_run.add_argument("--scenario", required=True, help="Scenario id (see list-scenarios)")
    p_run.add_argument("--buyer-model", dest="buyer_model", default="gpt-4o-mini")
    p_run.add_argument("--seller-model", dest="seller_model", default="gpt-4o-mini")
    p_run.add_argument("--summary-model", dest="summary_model", default="gpt-4o-mini")
    p_run.add_argument("--max-turns", dest="max_turns", type=int, default=20)
    p_run.add_argument("--output", default="runs/latest", help="Output directory for episode JSONs")
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--no-gateway", dest="no_gateway", action="store_true",
                       help="Disable the AP2 gateway (ablation)")
    p_run.add_argument("--scenarios-path", dest="scenarios_path",
                       default="dataset/scenarios.json")
    p_run.add_argument("--products-path", dest="products_path",
                       default="dataset/products.json")
    p_run.set_defaults(func=cmd_run)

    # run-all --------------------------------------------------------------
    p_all = sub.add_parser("run-all", help="Run every active scenario")
    p_all.add_argument("--archetype", default=None)
    p_all.add_argument("--buyer-model", dest="buyer_model", default="gpt-4o-mini")
    p_all.add_argument("--seller-model", dest="seller_model", default="gpt-4o-mini")
    p_all.add_argument("--summary-model", dest="summary_model", default="gpt-4o-mini")
    p_all.add_argument("--max-turns", dest="max_turns", type=int, default=20)
    p_all.add_argument("--output", default="runs/all")
    p_all.add_argument("--repeats", type=int, default=1)
    p_all.add_argument("--no-gateway", dest="no_gateway", action="store_true")
    p_all.add_argument("--scenarios-path", dest="scenarios_path",
                       default="dataset/scenarios.json")
    p_all.add_argument("--products-path", dest="products_path",
                       default="dataset/products.json")
    p_all.set_defaults(func=cmd_run_all)

    # report ---------------------------------------------------------------
    p_report = sub.add_parser("report", help="Aggregate anomalies / regret across an output dir")
    p_report.add_argument("output_dir", help="Directory containing episode JSON files")
    p_report.add_argument("--judge-model", dest="judge_model", default="gpt-4o-mini")
    p_report.add_argument("--no-regret", dest="no_regret", action="store_true",
                          help="Skip the regret oracle (useful when no API key is set)")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return int(func(args))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - clean error for end users
        _eprint(f"error: {exc}")
        if os.environ.get("A2A_CMA_CLI_DEBUG"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
