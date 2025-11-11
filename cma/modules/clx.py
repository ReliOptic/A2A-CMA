"""
CLX (Causal Logger & Explainer) Module

Responsible for:
- Logging causal traces (state → intervention → outcome)
- Counterfactual tagging and analysis
- A2A trace ID management
- Self-reinforcement data generation
"""

import logging
import json
import os
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CausalLoggerExplainer:
    """Causal Logger & Explainer - Logs causal data for self-reinforcement"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CLX module

        Args:
            config: Configuration dictionary for CLX
        """
        self.config = config
        self.enable_causal_logging = config.get("enable_causal_logging", True)
        self.enable_counterfactual = config.get("enable_counterfactual_tagging", True)
        self.log_format = config.get("log_format", "jsonl")
        self.log_dir = config.get("log_dir", "cma_logs")
        self.include_state_snapshots = config.get("include_state_snapshots", True)
        self.include_intervention_traces = config.get("include_intervention_traces", True)
        self.a2a_trace_id_enabled = config.get("a2a_trace_id_enabled", True)

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Session tracking
        self.session_id = self._generate_session_id()
        self.a2a_trace_id = self._generate_a2a_trace_id()
        self.causal_log = []
        self.counterfactual_pairs = []
        self.intervention_count = 0

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"cma_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _generate_a2a_trace_id(self) -> str:
        """Generate A2A-compatible trace ID"""
        return f"a2a_trace_{uuid.uuid4()}"

    def log_state_snapshot(
        self,
        turn: int,
        state_type: str,
        state_data: Dict[str, Any]
    ) -> str:
        """
        Log a state snapshot

        Args:
            turn: Current turn number
            state_type: Type of state (pre_turn, post_turn, pre_hook, post_hook)
            state_data: State data dictionary

        Returns:
            State snapshot ID
        """
        if not self.enable_causal_logging or not self.include_state_snapshots:
            return ""

        snapshot_id = f"state_{turn}_{state_type}_{uuid.uuid4().hex[:8]}"

        snapshot = {
            "snapshot_id": snapshot_id,
            "session_id": self.session_id,
            "a2a_trace_id": self.a2a_trace_id,
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "state_type": state_type,
            "state_data": state_data
        }

        self.causal_log.append(snapshot)

        logger.debug(f"[CLX] Logged state snapshot: {snapshot_id}")

        return snapshot_id

    def log_intervention(
        self,
        turn: int,
        intervention_type: str,
        intervention_data: Dict[str, Any],
        pre_state_id: str,
        post_state_id: Optional[str] = None
    ) -> str:
        """
        Log a CMA intervention

        Args:
            turn: Current turn number
            intervention_type: Type of intervention (sac_alignment, eg_exploration, rms_risk, etc.)
            intervention_data: Intervention details
            pre_state_id: ID of state before intervention
            post_state_id: ID of state after intervention (if available)

        Returns:
            Intervention ID
        """
        if not self.enable_causal_logging or not self.include_intervention_traces:
            return ""

        intervention_id = f"intervention_{turn}_{intervention_type}_{uuid.uuid4().hex[:8]}"
        self.intervention_count += 1

        intervention_log = {
            "intervention_id": intervention_id,
            "session_id": self.session_id,
            "a2a_trace_id": self.a2a_trace_id,
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "intervention_type": intervention_type,
            "intervention_data": intervention_data,
            "pre_state_id": pre_state_id,
            "post_state_id": post_state_id,
            "causal_chain": {
                "pre_state": pre_state_id,
                "intervention": intervention_id,
                "post_state": post_state_id
            }
        }

        self.causal_log.append(intervention_log)

        logger.info(f"[CLX] Logged intervention: {intervention_type} at turn {turn}")

        return intervention_id

    def log_outcome(
        self,
        negotiation_result: str,
        final_price: Optional[float],
        outcome_metrics: Dict[str, Any]
    ) -> str:
        """
        Log negotiation outcome

        Args:
            negotiation_result: Result (accepted, rejected, deadlock, etc.)
            final_price: Final agreed price
            outcome_metrics: Outcome metrics and analysis

        Returns:
            Outcome ID
        """
        if not self.enable_causal_logging:
            return ""

        outcome_id = f"outcome_{self.session_id}"

        outcome_log = {
            "outcome_id": outcome_id,
            "session_id": self.session_id,
            "a2a_trace_id": self.a2a_trace_id,
            "timestamp": datetime.now().isoformat(),
            "negotiation_result": negotiation_result,
            "final_price": final_price,
            "outcome_metrics": outcome_metrics,
            "total_interventions": self.intervention_count
        }

        self.causal_log.append(outcome_log)

        logger.info(f"[CLX] Logged outcome: {negotiation_result}")

        return outcome_id

    def create_counterfactual_pair(
        self,
        factual_state: Dict[str, Any],
        counterfactual_state: Dict[str, Any],
        intervention: Dict[str, Any],
        turn: int
    ) -> str:
        """
        Create a counterfactual pair for causal analysis

        Args:
            factual_state: Actual state that occurred
            counterfactual_state: Hypothetical state without intervention
            intervention: Intervention that was applied
            turn: Turn number

        Returns:
            Counterfactual pair ID
        """
        if not self.enable_counterfactual:
            return ""

        cf_pair_id = f"cf_pair_{turn}_{uuid.uuid4().hex[:8]}"

        cf_pair = {
            "cf_pair_id": cf_pair_id,
            "session_id": self.session_id,
            "a2a_trace_id": self.a2a_trace_id,
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "factual": factual_state,
            "counterfactual": counterfactual_state,
            "intervention": intervention,
            "causal_effect": self._compute_causal_effect(
                factual_state, counterfactual_state
            )
        }

        self.counterfactual_pairs.append(cf_pair)

        logger.debug(f"[CLX] Created counterfactual pair: {cf_pair_id}")

        return cf_pair_id

    def _compute_causal_effect(
        self,
        factual: Dict[str, Any],
        counterfactual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute causal effect between factual and counterfactual states

        Args:
            factual: Factual state
            counterfactual: Counterfactual state

        Returns:
            Causal effect metrics
        """
        # Simple effect computation (can be extended)
        effect = {
            "price_delta": None,
            "outcome_changed": False,
            "turn_delta": None
        }

        # Price effect
        if "final_price" in factual and "final_price" in counterfactual:
            effect["price_delta"] = factual["final_price"] - counterfactual["final_price"]

        # Outcome effect
        if "outcome" in factual and "outcome" in counterfactual:
            effect["outcome_changed"] = factual["outcome"] != counterfactual["outcome"]

        # Turn effect
        if "turns" in factual and "turns" in counterfactual:
            effect["turn_delta"] = factual["turns"] - counterfactual["turns"]

        return effect

    def generate_causal_trace(self) -> Dict[str, Any]:
        """
        Generate complete causal trace for the session

        Returns:
            Complete causal trace
        """
        causal_trace = {
            "session_id": self.session_id,
            "a2a_trace_id": self.a2a_trace_id,
            "timestamp": datetime.now().isoformat(),
            "total_logs": len(self.causal_log),
            "total_interventions": self.intervention_count,
            "total_counterfactuals": len(self.counterfactual_pairs),
            "causal_log": self.causal_log,
            "counterfactual_pairs": self.counterfactual_pairs
        }

        return causal_trace

    def save_causal_log(self, filename: Optional[str] = None) -> str:
        """
        Save causal log to file

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved log file
        """
        if not self.enable_causal_logging:
            return ""

        if filename is None:
            filename = f"{self.session_id}_causal_log.{self.log_format}"

        filepath = os.path.join(self.log_dir, filename)

        causal_trace = self.generate_causal_trace()

        if self.log_format == "json":
            with open(filepath, 'w') as f:
                json.dump(causal_trace, f, indent=2)
        elif self.log_format == "jsonl":
            with open(filepath, 'w') as f:
                # Write each log entry as a separate line
                for entry in self.causal_log:
                    f.write(json.dumps(entry) + '\n')
                # Write counterfactual pairs
                for cf_pair in self.counterfactual_pairs:
                    f.write(json.dumps(cf_pair) + '\n')

        logger.info(f"[CLX] Saved causal log to: {filepath}")

        return filepath

    def generate_self_reinforcement_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data for self-reinforcement learning

        Returns:
            List of training examples
        """
        training_examples = []

        # Extract intervention-outcome pairs
        for log_entry in self.causal_log:
            if "intervention_type" in log_entry:
                example = {
                    "state": log_entry.get("pre_state_id"),
                    "action": log_entry["intervention_type"],
                    "intervention_data": log_entry["intervention_data"],
                    "session_id": self.session_id,
                    "turn": log_entry.get("turn")
                }
                training_examples.append(example)

        # Add counterfactual examples for contrastive learning
        for cf_pair in self.counterfactual_pairs:
            example = {
                "factual": cf_pair["factual"],
                "counterfactual": cf_pair["counterfactual"],
                "causal_effect": cf_pair["causal_effect"],
                "intervention": cf_pair["intervention"],
                "type": "counterfactual",
                "session_id": self.session_id
            }
            training_examples.append(example)

        logger.info(
            f"[CLX] Generated {len(training_examples)} "
            f"self-reinforcement training examples"
        )

        return training_examples

    def export_for_rlhf(self, output_file: str) -> str:
        """
        Export data in RLHF-compatible format

        Args:
            output_file: Output filename

        Returns:
            Path to exported file
        """
        training_data = self.generate_self_reinforcement_data()

        filepath = os.path.join(self.log_dir, output_file)

        with open(filepath, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')

        logger.info(f"[CLX] Exported RLHF data to: {filepath}")

        return filepath

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of logging session"""
        return {
            "session_id": self.session_id,
            "a2a_trace_id": self.a2a_trace_id,
            "total_logs": len(self.causal_log),
            "total_interventions": self.intervention_count,
            "total_counterfactuals": len(self.counterfactual_pairs),
            "log_dir": self.log_dir
        }
