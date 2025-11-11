"""
CMA Hooks Module

Implements PreHook and PostHook for on-policy integration with A2A negotiation.

PreHook: Runs before each negotiation turn
- Semantic alignment check
- Exploration governance
- Input validation

PostHook: Runs after each negotiation turn
- Risk monitoring
- Causal logging
- Outcome analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class CMAHooks:
    """CMA Hook Manager - Coordinates PreHook and PostHook operations"""

    def __init__(self, cma_core):
        """
        Initialize CMA Hooks

        Args:
            cma_core: Reference to CMA core orchestrator
        """
        self.cma = cma_core
        self.prehook_enabled = cma_core.config.general_config.get("enable_prehook", True)
        self.posthook_enabled = cma_core.config.general_config.get("enable_posthook", True)

    def prehook(
        self,
        turn: int,
        role: str,
        conversation_history: List[Dict[str, str]],
        product_data: Dict[str, Any],
        budget: Optional[float],
        current_offer: float
    ) -> Dict[str, Any]:
        """
        PreHook: Execute before each negotiation turn

        Args:
            turn: Current turn number
            role: Agent role (buyer/seller)
            conversation_history: Conversation history
            product_data: Product information
            budget: Budget constraint
            current_offer: Current price offer

        Returns:
            PreHook result with recommendations and interventions
        """
        if not self.prehook_enabled:
            return {"enabled": False}

        logger.info(f"[PREHOOK] Executing PreHook for {role} at turn {turn}")

        # Log state snapshot
        pre_state = {
            "turn": turn,
            "role": role,
            "current_offer": current_offer,
            "conversation_length": len(conversation_history)
        }
        state_id = self.cma.clx.log_state_snapshot(turn, "pre_turn", pre_state)

        # 1. Semantic Alignment Check (SAC)
        alignment_score = self.cma.sac.compute_alignment_score(
            conversation_history, current_offer, role
        )

        # Check constraint alignment
        is_aligned, alignment_details = self.cma.sac.check_constraint_alignment(
            current_offer, role, conversation_history
        )

        # Detect semantic drift
        drift_analysis = self.cma.sac.detect_semantic_drift(conversation_history, role)

        # 2. Exploration Governance (EG)
        retail_price = float(product_data["Retail Price"].replace("$", "").replace(",", ""))
        wholesale_price = float(product_data["Wholesale Price"].replace("$", "").replace(",", ""))

        offer_history = [current_offer]  # In real integration, this comes from conversation
        exploration_recommendation = self.cma.eg.recommend_exploration_action(
            turn, offer_history, retail_price, wholesale_price, role
        )

        # Check for early termination prevention
        should_prevent = self.cma.eg.should_prevent_early_termination(turn, None)

        # 3. Compile PreHook results
        prehook_result = {
            "enabled": True,
            "turn": turn,
            "role": role,
            "state_snapshot_id": state_id,
            "alignment": {
                "score": alignment_score,
                "is_aligned": is_aligned,
                "details": alignment_details,
                "drift": drift_analysis
            },
            "exploration": {
                "recommendation": exploration_recommendation["recommendation"],
                "reason": exploration_recommendation["reason"],
                "pressure": exploration_recommendation["exploration_pressure"],
                "prevent_early_termination": should_prevent
            },
            "interventions": []
        }

        # 4. Generate interventions if needed
        if not is_aligned:
            intervention = {
                "type": "alignment_correction",
                "severity": "high",
                "message": "Constraint misalignment detected",
                "action": "restate_goals"
            }
            prehook_result["interventions"].append(intervention)

            # Log intervention
            self.cma.clx.log_intervention(
                turn, "sac_alignment_correction",
                {"alignment_details": alignment_details},
                state_id
            )

        if drift_analysis["drift_detected"]:
            intervention = {
                "type": "semantic_drift",
                "severity": "medium",
                "message": "Conversation drifting from price negotiation",
                "action": "refocus_on_price"
            }
            prehook_result["interventions"].append(intervention)

        if exploration_recommendation["recommendation"] == "explore" and turn > 1:
            intervention = {
                "type": "exploration_guidance",
                "severity": "low",
                "message": exploration_recommendation["reason"],
                "action": "continue_exploration"
            }
            prehook_result["interventions"].append(intervention)

        logger.info(
            f"[PREHOOK] Completed for {role}: "
            f"alignment={alignment_score:.2f}, "
            f"interventions={len(prehook_result['interventions'])}"
        )

        return prehook_result

    def posthook(
        self,
        turn: int,
        role: str,
        conversation_history: List[Dict[str, str]],
        product_data: Dict[str, Any],
        budget: Optional[float],
        offer_history: List[float],
        negotiation_result: Optional[str],
        final_price: Optional[float]
    ) -> Dict[str, Any]:
        """
        PostHook: Execute after each negotiation turn

        Args:
            turn: Current turn number
            role: Agent role
            conversation_history: Conversation history
            product_data: Product information
            budget: Budget constraint
            offer_history: Complete price offer history
            negotiation_result: Negotiation result (if concluded)
            final_price: Final price (if deal made)

        Returns:
            PostHook result with risk analysis and logging
        """
        if not self.posthook_enabled:
            return {"enabled": False}

        logger.info(f"[POSTHOOK] Executing PostHook for {role} at turn {turn}")

        # Log state snapshot
        post_state = {
            "turn": turn,
            "role": role,
            "offer_history": offer_history,
            "conversation_length": len(conversation_history),
            "negotiation_result": negotiation_result,
            "final_price": final_price
        }
        state_id = self.cma.clx.log_state_snapshot(turn, "post_turn", post_state)

        # 1. Risk Monitoring (RMS)
        retail_price = float(product_data["Retail Price"].replace("$", "").replace(",", ""))
        wholesale_price = float(product_data["Wholesale Price"].replace("$", "").replace(",", ""))
        current_offer = offer_history[-1] if offer_history else retail_price

        risk_assessment = self.cma.rms.comprehensive_risk_assessment(
            offer_history=offer_history,
            conversation_history=conversation_history,
            current_turn=turn,
            retail_price=retail_price,
            wholesale_price=wholesale_price,
            budget=budget,
            final_price=final_price,
            negotiation_result=negotiation_result,
            current_offer=current_offer,
            role=role
        )

        # 2. Causal Logging (CLX)
        # Log outcome if negotiation concluded
        if negotiation_result:
            outcome_metrics = {
                "result": negotiation_result,
                "final_price": final_price,
                "total_turns": turn,
                "risk_assessment": risk_assessment
            }
            outcome_id = self.cma.clx.log_outcome(
                negotiation_result, final_price, outcome_metrics
            )
        else:
            outcome_id = None

        # 3. Compile PostHook results
        posthook_result = {
            "enabled": True,
            "turn": turn,
            "role": role,
            "state_snapshot_id": state_id,
            "outcome_id": outcome_id,
            "risk_assessment": risk_assessment,
            "alerts": []
        }

        # 4. Generate alerts based on risk assessment
        if risk_assessment["risk_level"] == "high":
            alert = {
                "type": "high_risk",
                "severity": "critical",
                "message": f"High risk detected: {risk_assessment['recommendation']}",
                "risk_score": risk_assessment["risk_score"]
            }
            posthook_result["alerts"].append(alert)

            # Log intervention
            self.cma.clx.log_intervention(
                turn, "rms_high_risk_alert",
                {"risk_assessment": risk_assessment},
                state_id
            )

        if risk_assessment["deadlock_analysis"]["deadlock_detected"]:
            alert = {
                "type": "deadlock",
                "severity": "high",
                "message": "Deadlock detected in negotiation",
                "confidence": risk_assessment["deadlock_analysis"]["confidence"]
            }
            posthook_result["alerts"].append(alert)

        if risk_assessment["bias_analysis"]["bias_detected"]:
            alert = {
                "type": "bias",
                "severity": "medium",
                "message": "Negotiation bias detected",
                "bias_types": [
                    k for k, v in risk_assessment["bias_analysis"].items()
                    if isinstance(v, bool) and v and k != "bias_detected"
                ]
            }
            posthook_result["alerts"].append(alert)

        if risk_assessment["violation_analysis"]["violation_detected"]:
            alert = {
                "type": "constraint_violation",
                "severity": "critical",
                "message": "Constraint violation detected",
                "violations": risk_assessment["violation_analysis"]["violations"]
            }
            posthook_result["alerts"].append(alert)

        logger.info(
            f"[POSTHOOK] Completed for {role}: "
            f"risk_level={risk_assessment['risk_level']}, "
            f"alerts={len(posthook_result['alerts'])}"
        )

        return posthook_result

    def finalize_session(
        self,
        negotiation_result: str,
        final_price: Optional[float],
        total_turns: int
    ) -> Dict[str, Any]:
        """
        Finalize CMA session and generate summary

        Args:
            negotiation_result: Final negotiation result
            final_price: Final agreed price
            total_turns: Total number of turns

        Returns:
            Session summary
        """
        logger.info("[CMA] Finalizing session")

        # Save causal log
        log_file = self.cma.clx.save_causal_log()

        # Generate self-reinforcement data
        training_data = self.cma.clx.generate_self_reinforcement_data()

        # Export for RLHF
        rlhf_file = self.cma.clx.export_for_rlhf("rlhf_training_data.jsonl")

        # Compile session summary
        session_summary = {
            "negotiation_result": negotiation_result,
            "final_price": final_price,
            "total_turns": total_turns,
            "sac_summary": self.cma.sac.get_alignment_summary(),
            "eg_summary": self.cma.eg.get_exploration_summary(),
            "rms_summary": self.cma.rms.get_risk_summary(),
            "clx_summary": self.cma.clx.get_session_summary(),
            "apa_summary": self.cma.apa.get_protocol_summary(),
            "causal_log_file": log_file,
            "rlhf_file": rlhf_file,
            "training_examples": len(training_data)
        }

        logger.info(
            f"[CMA] Session finalized: "
            f"result={negotiation_result}, "
            f"turns={total_turns}, "
            f"training_examples={len(training_data)}"
        )

        return session_summary
