"""
SAC (Semantic Alignment Core) Module

Responsible for:
- Goal and constraint alignment detection
- Semantic drift monitoring
- Goal restatement for clarity
- Misalignment detection and correction
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class SemanticAlignmentCore:
    """Semantic Alignment Core - Ensures goal and constraint alignment"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SAC module

        Args:
            config: Configuration dictionary for SAC
        """
        self.config = config
        self.alignment_threshold = config.get("alignment_threshold", 0.7)
        self.enable_goal_restatement = config.get("enable_goal_restatement", True)
        self.detect_semantic_drift = config.get("detect_semantic_drift", True)

        # State tracking
        self.initial_goals = {}
        self.initial_constraints = {}
        self.alignment_history = []

    def analyze_goals_and_constraints(
        self,
        product_data: Dict[str, Any],
        budget: Optional[float],
        role: str  # "buyer" or "seller"
    ) -> Dict[str, Any]:
        """
        Analyze and extract goals and constraints from negotiation context

        Args:
            product_data: Product information
            budget: Budget constraint (for buyer)
            role: Agent role (buyer or seller)

        Returns:
            Dictionary containing goals and constraints
        """
        logger.info(f"[SAC] Analyzing goals and constraints for {role}")

        retail_price = float(product_data["Retail Price"].replace("$", "").replace(",", ""))
        wholesale_price = float(product_data["Wholesale Price"].replace("$", "").replace(",", ""))

        if role == "buyer":
            goals = {
                "primary": "minimize_price",
                "target_price": budget if budget else retail_price * 0.8,
                "acceptable_range": (wholesale_price * 0.9, budget if budget else retail_price)
            }
            constraints = {
                "hard_budget": budget,
                "cannot_exceed": budget,
                "minimum_turns": self.config.get("min_turns", 3)
            }
        else:  # seller
            goals = {
                "primary": "maximize_price",
                "target_price": retail_price,
                "acceptable_range": (wholesale_price, retail_price * 1.2)
            }
            constraints = {
                "minimum_price": wholesale_price,
                "cannot_go_below": wholesale_price,
                "reputation_preservation": True
            }

        self.initial_goals[role] = goals
        self.initial_constraints[role] = constraints

        return {
            "goals": goals,
            "constraints": constraints,
            "alignment_score": 1.0  # Initial alignment is perfect
        }

    def check_constraint_alignment(
        self,
        current_offer: float,
        role: str,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if current negotiation state aligns with initial constraints

        Args:
            current_offer: Current price offer
            role: Agent role
            conversation_history: Conversation history

        Returns:
            Tuple of (is_aligned, alignment_details)
        """
        if role not in self.initial_constraints:
            logger.warning(f"[SAC] No initial constraints found for {role}")
            return True, {"status": "unknown"}

        constraints = self.initial_constraints[role]
        violations = []

        if role == "buyer":
            if constraints["hard_budget"] and current_offer > constraints["hard_budget"]:
                violations.append({
                    "type": "budget_violation",
                    "severity": "high",
                    "message": f"Offer ${current_offer:.2f} exceeds budget ${constraints['hard_budget']:.2f}"
                })
        else:  # seller
            if current_offer < constraints["minimum_price"]:
                violations.append({
                    "type": "minimum_price_violation",
                    "severity": "high",
                    "message": f"Offer ${current_offer:.2f} below minimum ${constraints['minimum_price']:.2f}"
                })

        is_aligned = len(violations) == 0
        alignment_score = 1.0 if is_aligned else max(0.0, 1.0 - 0.3 * len(violations))

        alignment_details = {
            "is_aligned": is_aligned,
            "alignment_score": alignment_score,
            "violations": violations,
            "constraints_checked": constraints
        }

        self.alignment_history.append(alignment_details)

        if not is_aligned:
            logger.warning(f"[SAC] Constraint misalignment detected for {role}: {violations}")

        return is_aligned, alignment_details

    def detect_semantic_drift(
        self,
        conversation_history: List[Dict[str, str]],
        role: str
    ) -> Dict[str, Any]:
        """
        Detect if conversation has drifted from original goals

        Args:
            conversation_history: Full conversation history
            role: Agent role

        Returns:
            Drift analysis results
        """
        if not self.detect_semantic_drift or role not in self.initial_goals:
            return {"drift_detected": False, "drift_score": 0.0}

        # Simple heuristic-based drift detection
        # In production, this would use embedding similarity
        recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history

        # Check for off-topic keywords
        off_topic_indicators = [
            "warranty", "insurance", "delivery", "installation",
            "color", "size", "model", "brand", "alternative"
        ]

        drift_signals = 0
        for msg in recent_messages:
            content_lower = msg.get("message", "").lower()
            for indicator in off_topic_indicators:
                if indicator in content_lower and "price" not in content_lower:
                    drift_signals += 1

        drift_score = min(1.0, drift_signals / (len(recent_messages) + 1))
        drift_detected = drift_score > 0.3

        drift_analysis = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "drift_signals": drift_signals,
            "recommendation": "refocus_on_price" if drift_detected else "continue"
        }

        if drift_detected:
            logger.warning(f"[SAC] Semantic drift detected for {role}: score={drift_score:.2f}")

        return drift_analysis

    def restate_goals(
        self,
        role: str,
        current_state: Dict[str, Any]
    ) -> str:
        """
        Generate a restated goal prompt for realignment

        Args:
            role: Agent role
            current_state: Current negotiation state

        Returns:
            Restated goal prompt
        """
        if not self.enable_goal_restatement or role not in self.initial_goals:
            return ""

        goals = self.initial_goals[role]
        constraints = self.initial_constraints.get(role, {})

        if role == "buyer":
            restatement = f"""
            [Goal Realignment Reminder]
            Your primary goal: Purchase at minimum price
            Your budget constraint: ${constraints.get('hard_budget', 'N/A'):.2f}
            Current offer: ${current_state.get('current_offer', 0):.2f}
            Target range: ${goals['acceptable_range'][0]:.2f} - ${goals['acceptable_range'][1]:.2f}

            Remember: Do not exceed your budget. Focus on price negotiation.
            """
        else:  # seller
            restatement = f"""
            [Goal Realignment Reminder]
            Your primary goal: Sell at maximum price
            Your minimum price: ${constraints.get('minimum_price', 0):.2f}
            Current offer: ${current_state.get('current_offer', 0):.2f}
            Target range: ${goals['acceptable_range'][0]:.2f} - ${goals['acceptable_range'][1]:.2f}

            Remember: Do not go below minimum price. Maintain value positioning.
            """

        return restatement.strip()

    def compute_alignment_score(
        self,
        conversation_history: List[Dict[str, str]],
        current_offer: float,
        role: str
    ) -> float:
        """
        Compute overall alignment score for current negotiation state

        Args:
            conversation_history: Conversation history
            current_offer: Current price offer
            role: Agent role

        Returns:
            Alignment score (0.0 to 1.0)
        """
        # Check constraint alignment
        is_aligned, alignment_details = self.check_constraint_alignment(
            current_offer, role, conversation_history
        )
        constraint_score = alignment_details["alignment_score"]

        # Check semantic drift
        drift_analysis = self.detect_semantic_drift(conversation_history, role)
        drift_penalty = drift_analysis["drift_score"] * 0.3

        # Compute final score
        alignment_score = max(0.0, constraint_score - drift_penalty)

        logger.info(f"[SAC] Alignment score for {role}: {alignment_score:.2f}")

        return alignment_score

    def get_alignment_summary(self) -> Dict[str, Any]:
        """Get summary of alignment history"""
        return {
            "total_checks": len(self.alignment_history),
            "alignment_history": self.alignment_history,
            "average_alignment": sum(h["alignment_score"] for h in self.alignment_history) / len(self.alignment_history) if self.alignment_history else 1.0
        }
