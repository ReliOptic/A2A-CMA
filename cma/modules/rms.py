"""
RMS (Risk Monitoring Sentinel) Module

Responsible for:
- Deadlock detection (repetitive offers, no progress)
- Bias detection (seller dominance, buyer exploitation)
- Constraint violation detection
- Real-time risk flagging
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class RiskMonitoringSentinel:
    """Risk Monitoring Sentinel - Detects anomalies and risks in negotiation"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RMS module

        Args:
            config: Configuration dictionary for RMS
        """
        self.config = config
        self.enable_deadlock_detection = config.get("enable_deadlock_detection", True)
        self.enable_bias_detection = config.get("enable_bias_detection", True)
        self.enable_constraint_violation = config.get("enable_constraint_violation_detection", True)
        self.deadlock_repetition_threshold = config.get("deadlock_repetition_threshold", 3)
        self.bias_asymmetry_threshold = config.get("bias_asymmetry_threshold", 0.3)
        self.constraint_violation_tolerance = config.get("constraint_violation_tolerance", 0.05)

        # Risk tracking
        self.risk_flags = []
        self.deadlock_warnings = 0
        self.bias_warnings = 0
        self.violation_warnings = 0

    def detect_deadlock(
        self,
        offer_history: List[float],
        conversation_history: List[Dict[str, str]],
        current_turn: int
    ) -> Dict[str, Any]:
        """
        Detect if negotiation is in deadlock

        Args:
            offer_history: List of price offers
            conversation_history: Conversation history
            current_turn: Current turn number

        Returns:
            Deadlock analysis dictionary
        """
        if not self.enable_deadlock_detection or len(offer_history) < self.deadlock_repetition_threshold:
            return {"deadlock_detected": False, "confidence": 0.0}

        # Check for repeated offers
        recent_offers = offer_history[-self.deadlock_repetition_threshold:]
        offer_counts = Counter(recent_offers)
        most_common_offer, repetitions = offer_counts.most_common(1)[0]

        # Deadlock if same offer repeated threshold times
        repeated_offer_deadlock = repetitions >= self.deadlock_repetition_threshold

        # Check for stagnation (very small price changes)
        if len(offer_history) >= 5:
            recent_5 = offer_history[-5:]
            price_variance = max(recent_5) - min(recent_5)
            avg_price = sum(recent_5) / len(recent_5)
            stagnation_deadlock = (price_variance / avg_price) < 0.01 if avg_price > 0 else True
        else:
            stagnation_deadlock = False

        # Check for semantic deadlock (repeated rejections)
        if len(conversation_history) >= 4:
            recent_messages = conversation_history[-4:]
            rejection_keywords = ["no", "cannot", "can't", "won't", "impossible", "too high", "too low"]
            rejection_count = sum(
                1 for msg in recent_messages
                if any(keyword in msg.get("message", "").lower() for keyword in rejection_keywords)
            )
            semantic_deadlock = rejection_count >= 3
        else:
            semantic_deadlock = False

        deadlock_detected = repeated_offer_deadlock or stagnation_deadlock or semantic_deadlock
        confidence = 0.0

        if repeated_offer_deadlock:
            confidence += 0.5
        if stagnation_deadlock:
            confidence += 0.3
        if semantic_deadlock:
            confidence += 0.2

        confidence = min(1.0, confidence)

        deadlock_analysis = {
            "deadlock_detected": deadlock_detected,
            "confidence": confidence,
            "repeated_offer_deadlock": repeated_offer_deadlock,
            "stagnation_deadlock": stagnation_deadlock,
            "semantic_deadlock": semantic_deadlock,
            "most_common_offer": most_common_offer if repeated_offer_deadlock else None,
            "repetitions": repetitions if repeated_offer_deadlock else 0,
            "recommendation": "intervene" if deadlock_detected else "continue"
        }

        if deadlock_detected:
            self.deadlock_warnings += 1
            self.risk_flags.append({
                "type": "deadlock",
                "turn": current_turn,
                "confidence": confidence,
                "details": deadlock_analysis
            })
            logger.warning(
                f"[RMS] Deadlock detected at turn {current_turn} "
                f"(confidence: {confidence:.2f})"
            )

        return deadlock_analysis

    def detect_bias(
        self,
        offer_history: List[float],
        retail_price: float,
        wholesale_price: float,
        budget: Optional[float],
        final_price: Optional[float],
        negotiation_result: Optional[str]
    ) -> Dict[str, Any]:
        """
        Detect if there's systematic bias in negotiation

        Args:
            offer_history: Price offer history
            retail_price: Retail price
            wholesale_price: Wholesale price
            budget: Buyer budget
            final_price: Final agreed price (if any)
            negotiation_result: Negotiation result

        Returns:
            Bias analysis dictionary
        """
        if not self.enable_bias_detection:
            return {"bias_detected": False, "bias_type": None}

        price_range = retail_price - wholesale_price
        if price_range == 0:
            return {"bias_detected": False, "bias_type": "undefined_range"}

        # Seller dominance: Final price close to retail
        seller_dominance = False
        buyer_exploitation = False
        overpayment = False

        if final_price and negotiation_result == "accepted":
            # Calculate position in price range
            position = (final_price - wholesale_price) / price_range

            # Seller dominance if price > 70% of range (closer to retail)
            if position > 0.7:
                seller_dominance = True

            # Buyer exploitation if price < 30% of range (closer to wholesale)
            if position < 0.3:
                buyer_exploitation = True

            # Overpayment: Buyer pays above budget
            if budget and final_price > budget * (1 + self.constraint_violation_tolerance):
                overpayment = True

        # Calculate offer asymmetry (who moved more)
        if len(offer_history) >= 2:
            first_offer = offer_history[0]
            last_offer = offer_history[-1]
            total_movement = abs(last_offer - first_offer)
            seller_movement_ratio = total_movement / price_range if price_range > 0 else 0

            # Asymmetric if one party moved significantly more
            asymmetric_concession = seller_movement_ratio > self.bias_asymmetry_threshold
        else:
            asymmetric_concession = False
            seller_movement_ratio = 0.0

        bias_detected = seller_dominance or buyer_exploitation or overpayment or asymmetric_concession

        bias_analysis = {
            "bias_detected": bias_detected,
            "seller_dominance": seller_dominance,
            "buyer_exploitation": buyer_exploitation,
            "overpayment": overpayment,
            "asymmetric_concession": asymmetric_concession,
            "seller_movement_ratio": seller_movement_ratio,
            "price_position": (final_price - wholesale_price) / price_range if final_price and price_range > 0 else None,
            "recommendation": "review_fairness" if bias_detected else "fair_negotiation"
        }

        if bias_detected:
            self.bias_warnings += 1
            self.risk_flags.append({
                "type": "bias",
                "details": bias_analysis
            })

            bias_types = []
            if seller_dominance:
                bias_types.append("seller_dominance")
            if buyer_exploitation:
                bias_types.append("buyer_exploitation")
            if overpayment:
                bias_types.append("overpayment")
            if asymmetric_concession:
                bias_types.append("asymmetric_concession")

            logger.warning(f"[RMS] Bias detected: {', '.join(bias_types)}")

        return bias_analysis

    def detect_constraint_violations(
        self,
        current_offer: float,
        budget: Optional[float],
        wholesale_price: float,
        role: str
    ) -> Dict[str, Any]:
        """
        Detect constraint violations in real-time

        Args:
            current_offer: Current price offer
            budget: Buyer budget
            wholesale_price: Wholesale price
            role: Agent role

        Returns:
            Violation analysis dictionary
        """
        if not self.enable_constraint_violation:
            return {"violation_detected": False}

        violations = []

        if role == "buyer":
            # Buyer exceeding budget
            if budget and current_offer > budget * (1 + self.constraint_violation_tolerance):
                violations.append({
                    "type": "budget_exceeded",
                    "severity": "high",
                    "current_offer": current_offer,
                    "budget": budget,
                    "excess_amount": current_offer - budget,
                    "excess_percentage": ((current_offer - budget) / budget) * 100
                })
        elif role == "seller":
            # Seller going below wholesale
            if current_offer < wholesale_price * (1 - self.constraint_violation_tolerance):
                violations.append({
                    "type": "below_wholesale",
                    "severity": "high",
                    "current_offer": current_offer,
                    "wholesale_price": wholesale_price,
                    "deficit_amount": wholesale_price - current_offer,
                    "deficit_percentage": ((wholesale_price - current_offer) / wholesale_price) * 100
                })

        violation_detected = len(violations) > 0

        violation_analysis = {
            "violation_detected": violation_detected,
            "violations": violations,
            "violation_count": len(violations),
            "recommendation": "halt_and_review" if violation_detected else "continue"
        }

        if violation_detected:
            self.violation_warnings += 1
            self.risk_flags.append({
                "type": "constraint_violation",
                "details": violation_analysis
            })
            logger.warning(
                f"[RMS] Constraint violation detected for {role}: {violations}"
            )

        return violation_analysis

    def comprehensive_risk_assessment(
        self,
        offer_history: List[float],
        conversation_history: List[Dict[str, str]],
        current_turn: int,
        retail_price: float,
        wholesale_price: float,
        budget: Optional[float],
        final_price: Optional[float],
        negotiation_result: Optional[str],
        current_offer: float,
        role: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment

        Args:
            All negotiation state parameters

        Returns:
            Comprehensive risk assessment
        """
        # Run all detection modules
        deadlock_analysis = self.detect_deadlock(
            offer_history, conversation_history, current_turn
        )

        bias_analysis = self.detect_bias(
            offer_history, retail_price, wholesale_price,
            budget, final_price, negotiation_result
        )

        violation_analysis = self.detect_constraint_violations(
            current_offer, budget, wholesale_price, role
        )

        # Compute overall risk score
        risk_score = 0.0
        if deadlock_analysis["deadlock_detected"]:
            risk_score += 0.4 * deadlock_analysis["confidence"]
        if bias_analysis["bias_detected"]:
            risk_score += 0.3
        if violation_analysis["violation_detected"]:
            risk_score += 0.5

        risk_score = min(1.0, risk_score)

        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        comprehensive_assessment = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "deadlock_analysis": deadlock_analysis,
            "bias_analysis": bias_analysis,
            "violation_analysis": violation_analysis,
            "total_flags": len(self.risk_flags),
            "recommendation": self._generate_risk_recommendation(
                deadlock_analysis, bias_analysis, violation_analysis
            )
        }

        logger.info(
            f"[RMS] Comprehensive risk assessment: "
            f"Level={risk_level}, Score={risk_score:.2f}"
        )

        return comprehensive_assessment

    def _generate_risk_recommendation(
        self,
        deadlock_analysis: Dict[str, Any],
        bias_analysis: Dict[str, Any],
        violation_analysis: Dict[str, Any]
    ) -> str:
        """Generate recommendation based on risk analyses"""
        if violation_analysis["violation_detected"]:
            return "immediate_halt_constraint_violated"
        elif deadlock_analysis["deadlock_detected"] and deadlock_analysis["confidence"] > 0.7:
            return "mediate_or_terminate"
        elif bias_analysis["bias_detected"]:
            return "review_fairness_and_continue"
        else:
            return "continue_with_monitoring"

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of all risk flags"""
        return {
            "total_risk_flags": len(self.risk_flags),
            "deadlock_warnings": self.deadlock_warnings,
            "bias_warnings": self.bias_warnings,
            "violation_warnings": self.violation_warnings,
            "risk_flags": self.risk_flags
        }
