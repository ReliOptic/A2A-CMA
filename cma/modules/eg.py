"""
EG (Exploration Governor) Module

Responsible for:
- Setting minimum turn thresholds
- Preventing premature deal acceptance
- Managing exploration span (price range)
- Controlling convergence rate
"""

import logging
from typing import Dict, List, Any, Optional
import math

logger = logging.getLogger(__name__)


class ExplorationGovernor:
    """Exploration Governor - Manages exploration bounds and convergence"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EG module

        Args:
            config: Configuration dictionary for EG
        """
        self.config = config
        self.min_turns = config.get("min_turns", 3)
        self.max_turns = config.get("max_turns", 30)
        self.offer_span_threshold = config.get("offer_span_threshold", 0.1)
        self.early_termination_prevention = config.get("early_termination_prevention", True)
        self.exploration_pressure_decay = config.get("exploration_pressure_decay", 0.9)

        # State tracking
        self.turn_count = 0
        self.offer_history = []
        self.exploration_pressure = 1.0  # Starts high, decays over time

    def should_prevent_early_termination(
        self,
        current_turn: int,
        negotiation_result: Optional[str]
    ) -> bool:
        """
        Determine if early termination should be prevented

        Args:
            current_turn: Current turn number
            negotiation_result: Proposed negotiation result (accepted/rejected/None)

        Returns:
            True if termination should be prevented
        """
        if not self.early_termination_prevention:
            return False

        # Prevent acceptance before minimum turns
        if current_turn < self.min_turns and negotiation_result == "accepted":
            logger.info(f"[EG] Preventing early acceptance at turn {current_turn} (min: {self.min_turns})")
            return True

        return False

    def check_exploration_adequacy(
        self,
        offer_history: List[float],
        retail_price: float,
        wholesale_price: float
    ) -> Dict[str, Any]:
        """
        Check if price exploration has been adequate

        Args:
            offer_history: List of all price offers
            retail_price: Retail price of product
            wholesale_price: Wholesale price of product

        Returns:
            Dictionary with exploration adequacy analysis
        """
        if len(offer_history) < 2:
            return {
                "is_adequate": False,
                "reason": "insufficient_offers",
                "exploration_span": 0.0,
                "required_span": self.offer_span_threshold
            }

        # Calculate price range explored
        max_offer = max(offer_history)
        min_offer = min(offer_history)
        price_range = retail_price - wholesale_price

        if price_range == 0:
            exploration_span = 0.0
        else:
            exploration_span = (max_offer - min_offer) / price_range

        required_span = self.offer_span_threshold
        is_adequate = exploration_span >= required_span

        analysis = {
            "is_adequate": is_adequate,
            "exploration_span": exploration_span,
            "required_span": required_span,
            "max_offer": max_offer,
            "min_offer": min_offer,
            "price_range": price_range,
            "recommendation": "continue_exploration" if not is_adequate else "can_converge"
        }

        if not is_adequate:
            logger.info(
                f"[EG] Inadequate exploration: {exploration_span:.2%} "
                f"(required: {required_span:.2%})"
            )

        return analysis

    def compute_exploration_pressure(
        self,
        current_turn: int
    ) -> float:
        """
        Compute current exploration pressure (decays over turns)

        Args:
            current_turn: Current turn number

        Returns:
            Exploration pressure value (0.0 to 1.0)
        """
        # Exponential decay
        pressure = math.exp(-current_turn / (self.max_turns * 0.3))

        # Apply decay factor
        pressure *= (self.exploration_pressure_decay ** current_turn)

        # Clamp to [0, 1]
        pressure = max(0.0, min(1.0, pressure))

        self.exploration_pressure = pressure
        return pressure

    def recommend_exploration_action(
        self,
        current_turn: int,
        offer_history: List[float],
        retail_price: float,
        wholesale_price: float,
        role: str
    ) -> Dict[str, Any]:
        """
        Recommend whether agent should explore more or converge

        Args:
            current_turn: Current turn number
            offer_history: Price offer history
            retail_price: Retail price
            wholesale_price: Wholesale price
            role: Agent role (buyer or seller)

        Returns:
            Recommendation dictionary
        """
        # Check turn threshold
        below_min_turns = current_turn < self.min_turns

        # Check exploration adequacy
        exploration_analysis = self.check_exploration_adequacy(
            offer_history, retail_price, wholesale_price
        )

        # Compute exploration pressure
        pressure = self.compute_exploration_pressure(current_turn)

        # Make recommendation
        if below_min_turns:
            recommendation = "explore"
            reason = f"Below minimum turns ({current_turn}/{self.min_turns})"
        elif not exploration_analysis["is_adequate"]:
            recommendation = "explore"
            reason = f"Inadequate exploration span ({exploration_analysis['exploration_span']:.2%})"
        elif pressure > 0.3:
            recommendation = "explore"
            reason = f"High exploration pressure ({pressure:.2f})"
        else:
            recommendation = "converge"
            reason = "Adequate exploration completed"

        result = {
            "recommendation": recommendation,
            "reason": reason,
            "exploration_pressure": pressure,
            "current_turn": current_turn,
            "min_turns": self.min_turns,
            "exploration_analysis": exploration_analysis
        }

        logger.info(
            f"[EG] Exploration recommendation for {role}: {recommendation} "
            f"(turn {current_turn}, pressure {pressure:.2f})"
        )

        return result

    def get_turn_limits(self) -> Dict[str, int]:
        """Get configured turn limits"""
        return {
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "current_turn": self.turn_count
        }

    def update_turn_count(self, turn: int):
        """Update internal turn counter"""
        self.turn_count = turn

    def should_force_exploration(
        self,
        current_turn: int,
        offer_history: List[float],
        last_offer_delta: float
    ) -> bool:
        """
        Determine if exploration should be forced (prevent quick convergence)

        Args:
            current_turn: Current turn number
            offer_history: Price offer history
            last_offer_delta: Change in last offer

        Returns:
            True if exploration should be forced
        """
        # Force exploration if:
        # 1. Below minimum turns
        if current_turn < self.min_turns:
            return True

        # 2. Offers converging too quickly
        if len(offer_history) >= 3:
            recent_offers = offer_history[-3:]
            variance = max(recent_offers) - min(recent_offers)
            avg_offer = sum(recent_offers) / len(recent_offers)

            # If variance is less than 1% of average, force exploration
            if avg_offer > 0 and (variance / avg_offer) < 0.01:
                logger.info("[EG] Forcing exploration due to rapid convergence")
                return True

        # 3. Very small price changes (stagnation)
        if abs(last_offer_delta) < 10:  # Less than $10 change
            if current_turn < self.min_turns * 2:
                logger.info("[EG] Forcing exploration due to small price changes")
                return True

        return False

    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of exploration governance"""
        return {
            "total_turns": self.turn_count,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "current_exploration_pressure": self.exploration_pressure,
            "offer_history_length": len(self.offer_history)
        }
