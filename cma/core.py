"""
CMA Core Module

Main orchestrator for Causal Mediation Agent (CMA).
Coordinates all CMA modules and provides unified interface.
"""

import logging
from typing import Dict, List, Any, Optional

from .config import CMAConfig, DEFAULT_CMA_CONFIG
from .modules import (
    SemanticAlignmentCore,
    ExplorationGovernor,
    RiskMonitoringSentinel,
    CausalLoggerExplainer,
    A2AProtocolAdapter
)
from .hooks import CMAHooks

logger = logging.getLogger(__name__)


class CausalMediationAgent:
    """
    Causal Mediation Agent (CMA)

    On-policy A2A coordination unit that provides:
    - Semantic alignment and goal monitoring
    - Exploration governance
    - Risk detection and mitigation
    - Causal logging for self-reinforcement
    - A2A protocol compliance
    """

    def __init__(self, config: Optional[CMAConfig] = None):
        """
        Initialize Causal Mediation Agent

        Args:
            config: CMA configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CMA_CONFIG

        # Initialize all modules
        logger.info("[CMA] Initializing Causal Mediation Agent modules")

        self.sac = SemanticAlignmentCore(self.config.sac_config)
        self.eg = ExplorationGovernor(self.config.eg_config)
        self.rms = RiskMonitoringSentinel(self.config.rms_config)
        self.clx = CausalLoggerExplainer(self.config.clx_config)
        self.apa = A2AProtocolAdapter(self.config.apa_config)

        # Initialize hooks
        self.hooks = CMAHooks(self)

        # Session state
        self.session_active = False
        self.session_metadata = {}

        logger.info("[CMA] Causal Mediation Agent initialized successfully")

    def start_session(
        self,
        product_data: Dict[str, Any],
        buyer_model: str,
        seller_model: str,
        budget: Optional[float] = None,
        experiment_num: int = 0
    ) -> Dict[str, Any]:
        """
        Start a new CMA session

        Args:
            product_data: Product information
            buyer_model: Buyer model name
            seller_model: Seller model name
            budget: Buyer budget
            experiment_num: Experiment number

        Returns:
            Session initialization result
        """
        logger.info("[CMA] Starting new CMA session")

        self.session_active = True

        # Initialize SAC with goals and constraints
        buyer_analysis = self.sac.analyze_goals_and_constraints(
            product_data, budget, "buyer"
        )
        seller_analysis = self.sac.analyze_goals_and_constraints(
            product_data, None, "seller"
        )

        # Create agent cards
        buyer_card = self.apa.create_agent_card(
            agent_role="buyer",
            capabilities=["price_negotiation", "budget_constraint"],
            constraints=buyer_analysis["constraints"],
            metadata={"model": buyer_model}
        )

        seller_card = self.apa.create_agent_card(
            agent_role="seller",
            capabilities=["price_negotiation", "value_positioning"],
            constraints=seller_analysis["constraints"],
            metadata={"model": seller_model}
        )

        # Create A2A session metadata
        self.session_metadata = self.apa.create_a2a_session_metadata({
            "product_id": product_data.get("id"),
            "experiment_num": experiment_num,
            "buyer_model": buyer_model,
            "seller_model": seller_model,
            "budget": budget
        })

        # Log session start
        self.clx.log_state_snapshot(0, "session_start", {
            "product_data": product_data,
            "buyer_model": buyer_model,
            "seller_model": seller_model,
            "budget": budget,
            "experiment_num": experiment_num
        })

        session_result = {
            "session_id": self.clx.session_id,
            "a2a_trace_id": self.clx.a2a_trace_id,
            "buyer_analysis": buyer_analysis,
            "seller_analysis": seller_analysis,
            "buyer_card": buyer_card,
            "seller_card": seller_card,
            "session_metadata": self.session_metadata
        }

        logger.info(
            f"[CMA] Session started: "
            f"session_id={self.clx.session_id}, "
            f"trace_id={self.clx.a2a_trace_id}"
        )

        return session_result

    def execute_prehook(
        self,
        turn: int,
        role: str,
        conversation_history: List[Dict[str, str]],
        product_data: Dict[str, Any],
        budget: Optional[float],
        current_offer: float
    ) -> Dict[str, Any]:
        """
        Execute PreHook before negotiation turn

        Args:
            turn: Current turn number
            role: Agent role
            conversation_history: Conversation history
            product_data: Product data
            budget: Budget constraint
            current_offer: Current price offer

        Returns:
            PreHook execution result
        """
        return self.hooks.prehook(
            turn, role, conversation_history,
            product_data, budget, current_offer
        )

    def execute_posthook(
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
        Execute PostHook after negotiation turn

        Args:
            turn: Current turn number
            role: Agent role
            conversation_history: Conversation history
            product_data: Product data
            budget: Budget constraint
            offer_history: Price offer history
            negotiation_result: Negotiation result
            final_price: Final price

        Returns:
            PostHook execution result
        """
        return self.hooks.posthook(
            turn, role, conversation_history, product_data,
            budget, offer_history, negotiation_result, final_price
        )

    def end_session(
        self,
        negotiation_result: str,
        final_price: Optional[float],
        total_turns: int
    ) -> Dict[str, Any]:
        """
        End CMA session and finalize logging

        Args:
            negotiation_result: Final negotiation result
            final_price: Final price (if applicable)
            total_turns: Total number of turns

        Returns:
            Session summary
        """
        logger.info("[CMA] Ending CMA session")

        session_summary = self.hooks.finalize_session(
            negotiation_result, final_price, total_turns
        )

        self.session_active = False

        return session_summary

    def get_module_status(self) -> Dict[str, Any]:
        """
        Get status of all CMA modules

        Returns:
            Status dictionary for all modules
        """
        return {
            "session_active": self.session_active,
            "sac": {
                "enabled": True,
                "alignment_checks": len(self.sac.alignment_history)
            },
            "eg": {
                "enabled": True,
                "current_turn": self.eg.turn_count,
                "exploration_pressure": self.eg.exploration_pressure
            },
            "rms": {
                "enabled": True,
                "risk_flags": len(self.rms.risk_flags),
                "deadlock_warnings": self.rms.deadlock_warnings,
                "bias_warnings": self.rms.bias_warnings
            },
            "clx": {
                "enabled": True,
                "total_logs": len(self.clx.causal_log),
                "interventions": self.clx.intervention_count
            },
            "apa": {
                "enabled": True,
                "handshake_completed": self.apa.handshake_completed,
                "message_count": self.apa.message_count
            }
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive CMA report

        Returns:
            Complete CMA analysis report
        """
        return {
            "session_metadata": self.session_metadata,
            "module_status": self.get_module_status(),
            "alignment_summary": self.sac.get_alignment_summary(),
            "exploration_summary": self.eg.get_exploration_summary(),
            "risk_summary": self.rms.get_risk_summary(),
            "logging_summary": self.clx.get_session_summary(),
            "protocol_summary": self.apa.get_protocol_summary(),
            "causal_trace": self.clx.generate_causal_trace()
        }

    def __repr__(self) -> str:
        return (
            f"CausalMediationAgent("
            f"session_active={self.session_active}, "
            f"session_id={self.clx.session_id if hasattr(self.clx, 'session_id') else 'None'}"
            f")"
        )
