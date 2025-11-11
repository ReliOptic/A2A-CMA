"""
CMA (Causal Mediation Agent) Package

온폴리시 A2A 조정 유닛
On-Policy A2A Coordination Unit

CMA는 A2A 환경에서 온폴리시 상태로 작동하며, 메인 에이전트의 의사결정 흐름을
끊지 않고 인과적 피드백 루프를 형성합니다.

CMA operates in an on-policy state in A2A environments, forming a causal feedback
loop without interrupting the main agent's decision-making flow.

Core Modules:
- SAC (Semantic Alignment Core): Goal and constraint alignment
- EG (Exploration Governor): Exploration bounds and convergence control
- RMS (Risk Monitoring Sentinel): Risk detection and mitigation
- CLX (Causal Logger & Explainer): Causal logging for self-reinforcement
- APA (A2A Protocol Adapter): A2A protocol compliance

Usage:
    from cma import CausalMediationAgent

    cma = CausalMediationAgent()
    cma.start_session(product_data, buyer_model, seller_model, budget)

    # During negotiation
    prehook_result = cma.execute_prehook(turn, role, ...)
    posthook_result = cma.execute_posthook(turn, role, ...)

    # End session
    summary = cma.end_session(result, price, turns)
"""

from .core import CausalMediationAgent
from .config import CMAConfig, DEFAULT_CMA_CONFIG
from .hooks import CMAHooks
from .modules import (
    SemanticAlignmentCore,
    ExplorationGovernor,
    RiskMonitoringSentinel,
    CausalLoggerExplainer,
    A2AProtocolAdapter
)

__version__ = "1.0.0"
__author__ = "CMA Development Team"
__all__ = [
    'CausalMediationAgent',
    'CMAConfig',
    'DEFAULT_CMA_CONFIG',
    'CMAHooks',
    'SemanticAlignmentCore',
    'ExplorationGovernor',
    'RiskMonitoringSentinel',
    'CausalLoggerExplainer',
    'A2AProtocolAdapter'
]
