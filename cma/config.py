"""
CMA (Causal Mediation Agent) Configuration

This module contains configuration parameters for the CMA system.
"""

from typing import Dict, Any


class CMAConfig:
    """Configuration for Causal Mediation Agent"""

    def __init__(self):
        # SAC (Semantic Alignment Core) Configuration
        self.sac_config = {
            "alignment_threshold": 0.7,  # Minimum alignment score
            "semantic_similarity_model": "text-embedding-ada-002",
            "enable_goal_restatement": True,
            "detect_semantic_drift": True
        }

        # EG (Exploration Governor) Configuration
        self.eg_config = {
            "min_turns": 3,  # Minimum negotiation turns before accepting
            "max_turns": 30,  # Maximum turns (safety limit)
            "offer_span_threshold": 0.1,  # Minimum price exploration range (10%)
            "early_termination_prevention": True,
            "exploration_pressure_decay": 0.9  # Decay factor per turn
        }

        # RMS (Risk Monitoring Sentinel) Configuration
        self.rms_config = {
            "enable_deadlock_detection": True,
            "enable_bias_detection": True,
            "enable_constraint_violation_detection": True,
            "deadlock_repetition_threshold": 3,  # Same offer N times = deadlock
            "bias_asymmetry_threshold": 0.3,  # 30% advantage = bias
            "constraint_violation_tolerance": 0.05,  # 5% budget overage allowed
            "monitoring_interval": 1  # Check every N turns
        }

        # CLX (Causal Logger & Explainer) Configuration
        self.clx_config = {
            "enable_causal_logging": True,
            "enable_counterfactual_tagging": True,
            "log_format": "jsonl",
            "log_dir": "cma_logs",
            "include_state_snapshots": True,
            "include_intervention_traces": True,
            "a2a_trace_id_enabled": True
        }

        # APA (A2A Protocol Adapter) Configuration
        self.apa_config = {
            "protocol_version": "1.0",
            "enable_agent_card": True,
            "enable_handshake": True,
            "enable_trace_id": True,
            "enable_message_signing": False,  # Optional security feature
            "enable_message_verification": False,
            "message_format": "json",
            "compatibility_mode": "google_a2a"  # Google A2A standard
        }

        # General CMA Configuration
        self.general_config = {
            "on_policy_mode": True,  # Run during negotiation (on-policy)
            "enable_prehook": True,
            "enable_posthook": True,
            "enable_feedback_loop": True,
            "feedback_learning_rate": 0.01,
            "verbose": True,
            "log_level": "INFO"
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "sac": self.sac_config,
            "eg": self.eg_config,
            "rms": self.rms_config,
            "clx": self.clx_config,
            "apa": self.apa_config,
            "general": self.general_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CMAConfig':
        """Create configuration from dictionary"""
        config = cls()
        if "sac" in config_dict:
            config.sac_config.update(config_dict["sac"])
        if "eg" in config_dict:
            config.eg_config.update(config_dict["eg"])
        if "rms" in config_dict:
            config.rms_config.update(config_dict["rms"])
        if "clx" in config_dict:
            config.clx_config.update(config_dict["clx"])
        if "apa" in config_dict:
            config.apa_config.update(config_dict["apa"])
        if "general" in config_dict:
            config.general_config.update(config_dict["general"])
        return config


# Default configuration instance
DEFAULT_CMA_CONFIG = CMAConfig()
