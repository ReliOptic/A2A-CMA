"""
APA (A2A Protocol Adapter) Module

Responsible for:
- A2A protocol standardization (Google A2A format)
- Agent Card generation
- Handshake protocol
- Trace ID management
- Message signing and verification (optional)
"""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class A2AProtocolAdapter:
    """A2A Protocol Adapter - Ensures A2A standard compliance"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize APA module

        Args:
            config: Configuration dictionary for APA
        """
        self.config = config
        self.protocol_version = config.get("protocol_version", "1.0")
        self.enable_agent_card = config.get("enable_agent_card", True)
        self.enable_handshake = config.get("enable_handshake", True)
        self.enable_trace_id = config.get("enable_trace_id", True)
        self.enable_message_signing = config.get("enable_message_signing", False)
        self.enable_message_verification = config.get("enable_message_verification", False)
        self.message_format = config.get("message_format", "json")
        self.compatibility_mode = config.get("compatibility_mode", "google_a2a")

        # Protocol state
        self.agent_id = self._generate_agent_id()
        self.session_trace_id = self._generate_trace_id()
        self.handshake_completed = False
        self.agent_card = None
        self.message_count = 0

    def _generate_agent_id(self) -> str:
        """Generate unique agent ID"""
        return f"cma_agent_{uuid.uuid4().hex[:12]}"

    def _generate_trace_id(self) -> str:
        """Generate A2A trace ID"""
        return f"a2a_trace_{uuid.uuid4()}"

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        self.message_count += 1
        return f"msg_{self.agent_id}_{self.message_count}_{uuid.uuid4().hex[:8]}"

    def create_agent_card(
        self,
        agent_role: str,
        capabilities: List[str],
        constraints: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create A2A Agent Card

        Args:
            agent_role: Role of the agent (buyer, seller, mediator)
            capabilities: List of agent capabilities
            constraints: Agent constraints
            metadata: Additional metadata

        Returns:
            Agent Card dictionary
        """
        if not self.enable_agent_card:
            return {}

        agent_card = {
            "agent_id": self.agent_id,
            "agent_type": "CMA",
            "agent_version": "1.0.0",
            "protocol_version": self.protocol_version,
            "role": agent_role,
            "capabilities": capabilities,
            "constraints": constraints,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        if self.compatibility_mode == "google_a2a":
            # Add Google A2A specific fields
            agent_card["provider"] = "CMA"
            agent_card["schema_version"] = self.protocol_version

        self.agent_card = agent_card

        logger.info(f"[APA] Created Agent Card for {agent_role}")

        return agent_card

    def create_handshake_message(
        self,
        peer_agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create A2A handshake message

        Args:
            peer_agent_id: ID of peer agent (if known)

        Returns:
            Handshake message
        """
        if not self.enable_handshake:
            return {}

        handshake = {
            "message_type": "handshake",
            "message_id": self._generate_message_id(),
            "from_agent_id": self.agent_id,
            "to_agent_id": peer_agent_id,
            "trace_id": self.session_trace_id,
            "timestamp": datetime.now().isoformat(),
            "protocol_version": self.protocol_version,
            "agent_card": self.agent_card,
            "handshake_data": {
                "supported_protocols": ["google_a2a_v1"],
                "message_formats": ["json"],
                "capabilities": self.agent_card.get("capabilities", []) if self.agent_card else []
            }
        }

        logger.info(f"[APA] Created handshake message for peer: {peer_agent_id}")

        return handshake

    def process_handshake_response(
        self,
        handshake_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process handshake response from peer agent

        Args:
            handshake_response: Handshake response message

        Returns:
            Processed handshake result
        """
        if not self.enable_handshake:
            return {"success": True, "message": "Handshake disabled"}

        # Validate handshake response
        required_fields = ["message_type", "from_agent_id", "protocol_version"]
        missing_fields = [f for f in required_fields if f not in handshake_response]

        if missing_fields:
            logger.error(f"[APA] Invalid handshake response: missing {missing_fields}")
            return {
                "success": False,
                "error": f"Missing required fields: {missing_fields}"
            }

        # Check protocol compatibility
        peer_protocol = handshake_response.get("protocol_version")
        if peer_protocol != self.protocol_version:
            logger.warning(
                f"[APA] Protocol version mismatch: "
                f"ours={self.protocol_version}, peer={peer_protocol}"
            )

        self.handshake_completed = True

        result = {
            "success": True,
            "peer_agent_id": handshake_response.get("from_agent_id"),
            "peer_protocol_version": peer_protocol,
            "peer_capabilities": handshake_response.get("handshake_data", {}).get("capabilities", [])
        }

        logger.info(f"[APA] Handshake completed with {result['peer_agent_id']}")

        return result

    def wrap_message_a2a(
        self,
        content: str,
        message_type: str = "negotiation",
        metadata: Optional[Dict[str, Any]] = None,
        to_agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wrap a message in A2A protocol format

        Args:
            content: Message content
            message_type: Type of message
            metadata: Additional metadata
            to_agent_id: Target agent ID

        Returns:
            A2A-formatted message
        """
        message_id = self._generate_message_id()

        a2a_message = {
            "message_id": message_id,
            "message_type": message_type,
            "from_agent_id": self.agent_id,
            "to_agent_id": to_agent_id,
            "trace_id": self.session_trace_id if self.enable_trace_id else None,
            "timestamp": datetime.now().isoformat(),
            "protocol_version": self.protocol_version,
            "content": content,
            "metadata": metadata or {}
        }

        # Add message signature if enabled
        if self.enable_message_signing:
            a2a_message["signature"] = self._sign_message(a2a_message)

        logger.debug(f"[APA] Wrapped message in A2A format: {message_id}")

        return a2a_message

    def unwrap_a2a_message(
        self,
        a2a_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Unwrap A2A message and extract content

        Args:
            a2a_message: A2A-formatted message

        Returns:
            Unwrapped message data
        """
        # Verify message signature if enabled
        if self.enable_message_verification and "signature" in a2a_message:
            is_valid = self._verify_message_signature(a2a_message)
            if not is_valid:
                logger.warning(f"[APA] Message signature verification failed")
                return {
                    "success": False,
                    "error": "Signature verification failed"
                }

        unwrapped = {
            "success": True,
            "message_id": a2a_message.get("message_id"),
            "content": a2a_message.get("content"),
            "from_agent_id": a2a_message.get("from_agent_id"),
            "message_type": a2a_message.get("message_type"),
            "metadata": a2a_message.get("metadata", {})
        }

        return unwrapped

    def _sign_message(self, message: Dict[str, Any]) -> str:
        """
        Create signature for message (simple hash-based)

        Args:
            message: Message to sign

        Returns:
            Message signature
        """
        # Create a copy without signature field
        msg_copy = {k: v for k, v in message.items() if k != "signature"}

        # Convert to canonical JSON string
        msg_str = json.dumps(msg_copy, sort_keys=True)

        # Create hash signature
        signature = hashlib.sha256(msg_str.encode()).hexdigest()

        return signature

    def _verify_message_signature(self, message: Dict[str, Any]) -> bool:
        """
        Verify message signature

        Args:
            message: Message with signature

        Returns:
            True if signature is valid
        """
        if "signature" not in message:
            return False

        provided_signature = message["signature"]
        computed_signature = self._sign_message(message)

        return provided_signature == computed_signature

    def create_a2a_session_metadata(
        self,
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create A2A session metadata

        Args:
            session_data: Session information

        Returns:
            A2A session metadata
        """
        metadata = {
            "session_id": session_data.get("session_id"),
            "trace_id": self.session_trace_id,
            "agent_id": self.agent_id,
            "protocol_version": self.protocol_version,
            "session_start": datetime.now().isoformat(),
            "compatibility_mode": self.compatibility_mode,
            "session_data": session_data
        }

        return metadata

    def validate_a2a_compliance(
        self,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate if message is A2A compliant

        Args:
            message: Message to validate

        Returns:
            Validation result
        """
        required_fields = [
            "message_id", "message_type", "from_agent_id",
            "timestamp", "protocol_version"
        ]

        missing_fields = [f for f in required_fields if f not in message]

        if missing_fields:
            return {
                "compliant": False,
                "missing_fields": missing_fields,
                "error": f"Missing required A2A fields: {missing_fields}"
            }

        # Check trace ID if enabled
        if self.enable_trace_id and "trace_id" not in message:
            return {
                "compliant": False,
                "error": "Missing required trace_id"
            }

        return {
            "compliant": True,
            "message": "Message is A2A compliant"
        }

    def get_protocol_summary(self) -> Dict[str, Any]:
        """Get summary of protocol state"""
        return {
            "agent_id": self.agent_id,
            "session_trace_id": self.session_trace_id,
            "protocol_version": self.protocol_version,
            "handshake_completed": self.handshake_completed,
            "message_count": self.message_count,
            "compatibility_mode": self.compatibility_mode
        }
