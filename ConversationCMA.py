"""
ConversationCMA - CMA-Enhanced Conversation Class

Extends the base Conversation class with Causal Mediation Agent (CMA) capabilities.
Provides on-policy integration of CMA into A2A negotiation flow.
"""

import json
import logging
from typing import Optional
from Conversation import Conversation
from cma import CausalMediationAgent, CMAConfig

logger = logging.getLogger(__name__)


class ConversationCMA(Conversation):
    """
    CMA-enhanced Conversation class

    Integrates Causal Mediation Agent for:
    - On-policy monitoring and intervention
    - Semantic alignment checking
    - Exploration governance
    - Risk detection and mitigation
    - Causal logging for self-reinforcement
    """

    def __init__(
        self,
        product_data,
        buyer_model="gpt-3.5-turbo",
        seller_model="gpt-3.5-turbo",
        summary_model="gpt-3.5-turbo",
        max_turns=20,
        experiment_num=0,
        budget=None,
        cma_config: Optional[CMAConfig] = None,
        enable_cma=True
    ):
        """
        Initialize CMA-enhanced Conversation

        Args:
            product_data: Product information
            buyer_model: Buyer model name
            seller_model: Seller model name
            summary_model: Summary model name
            max_turns: Maximum negotiation turns
            experiment_num: Experiment number
            budget: Buyer budget
            cma_config: CMA configuration (optional)
            enable_cma: Enable/disable CMA (default: True)
        """
        # Initialize parent class
        super().__init__(
            product_data=product_data,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            experiment_num=experiment_num,
            budget=budget
        )

        # Initialize CMA
        self.enable_cma = enable_cma
        if self.enable_cma:
            self.cma = CausalMediationAgent(config=cma_config)
            logger.info("[ConversationCMA] CMA initialized")

            # Start CMA session
            self.cma_session = self.cma.start_session(
                product_data=product_data,
                buyer_model=buyer_model,
                seller_model=seller_model,
                budget=budget,
                experiment_num=experiment_num
            )
            logger.info(
                f"[ConversationCMA] CMA session started: "
                f"trace_id={self.cma_session['a2a_trace_id']}"
            )
        else:
            self.cma = None
            self.cma_session = None
            logger.info("[ConversationCMA] CMA disabled")

        # CMA tracking
        self.cma_prehook_results = []
        self.cma_posthook_results = []
        self.cma_interventions = []

    def run_negotiation_with_cma(self):
        """
        Run negotiation with CMA integration

        This method extends the parent run_negotiation() with CMA hooks.
        """
        print("\nStarting CMA-enhanced negotiation...")
        print("-" * 50)

        if self.enable_cma:
            print(f"CMA Session ID: {self.cma_session['session_id']}")
            print(f"A2A Trace ID: {self.cma_session['a2a_trace_id']}")
            print("-" * 50)

        # Generate buyer's introduction
        budget_info = f"\nYour maximum budget for this purchase is ${self.budget:.2f}." if self.budget is not None else ""

        intro_prompt = f"""You are a professional negotiation assistant aiming to purchase a product at the best possible price.

        Your task is to start the conversation naturally without revealing your role as a negotiation assistant.

        Please write a short and friendly message to the seller that:
        1. Expresses interest in the product and asks about the possibility of negotiating the price
        2. Sounds natural, polite, and engaging.

        Avoid over-explaining — just say "Hello" to start and smoothly lead into your interest.

        Product: {self.product_data['Product Name']}
        Retail Price: {self.product_data['Retail Price']}
        Features: {self.product_data['Features']}{budget_info}

        Keep the message concise and focused on opening the negotiation."""

        buyer_intro = self.buyer_model.get_response(intro_prompt)
        self.conversation_history.append({"speaker": "Buyer", "message": buyer_intro})
        print(f"\n[Initial] Buyer: {buyer_intro}")

        # Initialize current_price_offer
        self.current_price_offer = self.seller_price_offers[0]

        turn_count = 1

        # Main negotiation loop with CMA hooks
        while turn_count <= self.max_turns:
            # === PreHook: Execute before turn ===
            if self.enable_cma:
                prehook_result = self.cma.execute_prehook(
                    turn=turn_count,
                    role="seller",
                    conversation_history=self.conversation_history,
                    product_data=self.product_data,
                    budget=self.budget,
                    current_offer=self.current_price_offer
                )
                self.cma_prehook_results.append(prehook_result)

                # Log interventions
                if prehook_result.get("interventions"):
                    for intervention in prehook_result["interventions"]:
                        self.cma_interventions.append({
                            "turn": turn_count,
                            "phase": "prehook",
                            "intervention": intervention
                        })
                        logger.info(
                            f"[CMA Intervention] Turn {turn_count} PreHook: "
                            f"{intervention['type']} - {intervention['message']}"
                        )

            # Seller's turn
            seller_messages = self.format_seller_prompt()
            seller_response = self.seller_model.get_chat_response(seller_messages)
            self.conversation_history.append({"speaker": "Seller", "message": seller_response})
            print(f"\n[Turn {turn_count}] Seller: {seller_response}")

            # Extract price offer
            price_offer = self.extract_price_from_seller_message(seller_response)

            if price_offer:
                self.current_price_offer = price_offer

            # Update price history
            if turn_count >= len(self.seller_price_offers):
                self.seller_price_offers.append(self.current_price_offer)
            else:
                self.seller_price_offers[turn_count] = self.current_price_offer

            print(f"[Turn {turn_count}] Extracted Price Offer: {self.current_price_offer}")

            # Buyer's turn
            buyer_messages = self.format_buyer_prompt()
            buyer_response = self.buyer_model.get_chat_response(buyer_messages)
            self.conversation_history.append({"speaker": "Buyer", "message": buyer_response})
            print(f"\n[Turn {turn_count}] Buyer: {buyer_response}")

            # === PostHook: Execute after turn ===
            if self.enable_cma:
                posthook_result = self.cma.execute_posthook(
                    turn=turn_count,
                    role="both",
                    conversation_history=self.conversation_history,
                    product_data=self.product_data,
                    budget=self.budget,
                    offer_history=self.seller_price_offers,
                    negotiation_result=self.negotiation_result,
                    final_price=self.current_price_offer if self.negotiation_completed else None
                )
                self.cma_posthook_results.append(posthook_result)

                # Log alerts
                if posthook_result.get("alerts"):
                    for alert in posthook_result["alerts"]:
                        self.cma_interventions.append({
                            "turn": turn_count,
                            "phase": "posthook",
                            "alert": alert
                        })
                        logger.warning(
                            f"[CMA Alert] Turn {turn_count} PostHook: "
                            f"{alert['type']} ({alert['severity']}) - {alert['message']}"
                        )

            # Check if negotiation should end
            if self.evaluate_negotiation_state():
                print(f"\nNegotiation has concluded with result: {self.negotiation_result}")
                break

            turn_count += 1

        # Record completed turns
        self.completed_turns = turn_count

        # Handle max turns
        if turn_count > self.max_turns and not self.negotiation_completed:
            self.negotiation_completed = True
            self.negotiation_result = "max_turns_reached"
            print("\nNegotiation reached maximum allowed turns without natural conclusion.")

        print("\n" + "-" * 50)
        print("Negotiation process completed.")
        print(f"Turns completed: {self.completed_turns}")
        print(f"Negotiation result: {self.negotiation_result}")
        print(f"Final price offer: ${self.current_price_offer:.2f}")

        # === Finalize CMA Session ===
        if self.enable_cma:
            self.cma_summary = self.cma.end_session(
                negotiation_result=self.negotiation_result,
                final_price=self.current_price_offer if self.negotiation_result == "accepted" else None,
                total_turns=self.completed_turns
            )

            print("\n" + "=" * 50)
            print("CMA SESSION SUMMARY")
            print("=" * 50)
            print(f"Total CMA Interventions: {len(self.cma_interventions)}")
            print(f"Risk Flags: {self.cma_summary['rms_summary']['total_risk_flags']}")
            print(f"Alignment Checks: {len(self.cma_summary['sac_summary']['alignment_history'])}")
            print(f"Training Examples Generated: {self.cma_summary['training_examples']}")
            print(f"Causal Log: {self.cma_summary['causal_log_file']}")
            print(f"RLHF Data: {self.cma_summary['rlhf_file']}")
            print("=" * 50)

        return self.conversation_history

    def run_negotiation(self):
        """
        Override parent run_negotiation to use CMA-enhanced version
        """
        return self.run_negotiation_with_cma()

    def save_conversation(self, output_dir: str):
        """
        Save conversation with CMA data

        Args:
            output_dir: Output directory
        """
        # Call parent save
        super().save_conversation(output_dir)

        # Save CMA-specific data if enabled
        if self.enable_cma:
            import os
            output_file = os.path.join(
                output_dir,
                f"product_{self.product_id}_exp_{self.experiment_num}_cma.json"
            )

            cma_data = {
                "product_id": self.product_id,
                "experiment_num": self.experiment_num,
                "cma_session": self.cma_session,
                "cma_prehook_results": self.cma_prehook_results,
                "cma_posthook_results": self.cma_posthook_results,
                "cma_interventions": self.cma_interventions,
                "cma_summary": self.cma_summary if hasattr(self, 'cma_summary') else None,
                "cma_comprehensive_report": self.cma.generate_comprehensive_report()
            }

            with open(output_file, 'w') as f:
                json.dump(cma_data, f, indent=2)

            print(f"CMA data saved to: {output_file}")
