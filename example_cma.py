"""
Example: Using Causal Mediation Agent (CMA) with A2A Negotiation

This script demonstrates how to use CMA for enhanced A2A negotiation
with on-policy monitoring and causal logging.
"""

import json
import argparse
from ConversationCMA import ConversationCMA
from cma import CMAConfig


def run_cma_example():
    """Run a simple CMA-enhanced negotiation example"""

    # Load sample product data
    with open("dataset/products_mini.json", "r") as f:
        products = json.load(f)

    # Select first product
    product = products[0]
    print(f"\n{'='*60}")
    print(f"Product: {product['Product Name']}")
    print(f"Retail Price: {product['Retail Price']}")
    print(f"Wholesale Price: {product['Wholesale Price']}")
    print(f"{'='*60}\n")

    # Calculate budget (mid-range)
    retail_price = float(product["Retail Price"].replace("$", "").replace(",", ""))
    wholesale_price = float(product["Wholesale Price"].replace("$", "").replace(",", ""))
    budget = (retail_price + wholesale_price) / 2

    print(f"Buyer Budget: ${budget:.2f}\n")

    # Create CMA configuration
    cma_config = CMAConfig()
    cma_config.eg_config["min_turns"] = 3
    cma_config.rms_config["deadlock_repetition_threshold"] = 3
    cma_config.clx_config["enable_causal_logging"] = True

    # Create CMA-enhanced conversation
    conversation = ConversationCMA(
        product_data=product,
        buyer_model="gpt-3.5-turbo",
        seller_model="gpt-3.5-turbo",
        summary_model="gpt-3.5-turbo",
        max_turns=15,
        experiment_num=0,
        budget=budget,
        cma_config=cma_config,
        enable_cma=True
    )

    # Run negotiation with CMA
    print("\n" + "="*60)
    print("STARTING CMA-ENHANCED NEGOTIATION")
    print("="*60 + "\n")

    conversation.run_negotiation()

    # Save results
    output_dir = "results_cma_example"
    conversation.save_conversation(output_dir)

    print("\n" + "="*60)
    print("NEGOTIATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"CMA logs saved to: cma_logs/")

    # Print CMA summary
    if hasattr(conversation, 'cma_summary'):
        print("\n" + "="*60)
        print("CMA SUMMARY")
        print("="*60)
        print(f"Session ID: {conversation.cma_summary['clx_summary']['session_id']}")
        print(f"Total Interventions: {len(conversation.cma_interventions)}")
        print(f"Risk Flags: {conversation.cma_summary['rms_summary']['total_risk_flags']}")
        print(f"Training Examples: {conversation.cma_summary['training_examples']}")
        print(f"\nCausal Log: {conversation.cma_summary['causal_log_file']}")
        print(f"RLHF Data: {conversation.cma_summary['rlhf_file']}")
        print("="*60 + "\n")


def run_cma_comparison():
    """Run comparison between with/without CMA"""

    print("\n" + "="*60)
    print("CMA COMPARISON: WITH vs WITHOUT")
    print("="*60 + "\n")

    # Load sample product
    with open("dataset/products_mini.json", "r") as f:
        products = json.load(f)

    product = products[0]
    retail_price = float(product["Retail Price"].replace("$", "").replace(",", ""))
    wholesale_price = float(product["Wholesale Price"].replace("$", "").replace(",", ""))
    budget = (retail_price + wholesale_price) / 2

    results = {}

    # Run WITHOUT CMA
    print("\n[1/2] Running WITHOUT CMA...")
    conv_no_cma = ConversationCMA(
        product_data=product,
        buyer_model="gpt-3.5-turbo",
        seller_model="gpt-3.5-turbo",
        summary_model="gpt-3.5-turbo",
        max_turns=15,
        experiment_num=0,
        budget=budget,
        enable_cma=False
    )
    conv_no_cma.run_negotiation()

    results["without_cma"] = {
        "result": conv_no_cma.negotiation_result,
        "final_price": conv_no_cma.current_price_offer,
        "turns": conv_no_cma.completed_turns
    }

    # Run WITH CMA
    print("\n[2/2] Running WITH CMA...")
    conv_with_cma = ConversationCMA(
        product_data=product,
        buyer_model="gpt-3.5-turbo",
        seller_model="gpt-3.5-turbo",
        summary_model="gpt-3.5-turbo",
        max_turns=15,
        experiment_num=1,
        budget=budget,
        enable_cma=True
    )
    conv_with_cma.run_negotiation()

    results["with_cma"] = {
        "result": conv_with_cma.negotiation_result,
        "final_price": conv_with_cma.current_price_offer,
        "turns": conv_with_cma.completed_turns,
        "interventions": len(conv_with_cma.cma_interventions),
        "risk_flags": conv_with_cma.cma_summary['rms_summary']['total_risk_flags']
    }

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print("\nWITHOUT CMA:")
    print(f"  Result: {results['without_cma']['result']}")
    print(f"  Final Price: ${results['without_cma']['final_price']:.2f}")
    print(f"  Turns: {results['without_cma']['turns']}")

    print("\nWITH CMA:")
    print(f"  Result: {results['with_cma']['result']}")
    print(f"  Final Price: ${results['with_cma']['final_price']:.2f}")
    print(f"  Turns: {results['with_cma']['turns']}")
    print(f"  CMA Interventions: {results['with_cma']['interventions']}")
    print(f"  Risk Flags: {results['with_cma']['risk_flags']}")

    # Calculate improvement
    if results['with_cma']['final_price'] and results['without_cma']['final_price']:
        price_diff = results['without_cma']['final_price'] - results['with_cma']['final_price']
        price_diff_pct = (price_diff / results['without_cma']['final_price']) * 100

        print("\nIMPROVEMENT:")
        print(f"  Price Difference: ${price_diff:.2f} ({price_diff_pct:.1f}%)")

        if price_diff > 0:
            print(f"  ✓ CMA helped buyer save ${price_diff:.2f}!")
        elif price_diff < 0:
            print(f"  ✗ CMA resulted in higher price by ${-price_diff:.2f}")
        else:
            print(f"  = Same final price")

    print("="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CMA Example Script")
    parser.add_argument(
        "--mode",
        choices=["simple", "comparison"],
        default="simple",
        help="Run mode: simple (single run) or comparison (with/without CMA)"
    )

    args = parser.parse_args()

    if args.mode == "simple":
        run_cma_example()
    elif args.mode == "comparison":
        run_cma_comparison()


if __name__ == "__main__":
    main()
