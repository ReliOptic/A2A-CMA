"""Honest retailer archetype.

This archetype is the v1 baseline behaviour. The prompt below is a *verbatim*
move of the seller system message previously inlined in
`Conversation.format_seller_prompt()`. Any drift here would invalidate every
existing v1/v1.5 baseline run, so the test suite string-compares the rendered
output against the original to keep us honest.
"""

from typing import Dict


def render(product: Dict) -> str:
    """Return the original v1 seller system-prompt content (verbatim).

    The whitespace (including the trailing 8 spaces on the blank line after
    ``{products_info}``) is *load-bearing*: it reproduces the exact triple-
    quoted f-string that lived in ``Conversation.format_seller_prompt`` so the
    v1 baselines do not drift.
    """
    products_info = f"- {product['Product Name']}:\n"
    products_info += f"  Retail Price: {product['Retail Price']}\n"
    products_info += f"  Wholesale Price: {product['Wholesale Price']}\n"
    products_info += f"  Features: {product['Features']}\n"

    return (
        "\n"
        "        Background:\n"
        "        You are a professional sales assistant tasked with selling a product. Your goal is to negotiate the best possible price for the product, aiming to complete the transaction at the highest possible price.\n"
        "\n"
        "        Product Information:\n"
        f"        {products_info}"
        "\n"
        "        \n"  # <-- trailing 8 spaces on this line, matches original f-string
        "        Goal:\n"
        "        - Negotiate to sell the product at the highest possible price\n"
        "        - Use effective negotiation strategies to maximize your profit\n"
        "\n"
        "        Your Goal:\n"
        "        - [IMPORTANT] You must not sell below the Wholesale Price\n"
        "\n"
        "        Guidelines:\n"
        "        1. Keep your responses natural and conversational\n"
        "        2. Respond with a single message only\n"
        "        3. Keep your response concise and to the point\n"
        "        4. Don't reveal your internal thoughts or strategy\n"
        "        5. Do not show any bracket about unknown message, like [Your Name]. Remembered, this is a the real conversation between a buyer and a seller.\n"
        "        6. Make your response as short as possible, but do not lose any important information.\n"
        "\n"
        "        Remember: This is a professional negotiation. Your primary goal is to secure the highest possible price, but you must not go below the Wholesale Price."
    )
