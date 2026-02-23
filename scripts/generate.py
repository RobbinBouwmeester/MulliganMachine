"""Generate Commander decklists using a trained model."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.inference.generator import load_generator


def format_decklist(deck: dict) -> str:
    """Format a decklist for display."""
    lines = []
    lines.append(f"Commander: {deck['commander']}")
    lines.append(f"{'=' * 60}")

    # Group cards by type if possible
    cards = deck.get("cards", [])
    lines.append(f"\nDeck ({len(cards)} cards):")
    lines.append("-" * 40)

    for i, card in enumerate(cards, 1):
        lines.append(f"  {i:3d}. {card}")

    lines.append(f"\n{'=' * 60}")
    lines.append(f"Lands: {deck.get('n_lands', '?')}")
    lines.append(f"Non-lands: {deck.get('n_nonlands', '?')}")
    lines.append(f"Total: {len(cards) + 1} (including commander)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Commander decklists")
    parser.add_argument(
        "--commander",
        type=str,
        required=True,
        help='Commander card name, e.g. "Atraxa, Praetors\' Voice"',
    )
    parser.add_argument(
        "--cards",
        type=str,
        default=None,
        help="Comma-separated list of cards already in the deck",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/catalog"))
    parser.add_argument("--device", type=str, default="auto")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-lands", type=int, default=36)
    parser.add_argument("--n-decks", type=int, default=1, help="Number of decks to generate")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Save decklist(s) to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse partial card list
    partial_cards = None
    if args.cards:
        partial_cards = [c.strip() for c in args.cards.split(",") if c.strip()]

    # Load generator
    print(f"Loading model from {args.checkpoint}...")
    generator = load_generator(
        checkpoint_path=args.checkpoint,
        catalog_dir=args.catalog_dir,
        device=args.device,
    )

    # Generate
    print(f"Generating {args.n_decks} deck(s) for {args.commander}...")
    if partial_cards:
        print(f"  Starting cards: {', '.join(partial_cards)}")

    decks = generator.generate_multiple(
        commander=args.commander,
        n_decks=args.n_decks,
        partial_cards=partial_cards,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        target_lands=args.target_lands,
    )

    # Display
    for i, deck in enumerate(decks):
        if args.n_decks > 1:
            print(f"\n{'#' * 60}")
            print(f"# Deck Variant {i + 1}/{args.n_decks}")
            print(f"{'#' * 60}")
        print(format_decklist(deck))

    # Save
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(decks if args.n_decks > 1 else decks[0], f, indent=2)
        print(f"\nDeck(s) saved to {args.output}")


if __name__ == "__main__":
    main()
