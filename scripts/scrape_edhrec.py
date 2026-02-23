"""Scrape EDHREC for average Commander decklists."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.scraping.edhrec import scrape_edhrec


def main():
    parser = argparse.ArgumentParser(description="Scrape EDHREC for Commander decklists")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/edhrec"),
        help="Directory to save scraped data (default: data/raw/edhrec)",
    )
    parser.add_argument(
        "--max-commanders",
        type=int,
        default=None,
        help="Limit number of commanders to scrape (default: all)",
    )
    parser.add_argument(
        "--min-decks",
        type=int,
        default=100,
        help="Skip commanders with fewer decks than this (default: 100)",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Path to card catalog directory (default: data/catalog)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    decks = scrape_edhrec(
        output_dir=args.output_dir,
        catalog_dir=args.catalog_dir,
        max_commanders=args.max_commanders,
        min_decks=args.min_decks,
    )

    print(f"\n=== EDHREC Scrape Summary ===")
    print(f"  Total decklists:  {len(decks)}")
    print(f"  Saved to:         {args.output_dir / 'decklists.json'}")

    if decks:
        avg_cards = sum(len(d["cards"]) for d in decks) / len(decks)
        print(f"  Avg cards/deck:   {avg_cards:.1f}")
        print(f"\n  Example commanders:")
        for d in decks[:5]:
            print(
                f"    - {d['commander']} ({len(d['cards'])} cards, {d.get('num_decks', '?')} decks)"
            )


if __name__ == "__main__":
    main()
