"""Scrape MTGGoldfish for Commander decklists."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.scraping.mtggoldfish import scrape_mtggoldfish


def main():
    parser = argparse.ArgumentParser(description="Scrape MTGGoldfish for Commander decklists")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/mtggoldfish"),
        help="Directory to save scraped data (default: data/raw/mtggoldfish)",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Path to card catalog directory (default: data/catalog)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Max search pages to crawl, 30 decks/page (default: 500)",
    )
    parser.add_argument(
        "--max-decks",
        type=int,
        default=None,
        help="Max total decks to collect (default: unlimited)",
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

    decks = scrape_mtggoldfish(
        output_dir=args.output_dir,
        catalog_dir=args.catalog_dir,
        max_pages=args.max_pages,
        max_decks=args.max_decks,
    )

    print(f"\n=== MTGGoldfish Scrape Summary ===")
    print(f"  Total decklists:  {len(decks)}")
    print(f"  Saved to:         {args.output_dir / 'decklists.json'}")

    if decks:
        avg_cards = sum(len(d["cards"]) for d in decks) / len(decks)
        print(f"  Avg cards/deck:   {avg_cards:.1f}")
        commanders = set(d["commander"] for d in decks)
        print(f"  Unique commanders: {len(commanders)}")
        print(f"\n  Example commanders:")
        for cmd in sorted(commanders)[:5]:
            count = sum(1 for d in decks if d["commander"] == cmd)
            print(f"    - {cmd} ({count} decks)")


if __name__ == "__main__":
    main()
