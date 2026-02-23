"""Scrape Archidekt for Commander decklists."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.scraping.archidekt import scrape_archidekt


def main():
    parser = argparse.ArgumentParser(description="Scrape Archidekt for Commander decklists")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/archidekt"))
    parser.add_argument("--max-decks", type=int, default=10_000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    decks = scrape_archidekt(output_dir=args.output_dir, max_decks=args.max_decks)

    print(f"\n=== Archidekt Scrape Summary ===")
    print(f"  Total decklists: {len(decks)}")
    print(f"  Saved to:        {args.output_dir / 'decklists.json'}")


if __name__ == "__main__":
    main()
