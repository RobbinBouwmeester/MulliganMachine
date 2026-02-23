"""Download Scryfall card data and build the card catalog."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.data.scryfall import build_catalog


def main():
    parser = argparse.ArgumentParser(description="Download Scryfall data and build card catalog")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Directory to save the card catalog (default: data/catalog)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Directory for caching Scryfall bulk downloads (default: data/catalog)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    catalog = build_catalog(cache_dir=args.cache_dir, output_dir=args.output_dir)

    print(f"\n=== Card Catalog Summary ===")
    print(f"  Total cards:      {len(catalog['cards']):,}")
    print(f"  Commanders:       {len(catalog['commanders']):,}")
    print(f"  CI groups:        {len(catalog['color_identity_groups'])}")
    print(f"  Saved to:         {args.output_dir}")

    # Show a few example commanders
    print(f"\n  Example commanders:")
    for name in catalog["commanders"][:10]:
        print(f"    - {name}")
    if len(catalog["commanders"]) > 10:
        print(f"    ... and {len(catalog['commanders']) - 10} more")


if __name__ == "__main__":
    main()
