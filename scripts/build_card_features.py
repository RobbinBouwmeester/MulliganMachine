"""Build the card feature table (oracle embeddings + structured features)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.data.card_features import build_feature_table


def main():
    parser = argparse.ArgumentParser(description="Build card feature table")
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/catalog"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip oracle text embeddings (faster, structured features only)",
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for oracle text",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output = args.output_dir or args.catalog_dir
    table, n_features = build_feature_table(
        catalog_dir=args.catalog_dir,
        output_dir=output,
        use_oracle_embeddings=not args.no_oracle,
        oracle_model=args.oracle_model,
    )

    print(f"\nFeature table built: {table.shape[0]:,} vocab × {n_features} features")
    print(f"Saved to: {output / 'card_features.pt'}")


if __name__ == "__main__":
    main()
