"""Preprocess scraped decklists into tokenized training data."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.data.dataset import create_datasets, load_all_decklists
from mulligan_machine.data.tokenizer import DeckTokenizer


def main():
    parser = argparse.ArgumentParser(description="Preprocess decklists into training datasets")
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Path to card catalog directory",
    )
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        default=[Path("data/raw/edhrec"), Path("data/raw/moxfield"), Path("data/raw/archidekt")],
        help="Directories containing decklists.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save processed data stats",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.90,
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05,
    )
    parser.add_argument(
        "--min-cards", type=int, default=50,
        help="Minimum resolved cards per deck",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build tokenizer from catalog
    tokenizer = DeckTokenizer.from_catalog(args.catalog_dir)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

    # Load and split data
    train_ds, val_ds, test_ds = create_datasets(
        tokenizer=tokenizer,
        data_dirs=args.data_dirs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_cards=args.min_cards,
    )

    print(f"\n=== Dataset Summary ===")
    print(f"  Train: {len(train_ds):,} decks")
    print(f"  Val:   {len(val_ds):,} decks")
    print(f"  Test:  {len(test_ds):,} decks")

    # Show a sample
    if len(train_ds) > 0:
        sample = train_ds[0]
        tokens = sample["input_ids"].tolist()
        decoded = tokenizer.decode_tokens(tokens)
        print(f"\n  Sample deck (train[0]):")
        print(f"    Commander: {decoded['commander']}")
        print(f"    Cards ({len(decoded['cards'])}): {', '.join(decoded['cards'][:5])} ...")

    # Save stats
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "vocab_size": tokenizer.vocab_size,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    with open(args.output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to {args.output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
