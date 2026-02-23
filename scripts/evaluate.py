"""Evaluate generated decks against held-out test data."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.data.dataset import load_all_decklists, split_decklists
from mulligan_machine.data.scryfall import load_catalog
from mulligan_machine.data.tokenizer import DeckTokenizer
from mulligan_machine.evaluation.metrics import evaluate_batch, evaluate_deck
from mulligan_machine.inference.generator import load_generator


def main():
    parser = argparse.ArgumentParser(description="Evaluate the deck generator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/catalog"))
    parser.add_argument(
        "--data-dirs", type=Path, nargs="+",
        default=[Path("data/raw/edhrec"), Path("data/raw/moxfield"), Path("data/raw/archidekt")],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-eval", type=int, default=100, help="Max decks to evaluate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load data
    catalog = load_catalog(args.catalog_dir)
    all_decks = load_all_decklists(args.data_dirs)
    _, _, test_decks = split_decklists(all_decks)

    test_decks = test_decks[:args.max_eval]
    print(f"Evaluating on {len(test_decks)} test decks...")

    # Load generator
    generator = load_generator(
        checkpoint_path=args.checkpoint,
        catalog_dir=args.catalog_dir,
        device=args.device,
    )

    # Generate decks for each test commander
    generated = []
    references = []
    for i, ref_deck in enumerate(test_decks):
        commander = ref_deck["commander"]
        print(f"  [{i+1}/{len(test_decks)}] Generating for {commander}...")

        try:
            gen = generator.generate(
                commander=commander,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            generated.append(gen)
            references.append(ref_deck)
        except Exception as e:
            print(f"    Skipping {commander}: {e}")
            continue

    # Evaluate
    print(f"\nEvaluating {len(generated)} generated decks...")
    results = evaluate_batch(generated, references, catalog)

    # Display
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS ({results['n_decks']} decks)")
    print(f"{'=' * 60}")

    for key in sorted(results.keys()):
        if key in ("individual", "avg_mana_curve", "n_decks"):
            continue
        print(f"  {key}: {results[key]:.4f}")

    if "avg_mana_curve" in results:
        print(f"\n  Avg Mana Curve (non-lands):")
        for cmc, count in sorted(results["avg_mana_curve"].items()):
            bar = "█" * int(count)
            label = f"{cmc}+" if int(cmc) == 7 else str(cmc)
            print(f"    CMC {label}: {count:5.1f} {bar}")

    # Save
    if args.output:
        # Remove individual results from disk output to keep it manageable
        save_results = {k: v for k, v in results.items() if k != "individual"}
        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
