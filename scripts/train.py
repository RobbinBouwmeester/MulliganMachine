"""Train the DeckTransformer model."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mulligan_machine.data.dataset import create_datasets
from mulligan_machine.data.tokenizer import DeckTokenizer
from mulligan_machine.model.config import ModelConfig
from mulligan_machine.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the DeckTransformer model")

    # Data
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/catalog"))
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/edhrec"),
            Path("data/raw/mtggoldfish"),
            Path("data/raw/moxfield"),
            Path("data/raw/archidekt"),
        ],
    )

    # Architecture
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-card-features",
        action="store_true",
        help="Inject card features (oracle embeddings + structured) into model. "
        "Requires card_features.pt in catalog dir (run build_card_features.py first).",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--accumulate-grad",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch-size * this)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross-entropy loss (e.g. 0.1)",
    )
    parser.add_argument(
        "--stochastic-depth",
        type=float,
        default=0.0,
        help="Max stochastic depth (layer drop) rate (e.g. 0.1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (e.g. 0.05 for larger models)",
    )
    parser.add_argument(
        "--no-swiglu",
        action="store_true",
        help="Disable SwiGLU activation (use standard GELU FFN instead)",
    )
    parser.add_argument(
        "--no-rmsnorm",
        action="store_true",
        help="Disable RMSNorm (use standard LayerNorm instead)",
    )

    # Infrastructure
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--patience", type=int, default=0, help="Early stopping patience (0 = disabled)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load tokenizer
    tokenizer = DeckTokenizer.from_catalog(args.catalog_dir)

    # Load card features if requested
    feature_table = None
    n_card_features = 14  # default
    if args.use_card_features:
        from mulligan_machine.data.card_features import load_feature_table

        feature_table, n_card_features = load_feature_table(args.catalog_dir)
        print(f"Card features loaded: {feature_table.shape[1]} features per token")

    # Create datasets
    train_ds, val_ds, _ = create_datasets(
        tokenizer=tokenizer,
        data_dirs=args.data_dirs,
        feature_table=feature_table,
    )

    if len(train_ds) == 0:
        print("ERROR: No training data found. Run scrape_scryfall.py and scrape_edhrec.py first.")
        sys.exit(1)

    # Build config
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad,
        use_card_features=args.use_card_features,
        n_card_features=n_card_features,
        label_smoothing=args.label_smoothing,
        stochastic_depth_rate=args.stochastic_depth,
        weight_decay=args.weight_decay,
        use_swiglu=not args.no_swiglu,
        use_rmsnorm=not args.no_rmsnorm,
    )

    print(f"\n=== Training Configuration ===")
    print(f"  Vocab size:    {config.vocab_size:,}")
    print(f"  Model size:    ~{config.total_params_estimate() / 1e6:.1f}M params")
    print(
        f"  Architecture:  {config.n_layers}L / {config.d_model}D / {config.n_heads}H / {config.d_ff}FF"
    )
    arch_features = []
    if config.use_swiglu:
        arch_features.append("SwiGLU")
    if config.use_rmsnorm:
        arch_features.append("RMSNorm")
    if arch_features:
        print(f"  Modern arch:   {' + '.join(arch_features)}")
    print(
        f"  Card features: {'ON (' + str(n_card_features) + ' dims)' if args.use_card_features else 'OFF'}"
    )
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")
    print(
        f"  Batch size:    {config.batch_size} (effective {config.batch_size * config.accumulate_grad_batches} with {config.accumulate_grad_batches}x accumulation)"
    )
    print(f"  Max epochs:    {config.max_epochs}")
    print(f"  LR:            {config.learning_rate}")
    if config.label_smoothing > 0:
        print(f"  Label smooth:  {config.label_smoothing}")
    if config.stochastic_depth_rate > 0:
        print(f"  Stoch. depth:  {config.stochastic_depth_rate}")
    print()

    # Save config
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    # Train
    model = train(
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        early_stop_patience=args.patience,
    )

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
