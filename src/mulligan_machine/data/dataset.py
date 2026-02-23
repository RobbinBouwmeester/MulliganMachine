"""PyTorch Dataset for Commander decklists."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mulligan_machine.data.tokenizer import (
    DECK_SIZE,
    MAX_SEQ_LEN,
    SPECIAL_TOKENS,
    DeckTokenizer,
)

logger = logging.getLogger(__name__)


class DeckDataset(Dataset):
    """
    Dataset of tokenized Commander decklists.

    Each item is a token sequence:
        [BOS] commander [SEP] card_1 ... card_99 [EOS] [PAD...]

    The 99-card portion is randomly permuted each time __getitem__ is called,
    teaching the model order-invariance (since decks are unordered sets).
    """

    def __init__(
        self,
        decklists: list[dict[str, Any]],
        tokenizer: DeckTokenizer,
        shuffle_cards: bool = True,
        min_cards: int = 50,
        feature_table: torch.Tensor | None = None,
    ):
        """
        Args:
            decklists: List of {"commander": str, "cards": list[str]} dicts.
            tokenizer: The DeckTokenizer instance.
            shuffle_cards: If True, randomly permute card order each access.
            min_cards: Minimum number of resolved cards to include deck.
            feature_table: Optional (vocab_size, n_features) tensor of card features.
        """
        self.tokenizer = tokenizer
        self.shuffle_cards = shuffle_cards
        self.feature_table = feature_table
        self.samples: list[dict[str, Any]] = []

        n_skipped = 0
        for deck in decklists:
            commander = deck.get("commander", "")
            cards = deck.get("cards", [])

            # Check commander is in vocabulary
            if commander not in tokenizer.card_to_id:
                n_skipped += 1
                continue

            # Resolve card names to token IDs, skip unknown
            resolved = [c for c in cards if c in tokenizer.card_to_id]

            if len(resolved) < min_cards:
                n_skipped += 1
                continue

            # Truncate to 99 (some decks might have more due to data issues)
            resolved = resolved[:DECK_SIZE]

            self.samples.append(
                {
                    "commander": commander,
                    "cards": resolved,
                }
            )

        logger.info(
            "DeckDataset: %d decks loaded, %d skipped (unknown cards or too few)",
            len(self.samples),
            n_skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        cards = list(sample["cards"])

        # Randomly permute card order for order-invariance
        if self.shuffle_cards:
            random.shuffle(cards)

        # Encode to token IDs
        token_ids = self.tokenizer.encode_deck(
            commander=sample["commander"],
            cards=cards,
            pad=True,
        )

        result = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
        }

        # Look up card features per token if feature table is available
        if self.feature_table is not None:
            ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            # Clamp to valid range (special tokens will just get zero features)
            ids_clamped = ids_tensor.clamp(0, self.feature_table.size(0) - 1)
            result["card_features"] = self.feature_table[ids_clamped]  # (T, n_features)

        return result


def load_all_decklists(
    data_dirs: list[Path] | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Load decklists from all available data sources.

    Looks for decklists.json files in each data directory.

    Args:
        data_dirs: List of directories containing decklists.json.
                   Defaults to data/raw/edhrec, data/raw/moxfield, data/raw/archidekt.
        sources: Optional filter to only load from these sources.

    Returns:
        Combined list of decklist dicts.
    """
    if data_dirs is None:
        data_dirs = [
            Path("data/raw/edhrec"),
            Path("data/raw/moxfield"),
            Path("data/raw/archidekt"),
        ]

    all_decks: list[dict[str, Any]] = []

    for data_dir in data_dirs:
        decks_file = data_dir / "decklists.json"
        if not decks_file.exists():
            logger.debug("No decklists at %s, skipping", decks_file)
            continue

        source_name = data_dir.name
        if sources and source_name not in sources:
            continue

        with open(decks_file, "r", encoding="utf-8") as f:
            decks = json.load(f)

        # Tag each deck with its source if not already
        for d in decks:
            d.setdefault("source", source_name)

        logger.info("Loaded %d decklists from %s", len(decks), data_dir)
        all_decks.extend(decks)

    logger.info("Total decklists loaded: %d", len(all_decks))
    return all_decks


def split_decklists(
    decklists: list[dict[str, Any]],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split decklists into train/val/test sets.

    Uses deterministic shuffle for reproducibility.
    """
    rng = random.Random(seed)
    shuffled = list(decklists)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    logger.info("Split: %d train, %d val, %d test", len(train), len(val), len(test))
    return train, val, test


def create_datasets(
    tokenizer: DeckTokenizer,
    data_dirs: list[Path] | None = None,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
    min_cards: int = 50,
    feature_table: torch.Tensor | None = None,
) -> tuple[DeckDataset, DeckDataset, DeckDataset]:
    """
    Convenience function: load all data, split, and create Dataset objects.
    """
    all_decks = load_all_decklists(data_dirs)
    train_decks, val_decks, test_decks = split_decklists(all_decks, train_ratio, val_ratio, seed)

    train_ds = DeckDataset(
        train_decks, tokenizer, shuffle_cards=True, min_cards=min_cards, feature_table=feature_table
    )
    val_ds = DeckDataset(
        val_decks, tokenizer, shuffle_cards=False, min_cards=min_cards, feature_table=feature_table
    )
    test_ds = DeckDataset(
        test_decks, tokenizer, shuffle_cards=False, min_cards=min_cards, feature_table=feature_table
    )

    return train_ds, val_ds, test_ds
