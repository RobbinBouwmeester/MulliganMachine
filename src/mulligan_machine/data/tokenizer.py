"""Tokenizer for encoding/decoding Commander decklists as integer sequences."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Special token definitions
SPECIAL_TOKENS = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "SEP": 3,  # separates commander from the 99
    "MASK": 4,  # for optional MLM pretraining
}

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

# Sequence format:
#   [BOS] commander_id [SEP] card_1 card_2 ... card_99 [EOS]
# Total max length = 1 + 1 + 1 + 99 + 1 = 103
MAX_SEQ_LEN = 103
DECK_SIZE = 99  # 99 cards + commander = 100


@dataclass
class DeckTokenizer:
    """
    Maps between card names and token IDs.

    Token IDs 0-4 are reserved for special tokens.
    Card tokens start at offset NUM_SPECIAL_TOKENS (5).
    """

    card_to_id: dict[str, int] = field(default_factory=dict)
    id_to_card: dict[int, str] = field(default_factory=dict)
    vocab_size: int = 0

    @classmethod
    def from_catalog(cls, catalog_dir: Path) -> "DeckTokenizer":
        """Build tokenizer from a previously saved card catalog."""
        with open(catalog_dir / "card_to_id.json", "r", encoding="utf-8") as f:
            raw_card_to_id = json.load(f)

        # Offset all card IDs by NUM_SPECIAL_TOKENS to make room for special tokens
        card_to_id = {name: idx + NUM_SPECIAL_TOKENS for name, idx in raw_card_to_id.items()}
        id_to_card = {idx: name for name, idx in card_to_id.items()}
        vocab_size = NUM_SPECIAL_TOKENS + len(card_to_id)

        logger.info(
            "Tokenizer loaded: %d card tokens + %d special = %d total vocab",
            len(card_to_id),
            NUM_SPECIAL_TOKENS,
            vocab_size,
        )

        return cls(card_to_id=card_to_id, id_to_card=id_to_card, vocab_size=vocab_size)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_deck(
        self,
        commander: str,
        cards: list[str],
        pad: bool = True,
    ) -> list[int]:
        """
        Encode a full decklist into a token sequence.

        Format: [BOS] commander_token [SEP] card_1 ... card_99 [EOS] [PAD...]
        """
        if commander not in self.card_to_id:
            raise KeyError(f"Unknown commander card: {commander!r}")

        tokens = [SPECIAL_TOKENS["BOS"], self.card_to_id[commander], SPECIAL_TOKENS["SEP"]]

        for card_name in cards:
            if card_name in self.card_to_id:
                tokens.append(self.card_to_id[card_name])
            else:
                logger.debug("Skipping unknown card: %s", card_name)

        tokens.append(SPECIAL_TOKENS["EOS"])

        if pad:
            while len(tokens) < MAX_SEQ_LEN:
                tokens.append(SPECIAL_TOKENS["PAD"])
            tokens = tokens[:MAX_SEQ_LEN]

        return tokens

    def encode_partial_deck(
        self,
        commander: str,
        cards: list[str] | None = None,
    ) -> list[int]:
        """
        Encode a partial deck (for inference conditioning).

        Format: [BOS] commander_token [SEP] card_1 ... card_N
        No [EOS], no padding.
        """
        if commander not in self.card_to_id:
            raise KeyError(f"Unknown commander card: {commander!r}")

        tokens = [SPECIAL_TOKENS["BOS"], self.card_to_id[commander], SPECIAL_TOKENS["SEP"]]

        if cards:
            for card_name in cards:
                if card_name in self.card_to_id:
                    tokens.append(self.card_to_id[card_name])
                else:
                    logger.warning("Skipping unknown card in partial deck: %s", card_name)

        return tokens

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_tokens(self, token_ids: list[int]) -> dict[str, Any]:
        """
        Decode a token sequence back into a decklist.

        Returns: {"commander": str, "cards": list[str], "unknown_ids": list[int]}
        """
        commander = None
        cards: list[str] = []
        unknown_ids: list[int] = []
        past_sep = False

        for tid in token_ids:
            # Skip special tokens except SEP
            if tid == SPECIAL_TOKENS["PAD"]:
                continue
            if tid == SPECIAL_TOKENS["BOS"]:
                continue
            if tid == SPECIAL_TOKENS["EOS"]:
                break
            if tid == SPECIAL_TOKENS["MASK"]:
                continue
            if tid == SPECIAL_TOKENS["SEP"]:
                past_sep = True
                continue

            # Resolve card name
            if tid in self.id_to_card:
                card_name = self.id_to_card[tid]
                if not past_sep and commander is None:
                    commander = card_name
                else:
                    cards.append(card_name)
            else:
                unknown_ids.append(tid)

        return {
            "commander": commander,
            "cards": cards,
            "unknown_ids": unknown_ids,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def card_token_id(self, card_name: str) -> int | None:
        """Get the token ID for a card name, or None if not found."""
        return self.card_to_id.get(card_name)

    def token_to_card(self, token_id: int) -> str | None:
        """Get the card name for a token ID, or None if it's a special token."""
        return self.id_to_card.get(token_id)

    def is_special_token(self, token_id: int) -> bool:
        """Check if a token ID is a special token."""
        return token_id < NUM_SPECIAL_TOKENS

    def get_all_card_token_ids(self) -> list[int]:
        """Return all valid card token IDs (excludes special tokens)."""
        return sorted(self.id_to_card.keys())

    def catalog_token_id_to_token(self, catalog_id: int) -> int:
        """Convert a raw catalog card index (0-based) to the tokenizer token ID."""
        return catalog_id + NUM_SPECIAL_TOKENS

    def token_to_catalog_id(self, token_id: int) -> int:
        """Convert a tokenizer token ID back to the raw catalog card index."""
        return token_id - NUM_SPECIAL_TOKENS
