"""Constrained deck generation from the trained transformer model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mulligan_machine.data.scryfall import load_catalog
from mulligan_machine.data.tokenizer import (
    DECK_SIZE,
    MAX_SEQ_LEN,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    DeckTokenizer,
)
from mulligan_machine.model.config import ModelConfig
from mulligan_machine.model.transformer import DeckTransformer

logger = logging.getLogger(__name__)


class DeckGenerator:
    """
    Generate Commander decklists using constrained autoregressive decoding.

    Enforces:
      - Color identity: only cards within the commander's color identity
      - Singleton: no duplicate cards (except basic lands)
      - Card count: exactly 99 cards + commander = 100
      - Land awareness: nudge toward reasonable land counts
    """

    def __init__(
        self,
        model: DeckTransformer,
        tokenizer: DeckTokenizer,
        catalog: dict[str, Any],
        device: torch.device | str = "cpu",
        feature_table: torch.Tensor | None = None,
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.feature_table = feature_table.to(self.device) if feature_table is not None else None

        # Precompute card metadata for constraint enforcement
        self._build_card_metadata()

    def _build_card_metadata(self):
        """Build lookup structures for card properties."""
        cards = self.catalog["cards"]
        card_to_id = self.catalog["card_to_id"]
        V = self.tokenizer.vocab_size

        # Map token IDs to card properties
        self.token_is_land: dict[int, bool] = {}
        self.token_is_basic_land: dict[int, bool] = {}
        self.token_color_identity: dict[int, set[str]] = {}
        self.token_cmc: dict[int, float] = {}

        # Also build a boolean tensor mask for lands (for fast vectorized nudging)
        land_mask = torch.zeros(V, dtype=torch.bool)
        basic_land_mask = torch.zeros(V, dtype=torch.bool)

        for card in cards:
            catalog_id = card["token_id"]
            token_id = catalog_id + NUM_SPECIAL_TOKENS
            self.token_is_land[token_id] = card["is_land"]
            self.token_is_basic_land[token_id] = card["is_basic_land"]
            self.token_color_identity[token_id] = set(card["color_identity"])
            self.token_cmc[token_id] = card["cmc"]
            if card["is_land"]:
                land_mask[token_id] = True
            if card["is_basic_land"]:
                basic_land_mask[token_id] = True

        self.land_mask = land_mask.to(self.device)
        self.basic_land_mask = basic_land_mask.to(self.device)

        # Precompute color identity masks (token-level)
        # For each commander color identity key, create a boolean mask over the full vocab
        self.ci_masks: dict[str, torch.Tensor] = {}
        ci_groups = self.catalog["color_identity_groups"]
        for ci_key, catalog_ids in ci_groups.items():
            mask = torch.zeros(V, dtype=torch.bool)
            for cid in catalog_ids:
                mask[cid + NUM_SPECIAL_TOKENS] = True
            self.ci_masks[ci_key] = mask.to(self.device)

    def _get_commander_ci_key(self, commander_name: str) -> str:
        """Get the color identity key for a commander."""
        cards = self.catalog["cards"]
        card_to_id = self.catalog["card_to_id"]
        idx = card_to_id.get(commander_name)
        if idx is None:
            raise KeyError(f"Unknown commander: {commander_name!r}")
        return cards[idx]["color_identity_key"]

    @torch.no_grad()
    def generate(
        self,
        commander: str,
        partial_cards: list[str] | None = None,
        temperature: float = 0.8,
        top_k: int = 100,
        top_p: float = 0.95,
        target_lands: int = 36,
        land_nudge_start: int = 80,
        land_nudge_strength: float = 2.0,
    ) -> dict[str, Any]:
        """
        Generate a complete Commander decklist.

        Args:
            commander: Commander card name.
            partial_cards: Optional list of cards already in the deck.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Keep top-k tokens before sampling.
            top_p: Nucleus sampling threshold.
            target_lands: Target number of land cards in the deck.
            land_nudge_start: Position after which to start nudging land counts.
            land_nudge_strength: How strongly to nudge land selection.

        Returns:
            Dict with "commander", "cards", metadata.
        """
        ci_key = self._get_commander_ci_key(commander)
        ci_mask = self.ci_masks.get(ci_key)
        if ci_mask is None:
            logger.warning("No CI mask for %s, using unmasked generation", ci_key)
            ci_mask = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool, device=self.device)

        # Build initial sequence
        tokens = self.tokenizer.encode_partial_deck(commander, partial_cards)
        selected_tokens: set[int] = set(tokens)  # track what's been picked

        # Track basic land copy counts (cap at max_basic_copies each)
        max_basic_copies = 10
        basic_land_counts: dict[int, int] = {}
        for t in tokens:
            if self.token_is_basic_land.get(t, False):
                basic_land_counts[t] = basic_land_counts.get(t, 0) + 1

        # Count initial state
        n_cards = len(tokens) - 3  # subtract BOS, commander, SEP
        n_lands = sum(1 for t in tokens[3:] if self.token_is_land.get(t, False))  # cards after SEP

        # Generate remaining cards
        cards_needed = DECK_SIZE - n_cards

        for step in range(cards_needed):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Truncate if needed (shouldn't happen with proper MAX_SEQ_LEN)
            if input_tensor.size(1) > MAX_SEQ_LEN - 1:
                input_tensor = input_tensor[:, -(MAX_SEQ_LEN - 1) :]

            # Build card features for current sequence if available
            card_feats = None
            if self.feature_table is not None:
                ids_clamped = input_tensor.clamp(0, self.feature_table.size(0) - 1)
                card_feats = self.feature_table[ids_clamped]  # (1, T, n_features)

            logits = self.model.generate_next_token_logits(input_tensor, card_feats)  # (1, V)
            logits = logits.squeeze(0)  # (V,)

            # === Constraint masking ===

            # 1. Block special tokens
            for special_id in SPECIAL_TOKENS.values():
                logits[special_id] = float("-inf")

            # 2. Color identity constraint
            logits[~ci_mask] = float("-inf")

            # 3. Singleton constraint (except basic lands, which get a copy cap)
            for t in selected_tokens:
                if t >= NUM_SPECIAL_TOKENS and not self.token_is_basic_land.get(t, False):
                    logits[t] = float("-inf")

            # 3b. Cap basic land copies
            for t, count in basic_land_counts.items():
                if count >= max_basic_copies:
                    logits[t] = float("-inf")

            # 4. Land nudging (vectorized — toward end of generation)
            current_pos = n_cards + step
            if current_pos >= land_nudge_start:
                cards_remaining = DECK_SIZE - current_pos
                lands_needed = target_lands - n_lands

                if cards_remaining > 0:
                    land_ratio_needed = lands_needed / cards_remaining

                    # Use precomputed land_mask tensor for fast nudging
                    valid = logits > float("-inf")
                    land_and_valid = self.land_mask & valid
                    if land_ratio_needed > 0.5:
                        logits[land_and_valid] += land_nudge_strength
                    elif land_ratio_needed < 0.2:
                        logits[land_and_valid] -= land_nudge_strength

            # === Sampling ===

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-K filtering
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(logits, min(top_k, logits.size(0)))
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(0, top_k_idx, top_k_vals)
                logits = mask

            # Top-P (nucleus) filtering
            if 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back
                logits.scatter_(0, sorted_idx, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

            tokens.append(next_token)
            selected_tokens.add(next_token)

            if self.token_is_land.get(next_token, False):
                n_lands += 1
            if self.token_is_basic_land.get(next_token, False):
                basic_land_counts[next_token] = basic_land_counts.get(next_token, 0) + 1

        # Decode
        result = self.tokenizer.decode_tokens(tokens)

        # Add metadata
        result["n_lands"] = n_lands
        result["n_nonlands"] = len(result["cards"]) - n_lands
        result["temperature"] = temperature
        result["top_k"] = top_k
        result["top_p"] = top_p

        return result

    def generate_multiple(
        self,
        commander: str,
        n_decks: int = 5,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Generate multiple deck variants for the same commander."""
        decks = []
        for i in range(n_decks):
            logger.info("Generating deck %d/%d for %s", i + 1, n_decks, commander)
            deck = self.generate(commander, **kwargs)
            decks.append(deck)
        return decks


def load_generator(
    checkpoint_path: str | Path,
    catalog_dir: Path = Path("data/catalog"),
    device: str = "auto",
) -> DeckGenerator:
    """
    Load a trained model and create a DeckGenerator.

    Args:
        checkpoint_path: Path to a Lightning checkpoint (.ckpt) file.
        catalog_dir: Path to the card catalog directory.
        device: 'auto', 'cpu', 'cuda', etc.
    """
    from mulligan_machine.training.trainer import DeckTransformerLightning

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load catalog and tokenizer
    catalog = load_catalog(catalog_dir)
    tokenizer = DeckTokenizer.from_catalog(catalog_dir)

    # Load model from checkpoint
    lit_model = DeckTransformerLightning.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )
    model = lit_model.model

    # Load card features if available and model uses them
    feature_table = None
    if model.config.use_card_features:
        try:
            from mulligan_machine.data.card_features import load_feature_table

            feature_table, _ = load_feature_table(catalog_dir)
            logger.info("Loaded card feature table for inference")
        except FileNotFoundError:
            logger.warning(
                "Model uses card features but feature table not found — generating without features"
            )

    return DeckGenerator(
        model=model,
        tokenizer=tokenizer,
        catalog=catalog,
        device=device,
        feature_table=feature_table,
    )
