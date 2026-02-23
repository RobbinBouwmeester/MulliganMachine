"""
Card feature extraction for the DeckTransformer.

Produces a feature table that maps each token ID to a dense feature vector,
combining:
  1. **Oracle text embeddings** — from a sentence-transformer model, capturing
     the full semantic meaning of each card's abilities.
  2. **Structured features** — CMC, colors, card types, keywords, mechanics
     mined from oracle text, power/toughness, rarity.

The feature table is pre-computed once and saved to disk, then loaded at
train/inference time for fast lookup.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Mechanic patterns mined from oracle text ──────────────────────────────────
# Each tuple is (feature_name, regex_pattern) applied to oracle_text.
# These capture the *functional* aspects of cards.
MECHANIC_PATTERNS: list[tuple[str, str]] = [
    # Card advantage
    ("draws_cards", r"\bdraw[s]?\b.*\bcard"),
    ("cantrip", r"\bdraw a card\b"),
    ("card_selection", r"\b(scry|surveil|look at the top)\b"),
    # Removal
    ("destroys", r"\bdestroy[s]?\b.*\b(creature|artifact|enchantment|permanent|planeswalker)"),
    (
        "exiles",
        r"\bexile[s]?\b.*\b(creature|artifact|enchantment|permanent|planeswalker|graveyard)",
    ),
    (
        "board_wipe",
        r"\bdestroy all\b|\bexile all\b|\ball creatures get -|\bdeal \d+ damage to each",
    ),
    ("bounces", r"\breturn[s]?\b.*\bto (its|their) owner"),
    ("damage_deal", r"\bdeal[s]?\b.*\bdamage\b"),
    # Counterspells
    ("counters_spells", r"\bcounter target\b|\bcounter that\b|\bcounter it\b"),
    # Graveyard
    ("reanimates", r"\breturn[s]?\b.*\bfrom.{0,20}\bgraveyard\b.*\bto the battlefield\b"),
    ("graveyard_matters", r"\bgraveyard\b"),
    ("mills", r"\bmill\b"),
    ("self_sacrifice", r"\bsacrifice\b"),
    # Mana / ramp
    ("ramp", r"\bsearch.{0,30}\bland.{0,20}\b(onto|into play|to the battlefield)\b"),
    ("mana_dork", r"\badd\b.*\{[WUBRGC]"),
    ("cost_reduction", r"\bcost[s]?\b.*\bless\b|\breduce the cost\b"),
    # Tokens
    ("creates_tokens", r"\bcreate[s]?\b.*\btoken"),
    # +1/+1 counters
    ("plus_counters", r"\+1/\+1 counter"),
    ("minus_counters", r"-1/-1 counter"),
    # Life
    ("gains_life", r"\bgain[s]?\b.*\blife\b"),
    ("drains_life", r"\blose[s]?\b.*\blife\b|\bloses? life\b"),
    # Tutors
    ("tutors", r"\bsearch your library\b"),
    # Protection
    ("gives_protection", r"\b(hexproof|shroud|indestructible|protection from)\b"),
    ("gives_evasion", r"\b(flying|trample|menace|unblockable|shadow)\b"),
    # ETB / Triggers
    ("etb_trigger", r"\benters\b.{0,30}\b(battlefield|trigger)\b|\betb\b"),
    ("death_trigger", r"\bwhen.{0,30}\bdies\b|\bwhen.{0,30}\bput into.{0,15}\bgraveyard\b"),
    ("attack_trigger", r"\bwhenever.{0,30}\battacks\b"),
    # Equipment/Auras
    ("equip_or_aura", r"\bequip\b|\benchant\b"),
    # Proliferate and counters matter
    ("proliferate", r"\bproliferate\b"),
    # Copy effects
    ("copies", r"\bcopy\b.*\b(spell|creature|permanent)\b|\bcreate a token.{0,20}copy\b"),
    # Untap/tap effects
    ("untaps", r"\buntap\b"),
    # Discard
    ("forces_discard", r"\bdiscard[s]?\b"),
    # Extra turns
    ("extra_turn", r"\bextra turn\b|\badditional turn\b"),
    # Flash
    ("has_flash", r"\bflash\b"),
]

# Top MTG keywords to include as binary features
TOP_KEYWORDS = [
    "Flying",
    "Trample",
    "Haste",
    "First strike",
    "Double strike",
    "Deathtouch",
    "Lifelink",
    "Vigilance",
    "Reach",
    "Menace",
    "Flash",
    "Hexproof",
    "Indestructible",
    "Ward",
    "Defender",
    "Landfall",
    "Cascade",
    "Convoke",
    "Delve",
    "Flashback",
    "Kicker",
    "Madness",
    "Morph",
    "Cycling",
    "Equip",
    "Proliferate",
    "Explore",
    "Scry",
    "Surveil",
    "Mill",
]

RARITY_MAP = {
    "common": 0.0,
    "uncommon": 0.25,
    "rare": 0.5,
    "mythic": 0.75,
    "special": 1.0,
    "bonus": 0.5,
}

# Number of structured features
N_STRUCTURED = (
    1  # cmc (normalized)
    + 5  # WUBRG color identity
    + 8  # type categories (Creature, Instant, Sorcery, Artifact, Enchantment, Planeswalker, Land, Battle)
    + 2  # power, toughness (normalized)
    + 1  # rarity
    + 1  # edhrec_rank (normalized)
    + 1  # is_legendary
    + len(MECHANIC_PATTERNS)  # mechanic flags
    + len(TOP_KEYWORDS)  # keyword flags
)


def _build_structured_features(cards: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    """
    Build structured feature vectors for each card.

    Returns: {token_id: np.ndarray of shape (N_STRUCTURED,)}
    """
    # Get max EDHREC rank for normalization
    max_rank = max((c.get("edhrec_rank") or 99999) for c in cards)

    features: dict[int, np.ndarray] = {}

    for card in cards:
        token_id = card["token_id"]
        vec = []

        # CMC (normalized to ~[0, 1])
        vec.append(min(card.get("cmc", 0.0), 16.0) / 16.0)

        # Color identity (5 binary)
        ci = set(card.get("color_identity", []))
        for color in "WUBRG":
            vec.append(1.0 if color in ci else 0.0)

        # Type categories (8 binary)
        tc = set(card.get("type_categories", []))
        for type_cat in [
            "Creature",
            "Instant",
            "Sorcery",
            "Artifact",
            "Enchantment",
            "Planeswalker",
            "Land",
            "Battle",
        ]:
            vec.append(1.0 if type_cat in tc else 0.0)

        # Power / toughness (normalized; 0 for non-creatures)
        power_str = card.get("power", "")
        toughness_str = card.get("toughness", "")
        try:
            power = float(power_str) if power_str and power_str != "*" else 0.0
        except (ValueError, TypeError):
            power = 0.0
        try:
            toughness = float(toughness_str) if toughness_str and toughness_str != "*" else 0.0
        except (ValueError, TypeError):
            toughness = 0.0
        vec.append(min(power, 20.0) / 20.0)
        vec.append(min(toughness, 20.0) / 20.0)

        # Rarity
        rarity = card.get("rarity", "common")
        vec.append(RARITY_MAP.get(rarity, 0.0))

        # EDHREC rank (inverted & normalized — lower rank = more popular = higher value)
        rank = card.get("edhrec_rank") or max_rank
        vec.append(1.0 - min(rank, max_rank) / max_rank)

        # Is legendary
        type_line = card.get("type_line", "")
        vec.append(1.0 if "Legendary" in type_line else 0.0)

        # Mechanic patterns from oracle text
        oracle = (card.get("oracle_text") or "").lower()
        for _, pattern in MECHANIC_PATTERNS:
            vec.append(1.0 if re.search(pattern, oracle) else 0.0)

        # Top keywords
        keywords = set(card.get("keywords", []))
        for kw in TOP_KEYWORDS:
            vec.append(1.0 if kw in keywords else 0.0)

        features[token_id] = np.array(vec, dtype=np.float32)

    return features


def _build_oracle_embeddings(
    cards: list[dict[str, Any]],
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 512,
) -> dict[int, np.ndarray]:
    """
    Encode each card's oracle text (+ type line) using a sentence transformer.

    Returns: {token_id: np.ndarray of shape (embedding_dim,)}
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence transformer: %s", model_name)
    st_model = SentenceTransformer(model_name)

    # Build text inputs: combine type_line + oracle_text for full context
    texts = []
    token_ids = []
    for card in cards:
        type_line = card.get("type_line", "")
        oracle = card.get("oracle_text", "") or ""
        mana = card.get("mana_cost", "") or ""
        # Compose a rich text representation
        text = f"{card['name']}. {mana} {type_line}. {oracle}".strip()
        texts.append(text)
        token_ids.append(card["token_id"])

    logger.info("Encoding %d card texts...", len(texts))
    embeddings = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    result = {}
    for tid, emb in zip(token_ids, embeddings):
        result[tid] = emb.astype(np.float32)

    emb_dim = embeddings.shape[1] if len(embeddings) > 0 else 384
    logger.info("Oracle embeddings: %d cards × %d dims", len(result), emb_dim)
    return result


def build_feature_table(
    catalog_dir: Path = Path("data/catalog"),
    output_dir: Path | None = None,
    use_oracle_embeddings: bool = True,
    oracle_model: str = "all-mpnet-base-v2",
) -> tuple[torch.Tensor, int]:
    """
    Build a complete feature table for all cards and save to disk.

    The table is a (vocab_size, n_features) float tensor where:
      - Rows 0..4 are zero (special tokens: PAD, BOS, EOS, SEP, MASK)
      - Rows 5+ correspond to card token IDs

    Returns:
        (feature_table, n_features) where feature_table is (vocab_size, n_features)
    """
    from mulligan_machine.data.tokenizer import NUM_SPECIAL_TOKENS

    if output_dir is None:
        output_dir = catalog_dir

    # Load card catalog
    cards_path = catalog_dir / "cards.json"
    with open(cards_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    logger.info("Building card features for %d cards...", len(cards))

    # Build structured features
    structured = _build_structured_features(cards)

    # Build oracle embeddings (optional)
    if use_oracle_embeddings:
        oracle_embs = _build_oracle_embeddings(cards, model_name=oracle_model)
        oracle_dim = next(iter(oracle_embs.values())).shape[0] if oracle_embs else 384
    else:
        oracle_embs = None
        oracle_dim = 0

    # Combine into single table
    n_structured = N_STRUCTURED
    n_features = n_structured + oracle_dim

    # Determine vocab size
    max_token_id = max(c["token_id"] for c in cards)
    vocab_size = max_token_id + NUM_SPECIAL_TOKENS + 1  # +1 for 0-indexing

    logger.info(
        "Feature table: %d vocab × %d features (%d structured + %d oracle)",
        vocab_size,
        n_features,
        n_structured,
        oracle_dim,
    )

    table = np.zeros((vocab_size, n_features), dtype=np.float32)

    for card in cards:
        row_idx = card["token_id"] + NUM_SPECIAL_TOKENS
        # Structured features
        if card["token_id"] in structured:
            table[row_idx, :n_structured] = structured[card["token_id"]]
        # Oracle embeddings
        if oracle_embs and card["token_id"] in oracle_embs:
            table[row_idx, n_structured:] = oracle_embs[card["token_id"]]

    feature_table = torch.from_numpy(table)

    # Save
    save_path = output_dir / "card_features.pt"
    torch.save({"features": feature_table, "n_features": n_features}, save_path)
    logger.info("Saved feature table to %s", save_path)

    # Also save metadata
    meta_path = output_dir / "card_features_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_features": n_features,
                "n_structured": n_structured,
                "n_oracle": oracle_dim,
                "oracle_model": oracle_model if use_oracle_embeddings else None,
                "n_mechanic_patterns": len(MECHANIC_PATTERNS),
                "n_keywords": len(TOP_KEYWORDS),
                "vocab_size": vocab_size,
            },
            f,
            indent=2,
        )

    return feature_table, n_features


def load_feature_table(
    catalog_dir: Path = Path("data/catalog"),
) -> tuple[torch.Tensor, int]:
    """
    Load a pre-computed feature table from disk.

    Returns:
        (feature_table, n_features)
    """
    path = catalog_dir / "card_features.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found at {path}. "
            "Run `python scripts/build_card_features.py` first."
        )
    data = torch.load(path, weights_only=True)
    return data["features"], data["n_features"]
