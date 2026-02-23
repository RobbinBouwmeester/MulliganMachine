"""Download and process Scryfall bulk data into a card catalog for Commander."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# All 32 possible WUBRG subsets (including colorless = frozenset())
COLOR_IDENTITIES = []
WUBRG = ["W", "U", "B", "R", "G"]
for mask in range(32):
    combo = frozenset(WUBRG[i] for i in range(5) if mask & (1 << i))
    COLOR_IDENTITIES.append(combo)

DEFAULT_CATALOG_DIR = Path("data/catalog")
SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"


def _download_bulk_json(cache_dir: Path) -> list[dict[str, Any]]:
    """Download the Scryfall 'oracle_cards' bulk dataset (one entry per unique card)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    bulk_file = cache_dir / "oracle_cards.json"

    if bulk_file.exists():
        logger.info("Loading cached Scryfall bulk data from %s", bulk_file)
        with open(bulk_file, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching Scryfall bulk-data manifest...")
    resp = requests.get(SCRYFALL_BULK_URL, timeout=30)
    resp.raise_for_status()
    manifest = resp.json()

    oracle_entry = next((e for e in manifest["data"] if e["type"] == "oracle_cards"), None)
    if oracle_entry is None:
        raise RuntimeError("Could not find 'oracle_cards' in Scryfall bulk-data manifest")

    download_url = oracle_entry["download_uri"]
    logger.info("Downloading oracle_cards (~80 MB) from %s ...", download_url)

    resp = requests.get(download_url, timeout=300, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunks: list[bytes] = []
    with tqdm(total=total, unit="B", unit_scale=True, desc="Scryfall download") as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            chunks.append(chunk)
            pbar.update(len(chunk))

    raw = b"".join(chunks)
    cards = json.loads(raw)

    logger.info("Caching %d cards to %s", len(cards), bulk_file)
    with open(bulk_file, "w", encoding="utf-8") as f:
        json.dump(cards, f)

    return cards


def _is_commander_legal(card: dict[str, Any]) -> bool:
    """Check if a card is legal in Commander format."""
    legalities = card.get("legalities", {})
    return legalities.get("commander") in ("legal", "restricted")


def _is_potential_commander(card: dict[str, Any]) -> bool:
    """Check if a card can be used as a commander."""
    type_line = card.get("type_line", "")
    oracle_text = card.get("oracle_text", "")

    # Legendary creatures can be commanders
    if "Legendary" in type_line and "Creature" in type_line:
        return True

    # Some planeswalkers explicitly say they can be your commander
    if "can be your commander" in oracle_text.lower():
        return True

    # Background enchantments function with "choose a Background"
    if "Background" in type_line and "Legendary" in type_line and "Enchantment" in type_line:
        return True

    return False


def _extract_card_info(card: dict[str, Any]) -> dict[str, Any]:
    """Extract the relevant fields from a Scryfall card object."""
    # Handle double-faced cards: use front face for most info
    if "card_faces" in card and len(card["card_faces"]) > 0:
        front = card["card_faces"][0]
        oracle_text = front.get("oracle_text", "")
        mana_cost = front.get("mana_cost", "")
        type_line = front.get("type_line", card.get("type_line", ""))
        power = front.get("power")
        toughness = front.get("toughness")
    else:
        oracle_text = card.get("oracle_text", "")
        mana_cost = card.get("mana_cost", "")
        type_line = card.get("type_line", "")
        power = card.get("power")
        toughness = card.get("toughness")

    # Parse keywords
    keywords = card.get("keywords", [])

    # Parse type categories
    type_categories = []
    for t in [
        "Creature",
        "Instant",
        "Sorcery",
        "Artifact",
        "Enchantment",
        "Planeswalker",
        "Land",
        "Battle",
    ]:
        if t in type_line:
            type_categories.append(t)

    # Color identity as sorted list
    color_identity = sorted(card.get("color_identity", []))

    return {
        "name": card["name"],
        "scryfall_id": card.get("id", ""),
        "color_identity": color_identity,
        "color_identity_key": "".join(color_identity) if color_identity else "C",
        "cmc": card.get("cmc", 0.0),
        "mana_cost": mana_cost,
        "type_line": type_line,
        "type_categories": type_categories,
        "oracle_text": oracle_text,
        "keywords": keywords,
        "power": power,
        "toughness": toughness,
        "is_land": "Land" in type_line,
        "is_basic_land": (
            type_line.startswith("Basic Land")
            or card["name"]
            in {
                "Plains",
                "Island",
                "Swamp",
                "Mountain",
                "Forest",
                "Wastes",  # technically not "Basic Land" but often treated similarly
                "Snow-Covered Plains",
                "Snow-Covered Island",
                "Snow-Covered Swamp",
                "Snow-Covered Mountain",
                "Snow-Covered Forest",
            }
        ),
        "can_be_commander": _is_potential_commander(card),
        "rarity": card.get("rarity", "common"),
        "set_code": card.get("set", ""),
        "edhrec_rank": card.get("edhrec_rank"),
    }


def _color_identity_is_subset(card_ci: list[str], commander_ci: list[str]) -> bool:
    """Check if a card's color identity is a subset of a commander's color identity."""
    return set(card_ci).issubset(set(commander_ci))


def build_catalog(
    cache_dir: Path = DEFAULT_CATALOG_DIR,
    output_dir: Path = DEFAULT_CATALOG_DIR,
) -> dict[str, Any]:
    """
    Build the full card catalog from Scryfall data.

    Returns a dict with:
      - cards: list of card info dicts
      - card_to_id: dict mapping card name -> token ID
      - id_to_card: dict mapping token ID (str) -> card name
      - commanders: list of card names that can be commanders
      - color_identity_groups: dict mapping color-identity-key -> list of token IDs
    """
    raw_cards = _download_bulk_json(cache_dir)

    # Filter to Commander-legal cards
    commander_legal = [c for c in raw_cards if _is_commander_legal(c)]
    logger.info(
        "Filtered to %d Commander-legal cards (from %d total)",
        len(commander_legal),
        len(raw_cards),
    )

    # Extract info and deduplicate by name (Scryfall oracle cards should already be unique)
    seen_names: set[str] = set()
    cards: list[dict[str, Any]] = []
    for raw_card in commander_legal:
        name = raw_card["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        cards.append(_extract_card_info(raw_card))

    # Sort by name for deterministic ordering
    cards.sort(key=lambda c: c["name"])

    logger.info("Catalog contains %d unique Commander-legal cards", len(cards))

    # Build name <-> token ID mappings
    card_to_id: dict[str, int] = {}
    id_to_card: dict[int, str] = {}
    for idx, card in enumerate(cards):
        card["token_id"] = idx
        card_to_id[card["name"]] = idx
        id_to_card[idx] = card["name"]

    # Identify commanders
    commanders = [c["name"] for c in cards if c["can_be_commander"]]
    logger.info("Found %d potential commanders", len(commanders))

    # Build color identity groups: for each possible CI (WUBRG subset),
    # list all card IDs that are legal under that CI
    color_identity_groups: dict[str, list[int]] = {}
    all_ci_keys = set()
    for c in cards:
        all_ci_keys.add(c["color_identity_key"])

    # For each commander color identity, precompute legal card IDs
    # We key by the sorted color identity string, e.g. "WUBR", "G", "C" (colorless)
    # For each card, it's legal under a commander CI if card CI ⊆ commander CI
    # We'll store groups keyed by commander CI
    unique_commander_cis = set()
    for c in cards:
        if c["can_be_commander"]:
            unique_commander_cis.add(tuple(c["color_identity"]))

    # Also add all 32 possible CI combinations for completeness
    for mask in range(32):
        combo = tuple(sorted(WUBRG[i] for i in range(5) if mask & (1 << i)))
        unique_commander_cis.add(combo)
    unique_commander_cis.add(())  # colorless

    for ci in unique_commander_cis:
        ci_set = set(ci)
        ci_key = "".join(sorted(ci)) if ci else "C"
        legal_ids = [c["token_id"] for c in cards if set(c["color_identity"]).issubset(ci_set)]
        color_identity_groups[ci_key] = legal_ids

    # Save everything
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "cards.json", "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=1)

    with open(output_dir / "card_to_id.json", "w", encoding="utf-8") as f:
        json.dump(card_to_id, f)

    with open(output_dir / "id_to_card.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_card.items()}, f)

    with open(output_dir / "commanders.json", "w", encoding="utf-8") as f:
        json.dump(commanders, f, indent=1)

    with open(output_dir / "color_identity_groups.json", "w", encoding="utf-8") as f:
        json.dump(color_identity_groups, f)

    logger.info("Catalog saved to %s", output_dir)

    return {
        "cards": cards,
        "card_to_id": card_to_id,
        "id_to_card": id_to_card,
        "commanders": commanders,
        "color_identity_groups": color_identity_groups,
    }


def load_catalog(catalog_dir: Path = DEFAULT_CATALOG_DIR) -> dict[str, Any]:
    """Load a previously built catalog from disk."""
    with open(catalog_dir / "cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)

    with open(catalog_dir / "card_to_id.json", "r", encoding="utf-8") as f:
        card_to_id = json.load(f)

    with open(catalog_dir / "id_to_card.json", "r", encoding="utf-8") as f:
        id_to_card_raw = json.load(f)
        id_to_card = {int(k): v for k, v in id_to_card_raw.items()}

    with open(catalog_dir / "commanders.json", "r", encoding="utf-8") as f:
        commanders = json.load(f)

    with open(catalog_dir / "color_identity_groups.json", "r", encoding="utf-8") as f:
        color_identity_groups = json.load(f)

    return {
        "cards": cards,
        "card_to_id": card_to_id,
        "id_to_card": id_to_card,
        "commanders": commanders,
        "color_identity_groups": color_identity_groups,
    }


def get_card_features(cards: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """
    Build a compact feature dict for each card (useful for model feature injection).

    Returns: {token_id: {cmc, color_W, color_U, color_B, color_R, color_G,
                         is_creature, is_instant, is_sorcery, is_artifact,
                         is_enchantment, is_planeswalker, is_land, is_basic_land}}
    """
    features = {}
    for card in cards:
        ci = set(card["color_identity"])
        tc = set(card["type_categories"])
        features[card["token_id"]] = {
            "cmc": min(card["cmc"], 16.0) / 16.0,  # normalize to [0, 1]
            "color_W": 1.0 if "W" in ci else 0.0,
            "color_U": 1.0 if "U" in ci else 0.0,
            "color_B": 1.0 if "B" in ci else 0.0,
            "color_R": 1.0 if "R" in ci else 0.0,
            "color_G": 1.0 if "G" in ci else 0.0,
            "is_creature": 1.0 if "Creature" in tc else 0.0,
            "is_instant": 1.0 if "Instant" in tc else 0.0,
            "is_sorcery": 1.0 if "Sorcery" in tc else 0.0,
            "is_artifact": 1.0 if "Artifact" in tc else 0.0,
            "is_enchantment": 1.0 if "Enchantment" in tc else 0.0,
            "is_planeswalker": 1.0 if "Planeswalker" in tc else 0.0,
            "is_land": 1.0 if card["is_land"] else 0.0,
            "is_basic_land": 1.0 if card["is_basic_land"] else 0.0,
        }
    return features
