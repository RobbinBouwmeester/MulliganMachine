"""Scrape EDHREC for average Commander decklists."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

EDHREC_BASE = "https://json.edhrec.com/pages"
DEFAULT_OUTPUT_DIR = Path("data/raw/edhrec")
DEFAULT_CATALOG_DIR = Path("data/catalog")

# Rate-limiting: EDHREC is a community resource, be respectful
REQUEST_DELAY = 1.0  # seconds between requests

# Browser-like headers required for EDHREC JSON endpoints
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://edhrec.com/",
}


def _name_to_slug(name: str) -> str:
    """
    Convert a card name to an EDHREC URL slug.

    E.g. "Atraxa, Praetors' Voice" -> "atraxa-praetors-voice"
         "Korvold, Fae-Cursed King" -> "korvold-fae-cursed-king"
    """
    slug = name.lower()
    # Handle double-faced cards: take front face only
    if " // " in slug:
        slug = slug.split(" // ")[0]
    # Remove apostrophes, commas, and other punctuation
    slug = re.sub(r"[',\.\!\?\"\:\;]", "", slug)
    # Replace spaces and non-alphanumeric with dashes
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Clean up multiple or trailing dashes
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _get_commander_list(catalog_dir: Path = DEFAULT_CATALOG_DIR) -> list[dict[str, str]]:
    """
    Build commander list from the Scryfall catalog (no EDHREC API call needed).

    Returns list of dicts with 'name' and 'url_slug' keys.
    """
    commanders_file = catalog_dir / "commanders.json"
    if not commanders_file.exists():
        raise FileNotFoundError(
            f"No commanders file at {commanders_file}. Run scrape_scryfall.py first."
        )

    with open(commanders_file, "r", encoding="utf-8") as f:
        commander_names: list[str] = json.load(f)

    commanders = [{"name": name, "url_slug": _name_to_slug(name)} for name in commander_names]

    logger.info("Loaded %d commanders from catalog", len(commanders))
    return commanders


def _fetch_commander_page(url_slug: str) -> dict[str, Any] | None:
    """Fetch the EDHREC JSON page for a specific commander."""
    url = f"{EDHREC_BASE}/commanders/{url_slug}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code in (403, 404):
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


def _extract_average_deck(page_data: dict[str, Any], commander_name: str) -> dict[str, Any] | None:
    """
    Extract the average decklist from an EDHREC commander page.

    Returns: {"commander": str, "cards": list[str], "num_decks": int}
    """
    container = page_data.get("container", {})
    json_dict = container.get("json_dict", {})

    # The "cardlists" contain card recommendations with inclusion percentages
    cardlists = json_dict.get("cardlists", [])

    all_cards: list[dict[str, Any]] = []
    for section in cardlists:
        for card in section.get("cardviews", []):
            name = card.get("name", "")
            # Inclusion rate (0-100)
            inclusion = card.get("inclusion", 0)
            # Number of decks this data is based on
            num_decks = card.get("num_decks", 0)
            # Label can indicate if it's a land, new card, etc.
            label = card.get("label", "")

            if name and name != commander_name:
                all_cards.append(
                    {
                        "name": name,
                        "inclusion": inclusion,
                        "num_decks": num_decks,
                        "label": label,
                    }
                )

    if not all_cards:
        return None

    # Sort by inclusion rate (descending) and take top 99
    all_cards.sort(key=lambda c: c["inclusion"], reverse=True)
    top_cards = [c["name"] for c in all_cards[:99]]

    num_decks = max((c["num_decks"] for c in all_cards), default=0)

    return {
        "commander": commander_name,
        "cards": top_cards,
        "num_decks": num_decks,
        "source": "edhrec",
    }


def scrape_edhrec(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
    max_commanders: int | None = None,
    min_decks: int = 100,
) -> list[dict[str, Any]]:
    """
    Scrape EDHREC for average decklists of all popular commanders.

    Args:
        output_dir: Directory to save raw JSON data.
        catalog_dir: Directory containing the Scryfall card catalog.
        max_commanders: Limit number of commanders to scrape (None = all).
        min_decks: Skip commanders with fewer than this many decks on EDHREC.

    Returns:
        List of decklist dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decks_file = output_dir / "decklists.json"

    # Load existing progress if any
    existing_decks: list[dict[str, Any]] = []
    scraped_names: set[str] = set()
    if decks_file.exists():
        with open(decks_file, "r", encoding="utf-8") as f:
            existing_decks = json.load(f)
            scraped_names = {d["commander"] for d in existing_decks}
        logger.info("Resuming: %d commanders already scraped", len(scraped_names))

    # Get commander list from catalog
    commanders = _get_commander_list(catalog_dir)
    if max_commanders:
        commanders = commanders[:max_commanders]

    # Filter out already-scraped
    to_scrape = [c for c in commanders if c["name"] not in scraped_names]
    logger.info("Scraping %d commanders (%d already done)", len(to_scrape), len(scraped_names))

    new_decks: list[dict[str, Any]] = []

    for cmd in tqdm(to_scrape, desc="Scraping EDHREC"):
        page = _fetch_commander_page(cmd["url_slug"])
        time.sleep(REQUEST_DELAY)

        if page is None:
            continue

        deck = _extract_average_deck(page, cmd["name"])
        if deck is None:
            continue

        if deck["num_decks"] < min_decks:
            logger.debug(
                "Skipping %s (only %d decks, need %d)",
                cmd["name"],
                deck["num_decks"],
                min_decks,
            )
            continue

        if len(deck["cards"]) < 50:
            logger.debug("Skipping %s (only %d cards extracted)", cmd["name"], len(deck["cards"]))
            continue

        new_decks.append(deck)

        # Save progress every 50 commanders
        if len(new_decks) % 50 == 0:
            all_decks = existing_decks + new_decks
            with open(decks_file, "w", encoding="utf-8") as f:
                json.dump(all_decks, f, indent=1)
            logger.info("Progress saved: %d total decks", len(all_decks))

    # Final save
    all_decks = existing_decks + new_decks
    with open(decks_file, "w", encoding="utf-8") as f:
        json.dump(all_decks, f, indent=1)

    logger.info(
        "EDHREC scrape complete: %d new decks, %d total",
        len(new_decks),
        len(all_decks),
    )

    return all_decks


def load_edhrec_decks(data_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict[str, Any]]:
    """Load previously scraped EDHREC decklists."""
    decks_file = data_dir / "decklists.json"
    if not decks_file.exists():
        raise FileNotFoundError(f"No EDHREC data found at {decks_file}. Run scrape_edhrec() first.")

    with open(decks_file, "r", encoding="utf-8") as f:
        return json.load(f)
