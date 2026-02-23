"""Scrape Moxfield for Commander decklists."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

MOXFIELD_API = "https://api2.moxfield.com/v3"
DEFAULT_OUTPUT_DIR = Path("data/raw/moxfield")

# Be respectful of Moxfield's servers
REQUEST_DELAY = 1.5  # seconds between requests
HEADERS = {
    "User-Agent": "MulliganMachine/0.1 (MTG deck ML research project)",
    "Accept": "application/json",
}


def _search_decks(
    page: int = 1,
    page_size: int = 64,
    fmt: str = "commander",
    sort: str = "views",
) -> dict[str, Any] | None:
    """Search Moxfield for public Commander decks."""
    url = f"{MOXFIELD_API}/search"
    params = {
        "fmt": fmt,
        "page": page,
        "pageSize": page_size,
        "sortType": sort,
        "sortDirection": "Descending",
        "board": "mainboard",
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            logger.warning("Rate limited by Moxfield, sleeping 30s...")
            time.sleep(30)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Moxfield search failed (page %d): %s", page, e)
        return None


def _fetch_deck(deck_id: str) -> dict[str, Any] | None:
    """Fetch a single deck's full details from Moxfield."""
    url = f"{MOXFIELD_API}/decks/all/{deck_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            logger.warning("Rate limited, sleeping 30s...")
            time.sleep(30)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch deck %s: %s", deck_id, e)
        return None


def _extract_decklist(deck_data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract commander and card list from Moxfield deck data."""
    # Get commander(s) from commanders board
    commanders_board = deck_data.get("boards", {}).get("commanders", {}).get("cards", {})
    if not commanders_board:
        return None

    commander_names = []
    for card_key, card_data in commanders_board.items():
        card = card_data.get("card", {})
        name = card.get("name", "")
        if name:
            commander_names.append(name)

    if not commander_names:
        return None

    # For simplicity, use the first commander (handle partners as a single string later)
    commander = commander_names[0]

    # Get mainboard cards
    mainboard = deck_data.get("boards", {}).get("mainboard", {}).get("cards", {})
    cards = []
    for card_key, card_data in mainboard.items():
        card = card_data.get("card", {})
        name = card.get("name", "")
        quantity = card_data.get("quantity", 1)
        if name:
            # In Commander, quantity should be 1 (except basic lands)
            for _ in range(quantity):
                cards.append(name)

    if len(cards) < 50:
        return None

    return {
        "commander": commander,
        "partner": commander_names[1] if len(commander_names) > 1 else None,
        "cards": cards,
        "source": "moxfield",
        "deck_id": deck_data.get("publicId", ""),
        "name": deck_data.get("name", ""),
    }


def scrape_moxfield(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_decks: int = 10_000,
    page_size: int = 64,
) -> list[dict[str, Any]]:
    """
    Scrape Commander decklists from Moxfield.

    Args:
        output_dir: Directory to save data.
        max_decks: Maximum number of decks to collect.
        page_size: Results per search page.

    Returns:
        List of decklist dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decks_file = output_dir / "decklists.json"

    # Load existing progress
    existing: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if decks_file.exists():
        with open(decks_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
            seen_ids = {d.get("deck_id", "") for d in existing}
        logger.info("Resuming: %d decks already scraped", len(existing))

    new_decks: list[dict[str, Any]] = []
    total_collected = len(existing)
    page = 1
    max_pages = max_decks // page_size + 1

    pbar = tqdm(total=max_decks - total_collected, desc="Scraping Moxfield")

    while total_collected < max_decks and page <= max_pages:
        search_result = _search_decks(page=page, page_size=page_size)
        time.sleep(REQUEST_DELAY)

        if search_result is None:
            page += 1
            continue

        deck_summaries = search_result.get("data", [])
        if not deck_summaries:
            logger.info("No more results at page %d", page)
            break

        for summary in deck_summaries:
            if total_collected >= max_decks:
                break

            deck_id = summary.get("publicId", "")
            if deck_id in seen_ids:
                continue

            # Fetch full deck
            deck_data = _fetch_deck(deck_id)
            time.sleep(REQUEST_DELAY)

            if deck_data is None:
                continue

            decklist = _extract_decklist(deck_data)
            if decklist is None:
                continue

            new_decks.append(decklist)
            seen_ids.add(deck_id)
            total_collected += 1
            pbar.update(1)

            # Save progress periodically
            if len(new_decks) % 100 == 0:
                all_decks = existing + new_decks
                with open(decks_file, "w", encoding="utf-8") as f:
                    json.dump(all_decks, f)
                logger.info("Progress: %d total decks saved", len(all_decks))

        page += 1

    pbar.close()

    # Final save
    all_decks = existing + new_decks
    with open(decks_file, "w", encoding="utf-8") as f:
        json.dump(all_decks, f)

    logger.info("Moxfield scrape: %d new, %d total", len(new_decks), len(all_decks))
    return all_decks
