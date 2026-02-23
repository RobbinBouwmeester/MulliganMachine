"""Scrape Archidekt for Commander decklists."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

ARCHIDEKT_API = "https://archidekt.com/api"
DEFAULT_OUTPUT_DIR = Path("data/raw/archidekt")

REQUEST_DELAY = 1.0
HEADERS = {
    "User-Agent": "MulliganMachine/0.1 (MTG deck ML research project)",
    "Accept": "application/json",
}

# Archidekt format IDs: 3 = Commander / EDH
COMMANDER_FORMAT_ID = 3


def _search_decks(
    page: int = 1,
    page_size: int = 50,
    ordering: str = "-viewCount",
) -> dict[str, Any] | None:
    """Search Archidekt for public Commander decks."""
    url = f"{ARCHIDEKT_API}/decks/cards/"
    params = {
        "formats": COMMANDER_FORMAT_ID,
        "page": page,
        "pageSize": page_size,
        "ordering": ordering,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            logger.warning("Rate limited by Archidekt, sleeping 30s...")
            time.sleep(30)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Archidekt search failed (page %d): %s", page, e)
        return None


def _fetch_deck(deck_id: int) -> dict[str, Any] | None:
    """Fetch a single deck from Archidekt."""
    url = f"{ARCHIDEKT_API}/decks/{deck_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            time.sleep(30)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch Archidekt deck %d: %s", deck_id, e)
        return None


def _extract_decklist(deck_data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract commander and card list from Archidekt deck data."""
    cards_data = deck_data.get("cards", [])
    if not cards_data:
        return None

    commander = None
    cards: list[str] = []

    for entry in cards_data:
        card = entry.get("card", {})
        oracle_card = card.get("oracleCard", {})
        name = oracle_card.get("name", "") or card.get("name", "")
        quantity = entry.get("quantity", 1)
        categories = entry.get("categories", [])

        if not name:
            continue

        # Check if this card is in the "Commander" category
        if "Commander" in categories:
            if commander is None:
                commander = name
            continue

        for _ in range(quantity):
            cards.append(name)

    if commander is None or len(cards) < 50:
        return None

    return {
        "commander": commander,
        "cards": cards,
        "source": "archidekt",
        "deck_id": deck_data.get("id", ""),
        "name": deck_data.get("name", ""),
    }


def scrape_archidekt(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_decks: int = 10_000,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """
    Scrape Commander decklists from Archidekt.

    Args:
        output_dir: Directory to save data.
        max_decks: Maximum number of decks to collect.
        page_size: Results per page.

    Returns:
        List of decklist dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decks_file = output_dir / "decklists.json"

    existing: list[dict[str, Any]] = []
    seen_ids: set = set()
    if decks_file.exists():
        with open(decks_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
            seen_ids = {d.get("deck_id", "") for d in existing}
        logger.info("Resuming: %d decks already scraped", len(existing))

    new_decks: list[dict[str, Any]] = []
    total_collected = len(existing)
    page = 1
    max_pages = max_decks // page_size + 1

    pbar = tqdm(total=max_decks - total_collected, desc="Scraping Archidekt")

    while total_collected < max_decks and page <= max_pages:
        search_result = _search_decks(page=page, page_size=page_size)
        time.sleep(REQUEST_DELAY)

        if search_result is None:
            page += 1
            continue

        results = search_result.get("results", [])
        if not results:
            logger.info("No more results at page %d", page)
            break

        for result in results:
            if total_collected >= max_decks:
                break

            deck_id = result.get("id")
            if deck_id is None or deck_id in seen_ids:
                continue

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

            if len(new_decks) % 100 == 0:
                all_decks = existing + new_decks
                with open(decks_file, "w", encoding="utf-8") as f:
                    json.dump(all_decks, f)
                logger.info("Progress: %d total decks saved", len(all_decks))

        page += 1

    pbar.close()

    all_decks = existing + new_decks
    with open(decks_file, "w", encoding="utf-8") as f:
        json.dump(all_decks, f)

    logger.info("Archidekt scrape: %d new, %d total", len(new_decks), len(all_decks))
    return all_decks
