"""Scrape MTGGoldfish for Commander decklists."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

GOLDFISH_BASE = "https://www.mtggoldfish.com"
DEFAULT_OUTPUT_DIR = Path("data/raw/mtggoldfish")
DEFAULT_CATALOG_DIR = Path("data/catalog")

# Rate-limiting: be respectful
REQUEST_DELAY = 1.5  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.mtggoldfish.com/",
}


def _load_known_commanders(catalog_dir: Path) -> set[str]:
    """Load commander names from the Scryfall catalog for validation."""
    commanders_file = catalog_dir / "commanders.json"
    if commanders_file.exists():
        with open(commanders_file, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _search_deck_page(page: int = 1) -> list[dict[str, str]]:
    """
    Scrape one page of the MTGGoldfish Commander deck search.

    Returns list of {"deck_id": str, "commander_hint": str} dicts.
    """
    url = f"{GOLDFISH_BASE}/deck/custom/commander"
    params = {"page": page} if page > 1 else {}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            logger.warning("Search page %d returned %d", page, resp.status_code)
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        deck_links = soup.find_all("a", href=re.compile(r"^/deck/\d+$"))
        results = []
        for link in deck_links:
            deck_id = link["href"].split("/")[-1]
            commander_hint = link.get_text(strip=True)
            results.append(
                {
                    "deck_id": deck_id,
                    "commander_hint": commander_hint,
                }
            )
        return results
    except Exception as e:
        logger.warning("Failed to search page %d: %s", page, e)
        return []


def _download_deck(deck_id: str) -> str | None:
    """Download a deck as text from MTGGoldfish."""
    url = f"{GOLDFISH_BASE}/deck/download/{deck_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception as e:
        logger.warning("Failed to download deck %s: %s", deck_id, e)
        return None


def _parse_deck_text(
    text: str,
    commander_hint: str,
    known_commanders: set[str],
) -> dict[str, Any] | None:
    """
    Parse MTGGoldfish deck text format into our standard dict.

    Format: "1 Card Name" per line (sometimes "N Card Name" for basic lands).

    The commander is identified by:
      1. Matching the commander_hint from the search page
      2. Cross-referencing against the known commanders set
    """
    lines = text.strip().split("\n")
    all_cards: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Parse "N Card Name" format
        match = re.match(r"^(\d+)\s+(.+)$", line)
        if match:
            quantity = int(match.group(1))
            card_name = match.group(2).strip()
            for _ in range(quantity):
                all_cards.append(card_name)

    if len(all_cards) < 50:
        return None

    # Identify the commander
    commander = None

    # Strategy 1: exact match of commander_hint in known commanders
    if commander_hint in known_commanders:
        commander = commander_hint

    # Strategy 2: search for any known commander in the card list
    if commander is None:
        for card in all_cards:
            if card in known_commanders:
                commander = card
                break

    # Strategy 3: partial match of hint (handle "Name // Other Side" etc.)
    if commander is None:
        hint_lower = commander_hint.lower()
        for cmd in known_commanders:
            if cmd.lower().startswith(hint_lower) or hint_lower.startswith(cmd.lower()):
                commander = cmd
                break
            # Handle double-faced cards
            if " // " in cmd and cmd.split(" // ")[0].lower() == hint_lower:
                commander = cmd
                break

    if commander is None:
        logger.debug("Could not identify commander for hint '%s'", commander_hint)
        return None

    # Remove commander from the card list
    cards = [c for c in all_cards if c != commander]

    # Commander decks should have ~99 cards (excluding commander)
    if len(cards) < 50:
        return None

    # Trim to 99
    cards = cards[:99]

    return {
        "commander": commander,
        "cards": cards,
        "source": "mtggoldfish",
    }


def scrape_mtggoldfish(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
    max_pages: int = 500,
    max_decks: int | None = None,
) -> list[dict[str, Any]]:
    """
    Scrape Commander decklists from MTGGoldfish user-submitted decks.

    Args:
        output_dir: Directory to save raw JSON data.
        catalog_dir: Directory containing the Scryfall card catalog.
        max_pages: Maximum search pages to crawl (30 decks/page).
        max_decks: Maximum total decks to collect (None = no limit).

    Returns:
        List of decklist dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decks_file = output_dir / "decklists.json"

    # Load known commanders for matching
    known_commanders = _load_known_commanders(catalog_dir)
    if not known_commanders:
        logger.warning("No commander catalog found — commander matching may fail")

    # Load existing progress
    existing: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if decks_file.exists():
        with open(decks_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
            seen_ids = {d.get("deck_id", "") for d in existing if d.get("deck_id")}
        logger.info("Resuming: %d decks already scraped", len(existing))

    new_decks: list[dict[str, Any]] = []
    total = len(existing)
    failed = 0
    skipped = 0

    effective_max = max_decks or (max_pages * 30)
    pbar = tqdm(total=effective_max - total, desc="Scraping MTGGoldfish")

    for page in range(1, max_pages + 1):
        if max_decks and total >= max_decks:
            break

        entries = _search_deck_page(page)
        time.sleep(REQUEST_DELAY)

        if not entries:
            logger.info("No results at page %d, stopping", page)
            break

        for entry in entries:
            if max_decks and total >= max_decks:
                break

            deck_id = entry["deck_id"]
            if deck_id in seen_ids:
                skipped += 1
                continue

            text = _download_deck(deck_id)
            time.sleep(REQUEST_DELAY)

            if text is None:
                failed += 1
                continue

            deck = _parse_deck_text(text, entry["commander_hint"], known_commanders)
            if deck is None:
                failed += 1
                continue

            deck["deck_id"] = deck_id
            new_decks.append(deck)
            seen_ids.add(deck_id)
            total += 1
            pbar.update(1)

            # Save progress every 50 decks
            if len(new_decks) % 50 == 0:
                all_decks = existing + new_decks
                with open(decks_file, "w", encoding="utf-8") as f:
                    json.dump(all_decks, f)
                logger.info(
                    "Progress: %d total decks (%d new, %d failed)", total, len(new_decks), failed
                )

    pbar.close()

    # Final save
    all_decks = existing + new_decks
    with open(decks_file, "w", encoding="utf-8") as f:
        json.dump(all_decks, f)

    logger.info(
        "MTGGoldfish scrape complete: %d new, %d total (%d failed, %d skipped)",
        len(new_decks),
        len(all_decks),
        failed,
        skipped,
    )
    return all_decks
