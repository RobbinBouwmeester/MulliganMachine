"""Inspect MTGGoldfish user-submitted commander deck search."""

import re
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Inspect the custom deck search page structure
print("=== Custom Commander Deck Search Page 1 ===")
r = requests.get(
    "https://www.mtggoldfish.com/deck/custom/commander",
    headers=HEADERS,
    timeout=15,
)
soup = BeautifulSoup(r.text, "lxml")

# Find deck entries
deck_links = soup.find_all("a", href=re.compile(r"^/deck/\d+$"))
print(f"Deck links on page: {len(deck_links)}")
for link in deck_links[:5]:
    print(f"  {link['href']}: {link.get_text(strip=True)[:60]}")

# Try to find deck IDs + names from the table
tables = soup.select("table")
print(f"Tables found: {len(tables)}")
for i, table in enumerate(tables[:3]):
    rows = table.select("tr")
    print(f"  Table {i}: {len(rows)} rows")

# Find all deck IDs on the page
all_ids = set(re.findall(r"/deck/(\d+)", r.text))
print(f"Total unique deck IDs on page: {len(all_ids)}")

# Try downloading one of those decks
if all_ids:
    test_id = sorted(all_ids)[0]
    print(f"\n=== Downloading deck {test_id} ===")
    r2 = requests.get(
        f"https://www.mtggoldfish.com/deck/download/{test_id}",
        headers=HEADERS,
        timeout=15,
    )
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        text = r2.text.strip()
        lines = text.split("\n")
        print(f"Lines: {len(lines)}")
        for line in lines[:8]:
            print(f"  {line}")
        if len(lines) > 8:
            print(f"  ... ({len(lines) - 8} more)")

    # Also check the deck page itself to extract commander info
    print(f"\n=== Deck page {test_id} ===")
    time.sleep(1)
    r3 = requests.get(
        f"https://www.mtggoldfish.com/deck/{test_id}",
        headers=HEADERS,
        timeout=15,
    )
    if r3.status_code == 200:
        soup3 = BeautifulSoup(r3.text, "lxml")
        title = soup3.select_one("h1.deck-view-title, .title")
        print(f"Title tag: {title.get_text(strip=True) if title else 'not found'}")

        # Look for commander section
        sections = soup3.select(".deck-category-header")
        for sec in sections:
            text = sec.get_text(strip=True)
            print(f"  Category: {text}")

# Try page 2 to confirm pagination works
print("\n=== Page 2 ===")
time.sleep(1)
r4 = requests.get(
    "https://www.mtggoldfish.com/deck/custom/commander?page=2",
    headers=HEADERS,
    timeout=15,
)
print(f"Status: {r4.status_code}")
if r4.status_code == 200:
    page2_ids = set(re.findall(r"/deck/(\d+)", r4.text))
    print(f"Deck IDs on page 2: {len(page2_ids)}")
    overlap = all_ids & page2_ids
    print(f"Overlap with page 1: {len(overlap)}")

# Quick estimate
print("\n=== Estimate ===")
print(f"~{len(all_ids)} decks per page x 1259 pages = ~{len(all_ids) * 1259} total decks")
print("Each deck downloadable as clean '1 Card Name' text format")
