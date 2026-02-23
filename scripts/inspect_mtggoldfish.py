"""Deeper inspection of MTGGoldfish commander data structure."""

import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 1. Get all commander archetypes from metagame page
print("=== Commander Archetypes ===")
r = requests.get("https://www.mtggoldfish.com/metagame/commander", headers=HEADERS, timeout=15)
soup = BeautifulSoup(r.text, "lxml")
links = soup.find_all("a", href=re.compile(r"/archetype/commander-"))
slugs = set()
for link in links:
    href = link["href"].split("#")[0]
    slugs.add(href)
print(f"Unique commander archetypes on metagame page: {len(slugs)}")
for s in sorted(slugs)[:10]:
    print(f"  {s}")

# 2. Check if there are more pages of archetypes
print("\n=== Checking Pagination ===")
for page_num in [2, 3]:
    r_page = requests.get(
        f"https://www.mtggoldfish.com/metagame/commander/full#paper",
        headers=HEADERS,
        timeout=15,
    )
    if r_page.status_code == 200:
        page_links = r_page.text.count("/archetype/commander-")
        print(f"  /metagame/commander/full: {page_links} archetype references")
    break

# 3. Check an archetype page in detail — what decks does it contain?
print("\n=== Archetype Page: Atraxa ===")
r2 = requests.get(
    "https://www.mtggoldfish.com/archetype/commander-atraxa-praetors-voice",
    headers=HEADERS,
    timeout=15,
)
soup2 = BeautifulSoup(r2.text, "lxml")

# Find all deck download links
download_links = set()
for a in soup2.find_all("a", href=re.compile(r"/deck/download/")):
    download_links.add(a["href"].split("?")[0])
print(f"Download links: {len(download_links)}")
for dl in sorted(download_links)[:5]:
    print(f"  {dl}")

# Find individual deck links
deck_ids = set(re.findall(r"/deck/(\d+)", r2.text))
print(f"Unique deck IDs: {len(deck_ids)}")

# Check the "decks" tab if it exists
print("\n=== Archetype Decks Tab ===")
r3 = requests.get(
    "https://www.mtggoldfish.com/archetype/commander-atraxa-praetors-voice#paper",
    headers=HEADERS,
    timeout=15,
)
soup3 = BeautifulSoup(r3.text, "lxml")
deck_ids_paper = set(re.findall(r"/deck/(\d+)", r3.text))
print(f"Deck IDs on paper tab: {len(deck_ids_paper)}")

# 4. Download one of those decks to see the format
if deck_ids:
    test_id = sorted(deck_ids)[0]
    print(f"\n=== Downloading deck {test_id} ===")
    r4 = requests.get(
        f"https://www.mtggoldfish.com/deck/download/{test_id}",
        headers=HEADERS,
        timeout=15,
    )
    print(f"Status: {r4.status_code}")
    if r4.status_code == 200:
        text = r4.text.strip()
        lines = text.split("\n")
        print(f"Total lines: {len(lines)}")
        # Check for commander indication
        for line in lines[:5]:
            print(f"  {line}")
        print("  ...")
        for line in lines[-5:]:
            print(f"  {line}")

# 5. Check the tournament/user deck search
print("\n=== Deck Search ===")
r5 = requests.get(
    "https://www.mtggoldfish.com/deck/custom/commander",
    headers=HEADERS,
    timeout=15,
)
print(f"Custom deck search status: {r5.status_code}")
if r5.status_code == 200:
    soup5 = BeautifulSoup(r5.text, "lxml")
    deck_ids_custom = set(re.findall(r"/deck/(\d+)", r5.text))
    print(f"Deck IDs on custom page: {len(deck_ids_custom)}")
    # Look for pagination
    pagination = soup5.select("a[href*='page=']")
    pages_found = set()
    for p in pagination:
        href = p.get("href", "")
        m = re.search(r"page=(\d+)", href)
        if m:
            pages_found.add(int(m.group(1)))
    if pages_found:
        print(f"Pagination pages: up to {max(pages_found)}")

print("\n=== Summary ===")
print(f"Metagame archetypes: {len(slugs)}")
print(f"Decks downloadable per archetype: ~{len(deck_ids)}")
print("Deck format: '1 Card Name' per line, clean text")
