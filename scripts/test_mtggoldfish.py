"""Test MTGGoldfish as a data source for Commander decklists."""

import requests
import sys

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# 1. Test Commander metagame page (lists popular commanders)
print("=== 1. Commander Metagame Page ===")
r = requests.get("https://www.mtggoldfish.com/metagame/commander", headers=HEADERS, timeout=15)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    # Check for deck links
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(r.text, "lxml")
    # Look for archetype/deck links
    links = soup.select("a[href*='/archetype/']")
    print(f"Archetype links found: {len(links)}")
    for link in links[:5]:
        print(f"  {link.get('href', '?')}: {link.get_text(strip=True)[:60]}")

# 2. Test a specific commander deck page
print("\n=== 2. Sample Deck Page ===")
r2 = requests.get("https://www.mtggoldfish.com/archetype/commander", headers=HEADERS, timeout=15)
print(f"Status: {r2.status_code}")

# 3. Test the deck download endpoint (MTGGoldfish has a CSV/text download for decks)
print("\n=== 3. Testing Deck Download Formats ===")
# Try the popular decks page
r3 = requests.get(
    "https://www.mtggoldfish.com/metagame/commander#paper", headers=HEADERS, timeout=15
)
print(f"Commander metagame #paper status: {r3.status_code}")
if r3.status_code == 200:
    soup3 = BeautifulSoup(r3.text, "lxml")
    # Look for deck links with IDs
    deck_links = soup3.select("a[href*='/deck/']")
    print(f"Deck links: {len(deck_links)}")
    for link in deck_links[:5]:
        href = link.get("href", "")
        text = link.get_text(strip=True)[:60]
        print(f"  {href}: {text}")

    # Look for archetype links
    arch_links = soup3.select("a[href*='/archetype/']")
    print(f"Archetype links: {len(arch_links)}")
    for link in arch_links[:5]:
        href = link.get("href", "")
        text = link.get_text(strip=True)[:60]
        print(f"  {href}: {text}")

# 4. Test MTGGoldfish's deck download (they offer .txt downloads)
print("\n=== 4. Testing Text Deck Download ===")
# Typical format: /deck/download/{deck_id}
# First find a deck ID from the page
if r3.status_code == 200:
    import re

    deck_ids = re.findall(r"/deck/(\d+)", r3.text)
    unique_ids = list(set(deck_ids))
    print(f"Found {len(unique_ids)} unique deck IDs")
    if unique_ids:
        test_id = unique_ids[0]
        print(f"Testing download for deck ID: {test_id}")
        r4 = requests.get(
            f"https://www.mtggoldfish.com/deck/download/{test_id}",
            headers=HEADERS,
            timeout=15,
        )
        print(f"Download status: {r4.status_code}")
        if r4.status_code == 200:
            lines = r4.text.strip().split("\n")
            print(f"Lines in deck: {len(lines)}")
            for line in lines[:10]:
                print(f"  {line}")
            print("  ...")
            for line in lines[-3:]:
                print(f"  {line}")

# 5. Test the "average deck" page for a commander
print("\n=== 5. Average Deck for a Commander ===")
r5 = requests.get(
    "https://www.mtggoldfish.com/archetype/atraxa-praetors-voice",
    headers=HEADERS,
    timeout=15,
)
print(f"Atraxa archetype page status: {r5.status_code}")
if r5.status_code == 200:
    soup5 = BeautifulSoup(r5.text, "lxml")
    title = soup5.select_one("title")
    print(f"Title: {title.get_text(strip=True) if title else '?'}")
    # Look for the deck table
    deck_table = soup5.select(".deck-container .deck-col .deck-category")
    print(f"Deck categories: {len(deck_table)}")
    for cat in deck_table[:3]:
        header = cat.select_one(".deck-category-header")
        cards = cat.select(".deck-category-card-text a")
        h_text = header.get_text(strip=True) if header else "?"
        print(f"  {h_text}: {len(cards)} cards")

    # Check for download link
    download_links = soup5.select("a[href*='/deck/download/']")
    print(f"Download links: {len(download_links)}")
    for dl in download_links[:3]:
        print(f"  {dl.get('href', '?')}")

print("\nDone!")
