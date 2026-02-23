"""Extract commander info from MTGGoldfish deck pages."""

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

# Test a few deck pages to find commander identification
deck_ids = ["7643012", "7643003", "7643001"]

for did in deck_ids:
    print(f"\n=== Deck {did} ===")
    r = requests.get(f"https://www.mtggoldfish.com/deck/{did}", headers=HEADERS, timeout=15)
    time.sleep(1)
    if r.status_code != 200:
        print(f"  Status: {r.status_code}")
        continue

    soup = BeautifulSoup(r.text, "lxml")

    # Title
    h1 = soup.select_one("h1, .deck-view-title")
    if h1:
        print(f"  Title: {h1.get_text(strip=True)[:80]}")

    # Look for commander category in deck display
    # MTGGoldfish groups cards by type, and commander is usually in its own section
    headers = soup.select(".deck-category-header")
    for hdr in headers:
        text = hdr.get_text(strip=True)
        print(f"  Category: {text}")

    # Alternative: look for "Commander" in any element
    commander_els = soup.find_all(string=re.compile(r"Commander", re.IGNORECASE))
    for el in commander_els[:5]:
        parent = el.parent
        if parent:
            tag = parent.name
            text = parent.get_text(strip=True)[:80]
            print(f"  Commander ref [{tag}]: {text}")

    # Check the download to see if commander is separated
    r2 = requests.get(
        f"https://www.mtggoldfish.com/deck/download/{did}", headers=HEADERS, timeout=15
    )
    time.sleep(0.5)
    if r2.status_code == 200:
        text = r2.text.strip()
        lines = text.split("\n")
        print(f"  Download: {len(lines)} lines")
        # Check if there's a blank line separating commander from deck
        for i, line in enumerate(lines):
            if line.strip() == "":
                print(f"  Blank line at position {i}")
        print(f"  First 3 lines: {lines[:3]}")
        print(f"  Last 3 lines: {lines[-3:]}")
