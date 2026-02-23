"""Test MTGGoldfish deck page with different approaches."""

import re
import time
import requests
from bs4 import BeautifulSoup

HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.mtggoldfish.com/deck/custom/commander",
}

did = "7643012"

# Try with full browser headers
print(f"=== Deck {did} with HTML Accept ===")
r = requests.get(f"https://www.mtggoldfish.com/deck/{did}", headers=HEADERS_HTML, timeout=15)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type', '?')}")

if r.status_code == 200:
    soup = BeautifulSoup(r.text, "lxml")
    title = soup.select_one("title")
    print(f"Title: {title.get_text(strip=True) if title else '?'}")
else:
    print(f"Body preview: {r.text[:300]}")

# The deck download works - let's check if the deck name on the search page
# gives us the commander name
print(f"\n=== Search page deck info ===")
time.sleep(1)
r2 = requests.get(
    "https://www.mtggoldfish.com/deck/custom/commander",
    headers=HEADERS_HTML,
    timeout=15,
)
soup2 = BeautifulSoup(r2.text, "lxml")

# Find all deck entries with their names
# The deck links show the commander name as link text
deck_links = soup2.find_all("a", href=re.compile(r"^/deck/\d+$"))
for link in deck_links[:10]:
    did = link["href"].split("/")[-1]
    name = link.get_text(strip=True)
    # The link text IS the commander name on the search page
    print(f"  Deck {did}: '{name}'")

# The download doesn't separate the commander. But since the search page
# tells us the commander name AND the download gives us all cards,
# we can: 1) Get commander from search, 2) Download full list, 3) Remove commander from list

# Actually, let's check if the deck/download includes commander info
print("\n=== Download format check ===")
time.sleep(1)
# Download a known deck
r3 = requests.get(
    "https://www.mtggoldfish.com/deck/download/7643012",
    headers=HEADERS_HTML,
    timeout=15,
)
print(f"Status: {r3.status_code}")
if r3.status_code == 200:
    lines = r3.text.strip().split("\n")
    print(f"Total lines: {len(lines)}")
    # Show all lines to check for commander marker
    for i, line in enumerate(lines):
        if i < 5 or i > len(lines) - 5 or line.strip() == "":
            print(f"  [{i:3d}] '{line}'")

# Alternative approach: use the #commander hash or the sidebar
# Check MTGGoldfish API endpoint if one exists
print("\n=== Testing API endpoint ===")
time.sleep(1)
r4 = requests.get(
    f"https://www.mtggoldfish.com/deck/{did}.json",
    headers=HEADERS_HTML,
    timeout=15,
)
print(f"JSON endpoint status: {r4.status_code}")

# Check the visual page (/deck/visual/ID)
time.sleep(1)
r5 = requests.get(
    f"https://www.mtggoldfish.com/deck/visual/7643012",
    headers=HEADERS_HTML,
    timeout=15,
)
print(f"Visual endpoint status: {r5.status_code}")
if r5.status_code == 200:
    soup5 = BeautifulSoup(r5.text, "lxml")
    title5 = soup5.select_one("title")
    print(f"Visual title: {title5.get_text(strip=True) if title5 else '?'}")
