"""Quick test of Moxfield API endpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

HEADERS = {
    "User-Agent": "MulliganMachine/0.1 (MTG deck ML research project)",
    "Accept": "application/json",
}

# Test search endpoint
print("=== Testing Moxfield Search ===")
r = requests.get(
    "https://api2.moxfield.com/v3/search",
    params={
        "fmt": "commander",
        "page": 1,
        "pageSize": 2,
        "sortType": "views",
        "sortDirection": "Descending",
        "board": "mainboard",
    },
    headers=HEADERS,
    timeout=15,
)
print(f"Status: {r.status_code}")
if r.status_code != 200:
    print(f"Response: {r.text[:500]}")
    sys.exit(1)

data = r.json()
print(f"Keys: {list(data.keys())}")
total = data.get("totalResults", "?")
print(f"Total Commander decks: {total}")
items = data.get("data", [])
print(f"Items returned: {len(items)}")
for item in items[:2]:
    pid = item.get("publicId", "?")
    name = item.get("name", "?")
    print(f"  {pid}: {name}")

if not items:
    print("No search results!")
    sys.exit(1)

# Test fetching a single deck
deck_id = items[0]["publicId"]
print(f"\n=== Fetching deck {deck_id} ===")
r2 = requests.get(
    f"https://api2.moxfield.com/v3/decks/all/{deck_id}",
    headers=HEADERS,
    timeout=15,
)
print(f"Status: {r2.status_code}")
if r2.status_code != 200:
    print(f"Response: {r2.text[:500]}")
    sys.exit(1)

deck = r2.json()
boards = deck.get("boards", {})
print(f"Boards: {list(boards.keys())}")

commanders = boards.get("commanders", {}).get("cards", {})
cmd_names = [v["card"]["name"] for v in commanders.values()]
print(f"Commander(s): {cmd_names}")

mainboard = boards.get("mainboard", {}).get("cards", {})
card_count = sum(v.get("quantity", 1) for v in mainboard.values())
print(f"Mainboard cards: {len(mainboard)} unique, {card_count} total")

print("\nMoxfield API is working!")
