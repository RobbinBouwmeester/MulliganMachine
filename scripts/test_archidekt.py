"""Quick test of Archidekt API endpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

HEADERS = {
    "User-Agent": "MulliganMachine/0.1 (MTG deck ML research project)",
    "Accept": "application/json",
}

# Test search endpoint
print("=== Testing Archidekt Search ===")
r = requests.get(
    "https://archidekt.com/api/decks/cards/",
    params={
        "formats": 3,  # Commander
        "page": 1,
        "pageSize": 2,
        "ordering": "-viewCount",
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
count = data.get("count", "?")
print(f"Total Commander decks: {count}")
results = data.get("results", [])
print(f"Items returned: {len(results)}")
for item in results[:2]:
    did = item.get("id", "?")
    name = item.get("name", "?")
    print(f"  {did}: {name}")

if not results:
    print("No search results!")
    sys.exit(1)

# Test fetching a single deck
deck_id = results[0]["id"]
print(f"\n=== Fetching deck {deck_id} ===")
r2 = requests.get(
    f"https://archidekt.com/api/decks/{deck_id}/",
    headers=HEADERS,
    timeout=15,
)
print(f"Status: {r2.status_code}")
if r2.status_code != 200:
    print(f"Response: {r2.text[:500]}")
    sys.exit(1)

deck = r2.json()
print(f"Top keys: {list(deck.keys())[:10]}")
cards = deck.get("cards", [])
print(f"Cards count: {len(cards)}")
if cards:
    c = cards[0]
    print(f"Card sample keys: {list(c.keys())}")
    card_info = c.get("card", {})
    oracle = card_info.get("oracleCard", {})
    print(f"  name: {oracle.get('name', '?')}")
    print(f"  categories: {c.get('categories', [])}")

print("\nArchidekt API is working!")
