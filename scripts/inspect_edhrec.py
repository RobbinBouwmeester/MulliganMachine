"""Inspect an EDHREC commander page to verify structure."""

import sys, json, requests

sys.path.insert(0, "src")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://edhrec.com/",
}

url = "https://json.edhrec.com/pages/commanders/atraxa-praetors-voice.json"
resp = requests.get(url, headers=HEADERS, timeout=30)
print(f"Status: {resp.status_code}")

data = resp.json()
print(f"Top-level keys: {list(data.keys())}")

container = data.get("container", {})
print(f"Container keys: {list(container.keys())}")

json_dict = container.get("json_dict", {})
print(f"json_dict keys: {list(json_dict.keys())}")

cardlists = json_dict.get("cardlists", [])
print(f"Number of cardlists: {len(cardlists)}")

total_cards = 0
for i, section in enumerate(cardlists):
    header = section.get("header", "?")
    cards = section.get("cardviews", [])
    total_cards += len(cards)
    print(f"  Section {i}: '{header}' - {len(cards)} cards")
    if cards:
        c = cards[0]
        print(f"    Sample card keys: {list(c.keys())}")
        print(
            f"    Sample: name={c.get('name')}, inclusion={c.get('inclusion')}, num_decks={c.get('num_decks')}"
        )

print(f"\nTotal recommended cards: {total_cards}")
