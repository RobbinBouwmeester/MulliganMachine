"""Inspect what card features are available in the Scryfall catalog."""
import json
from collections import Counter

cards = json.load(open("data/catalog/cards.json"))

types = set()
all_keywords = set()
rarities = set()
has_oracle = 0
max_oracle_len = 0
avg_oracle_len = 0

for c in cards:
    types.update(c.get("type_categories", []))
    all_keywords.update(c.get("keywords", []))
    rarities.add(c.get("rarity", ""))
    if c.get("oracle_text"):
        has_oracle += 1
        max_oracle_len = max(max_oracle_len, len(c["oracle_text"]))
        avg_oracle_len += len(c["oracle_text"])

print(f"Total cards: {len(cards)}")
print(f"Cards with oracle text: {has_oracle} ({has_oracle/len(cards)*100:.0f}%)")
print(f"Avg oracle text length: {avg_oracle_len/max(has_oracle,1):.0f} chars")
print(f"Max oracle text length: {max_oracle_len} chars")
print(f"Type categories: {sorted(types)}")
print(f"Unique keywords: {len(all_keywords)}")
kw_sorted = sorted(all_keywords)
print(f"Sample keywords (first 30): {kw_sorted[:30]}")
print(f"Rarities: {sorted(rarities)}")

cmcs = [int(c["cmc"]) for c in cards if c.get("cmc") is not None]
print(f"CMC range: {min(cmcs)}-{max(cmcs)}")
print(f"CMC dist (top 10): {Counter(cmcs).most_common(10)}")

# Count keywords per card
kw_counts = [len(c.get("keywords", [])) for c in cards]
print(f"\nKeywords per card: avg={sum(kw_counts)/len(kw_counts):.1f}, max={max(kw_counts)}")

# Show a few example cards with rich oracle text
print("\n=== Example cards ===")
for name in ["Sol Ring", "Rhystic Study", "Swords to Plowshares", "Atraxa, Praetors' Voice"]:
    for c in cards:
        if c["name"] == name:
            print(f"\n{c['name']} ({c['mana_cost']}) - {c['type_line']}")
            print(f"  CMC: {c['cmc']}, Colors: {c['color_identity']}, Keywords: {c['keywords']}")
            print(f"  Oracle: {c.get('oracle_text', 'N/A')[:200]}")
            break
