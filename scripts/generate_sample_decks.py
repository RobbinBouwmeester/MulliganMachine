"""Generate and display decks for several popular commanders."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging

logging.basicConfig(level=logging.WARNING)

from mulligan_machine.inference.generator import load_generator

CHECKPOINT = "checkpoints/last.ckpt"
CATALOG = Path("data/catalog")

print("Loading model...")
gen = load_generator(CHECKPOINT, CATALOG, device="cuda")

commanders = [
    "Atraxa, Praetors' Voice",
    "Krenko, Mob Boss",
    "Meren of Clan Nel Toth",
]

for cmdr in commanders:
    print(f"\n{'='*70}")
    print(f"  COMMANDER: {cmdr}")
    print(f"{'='*70}")
    try:
        deck = gen.generate(cmdr, temperature=0.75, top_k=80, top_p=0.92)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    cards = deck["cards"]
    n_lands = deck["n_lands"]
    n_nonlands = deck["n_nonlands"]

    # Categorize cards
    catalog_cards = {c["name"]: c for c in gen.catalog["cards"]}
    creatures = []
    instants = []
    sorceries = []
    artifacts = []
    enchantments = []
    planeswalkers = []
    lands = []
    other = []

    for card in cards:
        info = catalog_cards.get(card)
        if not info:
            other.append(card)
            continue
        tc = set(info.get("type_categories", []))
        if info.get("is_land"):
            lands.append(card)
        elif "Creature" in tc:
            creatures.append(card)
        elif "Instant" in tc:
            instants.append(card)
        elif "Sorcery" in tc:
            sorceries.append(card)
        elif "Artifact" in tc:
            artifacts.append(card)
        elif "Enchantment" in tc:
            enchantments.append(card)
        elif "Planeswalker" in tc:
            planeswalkers.append(card)
        else:
            other.append(card)

    for label, group in [
        ("Creatures", creatures),
        ("Instants", instants),
        ("Sorceries", sorceries),
        ("Artifacts", artifacts),
        ("Enchantments", enchantments),
        ("Planeswalkers", planeswalkers),
        ("Lands", lands),
        ("Other", other),
    ]:
        if group:
            print(f"\n  {label} ({len(group)}):")
            for c in sorted(group):
                print(f"    {c}")

    print(f"\n  --- Total: {len(cards)} cards + commander = {len(cards)+1}")
    print(f"  --- Lands: {n_lands}, Non-lands: {n_nonlands}")
