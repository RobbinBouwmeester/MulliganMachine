"""Evaluation metrics for generated Commander decks."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mulligan_machine.data.scryfall import load_catalog
from mulligan_machine.data.tokenizer import DeckTokenizer

logger = logging.getLogger(__name__)


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets of card names."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def precision_at_k(predicted: list[str], reference: set[str], k: int) -> float:
    """Fraction of top-K predicted cards that appear in the reference set."""
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if c in reference)
    return hits / len(top_k)


def recall_at_k(predicted: list[str], reference: set[str], k: int) -> float:
    """Fraction of reference cards recovered in the top-K predictions."""
    if not reference:
        return 1.0
    top_k = set(predicted[:k])
    hits = len(top_k & reference)
    return hits / len(reference)


def color_identity_violation_rate(
    generated_deck: dict[str, Any],
    catalog: dict[str, Any],
) -> float:
    """
    Fraction of cards in the generated deck that violate the commander's color identity.

    Returns 0.0 if all cards are legal.
    """
    commander_name = generated_deck["commander"]
    cards = generated_deck["cards"]

    card_info = {c["name"]: c for c in catalog["cards"]}
    commander_ci = set(card_info.get(commander_name, {}).get("color_identity", []))

    violations = 0
    for card_name in cards:
        info = card_info.get(card_name)
        if info is None:
            continue
        card_ci = set(info.get("color_identity", []))
        if not card_ci.issubset(commander_ci):
            violations += 1

    return violations / len(cards) if cards else 0.0


def land_count(
    generated_deck: dict[str, Any],
    catalog: dict[str, Any],
) -> int:
    """Count the number of land cards in a generated deck."""
    card_info = {c["name"]: c for c in catalog["cards"]}
    return sum(
        1 for name in generated_deck["cards"] if card_info.get(name, {}).get("is_land", False)
    )


def mana_curve(
    generated_deck: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[int, int]:
    """
    Compute the mana curve (CMC distribution) of non-land cards.

    Returns: {cmc: count} dict, where CMC is clamped to [0, 7+].
    """
    card_info = {c["name"]: c for c in catalog["cards"]}
    curve: dict[int, int] = {i: 0 for i in range(8)}  # 0-6, 7+

    for name in generated_deck["cards"]:
        info = card_info.get(name)
        if info is None or info.get("is_land", False):
            continue
        cmc = int(info.get("cmc", 0))
        bucket = min(cmc, 7)
        curve[bucket] = curve.get(bucket, 0) + 1

    return curve


def singleton_violations(generated_deck: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    """
    Find cards that appear more than once (excluding basic lands).

    Returns list of card names that violate the singleton rule.
    """
    card_info = {c["name"]: c for c in catalog["cards"]}
    counts = Counter(generated_deck["cards"])
    violations = []
    for name, count in counts.items():
        if count > 1:
            info = card_info.get(name, {})
            if not info.get("is_basic_land", False):
                violations.append(name)
    return violations


def evaluate_deck(
    generated_deck: dict[str, Any],
    reference_deck: dict[str, Any] | None,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute all metrics for a single generated deck.

    Args:
        generated_deck: {"commander": str, "cards": list[str]}
        reference_deck: Optional real deck to compare against
        catalog: Card catalog from Scryfall

    Returns:
        Dict of metric name -> value
    """
    metrics: dict[str, Any] = {}

    gen_cards = generated_deck["cards"]
    metrics["n_cards"] = len(gen_cards)
    metrics["n_lands"] = land_count(generated_deck, catalog)
    metrics["n_nonlands"] = metrics["n_cards"] - metrics["n_lands"]
    metrics["ci_violation_rate"] = color_identity_violation_rate(generated_deck, catalog)
    metrics["singleton_violations"] = singleton_violations(generated_deck, catalog)
    metrics["n_singleton_violations"] = len(metrics["singleton_violations"])
    metrics["mana_curve"] = mana_curve(generated_deck, catalog)

    if reference_deck is not None:
        ref_cards = set(reference_deck["cards"])
        gen_set = set(gen_cards)
        metrics["jaccard"] = jaccard_similarity(gen_set, ref_cards)
        metrics["precision_at_10"] = precision_at_k(gen_cards, ref_cards, 10)
        metrics["precision_at_50"] = precision_at_k(gen_cards, ref_cards, 50)
        metrics["precision_at_99"] = precision_at_k(gen_cards, ref_cards, 99)
        metrics["recall_at_99"] = recall_at_k(gen_cards, ref_cards, 99)

    return metrics


def evaluate_batch(
    generated_decks: list[dict[str, Any]],
    reference_decks: list[dict[str, Any]] | None,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate a batch of generated decks and compute aggregate statistics.

    Args:
        generated_decks: List of generated decklists
        reference_decks: Optional list of reference decklists (same length, same order)
        catalog: Card catalog

    Returns:
        Dict with per-metric mean, std, min, max across all decks
    """
    all_metrics: list[dict[str, Any]] = []

    for i, gen in enumerate(generated_decks):
        ref = reference_decks[i] if reference_decks else None
        m = evaluate_deck(gen, ref, catalog)
        all_metrics.append(m)

    # Aggregate numeric metrics
    numeric_keys = [
        "n_cards",
        "n_lands",
        "n_nonlands",
        "ci_violation_rate",
        "n_singleton_violations",
    ]
    if reference_decks:
        numeric_keys += [
            "jaccard",
            "precision_at_10",
            "precision_at_50",
            "precision_at_99",
            "recall_at_99",
        ]

    summary: dict[str, Any] = {"n_decks": len(all_metrics)}
    for key in numeric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))

    # Aggregate mana curve
    if all_metrics:
        avg_curve = {i: 0.0 for i in range(8)}
        for m in all_metrics:
            for bucket, count in m.get("mana_curve", {}).items():
                avg_curve[int(bucket)] += count
        for bucket in avg_curve:
            avg_curve[bucket] /= len(all_metrics)
        summary["avg_mana_curve"] = avg_curve

    summary["individual"] = all_metrics
    return summary
