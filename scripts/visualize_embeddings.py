"""
Generate an interactive 2D embedding visualization of all MTG cards,
combining **functional ability features** (regex-extracted mechanics)
with **semantic embeddings** (all-mpnet-base-v2).

The regex features (~272 binary flags) capture explicit mechanics like
"draw", "destroy", "ramp", "create tokens", etc.  The semantic
embeddings (768-dim) capture nuanced ability meaning that patterns miss.
Both are L2-normalized and concatenated with a tunable weight, then
UMAP projects the combined vector to 2D using cosine distance.

Improvements over the basic version:
  - Card image shown when clicking a card (via Scryfall)
  - EDHREC popularity → marker size (bigger = more popular)
  - Toggle between coloring by card type vs. color identity
  - HDBSCAN cluster labels annotate mechanical regions on the map
  - Semantic embeddings cached to disk for fast re-runs
  - KNN via sklearn (fast, parallelised)
  - Full CLI via argparse

Usage:
  python scripts/visualize_embeddings.py [options]

  --catalog-dir PATH      Card catalog directory (default: data/catalog)
  --output PATH           Output HTML file (default: card_embeddings.html)
  --k-neighbors INT       Nearest neighbors to show (default: 20)
  --regex-weight FLOAT    0=semantic only, 1=regex only (default: 0.5)
  --umap-neighbors INT    UMAP n_neighbors (default: 30)
  --umap-min-dist FLOAT   UMAP min_dist (default: 0.15)
  --semantic-model STR    Sentence-transformer model (default: all-mpnet-base-v2)
  --min-cluster-size INT  HDBSCAN min cluster size (default: 100)
  --no-cache              Recompute semantic embeddings even if cached
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import hdbscan
import numpy as np
import plotly.graph_objects as go
import umap
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEMANTIC_MODEL = "all-mpnet-base-v2"

# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive ability taxonomy  (~272 features)
# Each entry: (feature_name, regex applied to lowercased oracle_text)
# ═══════════════════════════════════════════════════════════════════════════════

ABILITY_PATTERNS: list[tuple[str, str]] = [
    # ── Card Advantage ─────────────────────────────────────────────────────────
    ("draw_card", r"\bdraw[s]?\s+(?:a\s+)?card"),
    ("draw_multiple", r"\bdraw[s]?\s+(?:two|three|four|five|six|seven|\d{1,2})\s+card"),
    ("impulse_draw", r"\bexile\s+the\s+top\b.*\b(?:may\s+(?:play|cast)|until)\b"),
    (
        "tutor_to_hand",
        r"\bsearch\s+your\s+library\b.*\b(?:into\s+your\s+hand|put\b.*\bhand|reveal)\b",
    ),
    ("tutor_to_top", r"\bsearch\s+your\s+library\b.*\b(?:top\s+of|on\s+top)\b"),
    (
        "tutor_to_battlefield",
        r"\bsearch\s+your\s+library\b.*\b(?:onto\s+the\s+battlefield|to\s+the\s+battlefield|into\s+play)\b",
    ),
    ("tutor_any", r"\bsearch\s+your\s+library\b"),
    ("scry", r"\bscry\b"),
    ("surveil", r"\bsurveil\b"),
    ("look_at_top", r"\blook\s+at\s+the\s+top\b"),
    ("reveal_top", r"\breveal\s+(?:the\s+)?top\b"),
    ("card_filtering", r"\b(?:scry|surveil|look\s+at\s+the\s+top)\b"),
    # ── Removal ────────────────────────────────────────────────────────────────
    ("destroy_creature", r"\bdestroy\s+target\s+(?:creature|attacking|blocking)\b"),
    ("destroy_artifact", r"\bdestroy\s+target\s+artifact\b"),
    ("destroy_enchantment", r"\bdestroy\s+target\s+enchantment\b"),
    (
        "destroy_artifact_or_enchantment",
        r"\bdestroy\s+target\s+(?:artifact\s+or\s+enchantment|noncreature)\b",
    ),
    ("destroy_permanent", r"\bdestroy\s+target\s+(?:permanent|nonland\s+permanent)\b"),
    ("destroy_planeswalker", r"\bdestroy\s+target\s+planeswalker\b"),
    ("destroy_land", r"\bdestroy\s+target\s+land\b"),
    (
        "board_wipe_creatures",
        r"\bdestroy\s+all\s+creature|\ball\s+creatures\s+get\s+-|\bdamage\s+to\s+each\s+creature\b",
    ),
    ("board_wipe_all", r"\bdestroy\s+all\b|\bexile\s+all\b"),
    ("exile_creature", r"\bexile\s+target\s+(?:creature|attacking)\b"),
    ("exile_permanent", r"\bexile\s+target\s+(?:permanent|nonland)\b"),
    ("exile_spell", r"\bexile\s+target\s+(?:spell|instant|sorcery)\b"),
    ("bounce_creature", r"\breturn\s+target\s+creature\b.*\b(?:owner|hand)\b"),
    ("bounce_permanent", r"\breturn\s+target\s+(?:permanent|nonland)\b.*\b(?:owner|hand)\b"),
    ("bounce_all", r"\breturn\s+all\b.*\b(?:owner|hand)\b"),
    ("damage_targeted", r"\bdeal[s]?\s+\d+\s+damage\s+to\s+(?:target|any)\b"),
    ("damage_each_creature", r"\bdeal[s]?\s+\d+\s+damage\s+to\s+each\s+creature\b"),
    ("damage_each_opponent", r"\bdeal[s]?\s+\d+\s+damage\s+to\s+each\s+opponent\b"),
    ("damage_to_player", r"\bdeal[s]?\s+\d+\s+damage\s+to\s+(?:target\s+)?(?:player|opponent)\b"),
    ("minus_toughness", r"get[s]?\s+-\d+/-\d+"),
    ("fight_mechanic", r"\bfight[s]?\b"),
    ("bite_mechanic", r"\bdeals?\s+damage\s+equal\s+to\b.*\bpower\b"),
    (
        "forced_sacrifice",
        r"\b(?:each|target)\s+(?:player|opponent)\b.*\bsacrifice\b|\bsacrifice[s]?\s+(?:a|an)\b",
    ),
    ("tuck", r"\bput\b.*\bon\s+the\s+bottom\s+of\b.*\blibrary\b"),
    # ── Counterspells ──────────────────────────────────────────────────────────
    ("counter_spell", r"\bcounter\s+target\s+spell\b"),
    ("counter_creature_spell", r"\bcounter\s+target\s+creature\s+spell\b"),
    ("counter_noncreature", r"\bcounter\s+target\s+noncreature\b"),
    ("counter_activated", r"\bcounter\s+target\s+activated\b"),
    ("counter_conditional", r"\bcounter\b.*\bunless\b"),
    # ── Mana & Ramp ────────────────────────────────────────────────────────────
    ("add_mana", r"\badd\s+\{[WUBRGC]"),
    ("add_any_color", r"\badd\s+(?:one\s+mana\s+of\s+any|mana\s+of\s+any)"),
    ("add_multiple_mana", r"\badd\s+\{[WUBRGC]\}\{[WUBRGC]\}"),
    ("fetch_land_basic", r"\bsearch\b.*\bbasic\s+land\b.*\b(?:battlefield|onto|into play)\b"),
    ("fetch_land_any", r"\bsearch\b.*\bland\b.*\b(?:battlefield|onto|into play)\b"),
    ("extra_land_play", r"\bplay\s+an?\s+additional\s+land\b"),
    ("cost_reduction", r"\bcost[s]?\s+\{?\d+\}?\s+less\b|\breduce\s+the\s+cost\b"),
    ("treasure_creation", r"\btreasure\b"),
    ("mana_doubling", r"\bdouble\b.*\bmana\b|\badd\s+an\s+additional\b"),
    ("mana_rock", r"\{T\}.*\badd\b.*\{[WUBRGC]"),
    ("ritual_mana", r"\badd\s+\{[WUBRGC]\}\{[WUBRGC]\}\{[WUBRGC]\}"),
    # ── Tokens ─────────────────────────────────────────────────────────────────
    ("create_token", r"\bcreate[s]?\s+(?:a\s+|an?\s+|\d+\s+).*\btoken"),
    ("create_token_small", r"\bcreate[s]?\s+.*\b[01]/[01]\b.*\btoken"),
    ("create_token_medium", r"\bcreate[s]?\s+.*\b[2-3]/[2-3]\b.*\btoken"),
    ("create_token_big", r"\bcreate[s]?\s+.*\b[4-9]+/[4-9]+\b.*\btoken"),
    ("token_copy", r"\bcreate\s+a\s+token\b.*\bcopy\b"),
    ("populate", r"\bpopulate\b"),
    ("food_token", r"\bfood\b"),
    ("clue_token", r"\bclue\b"),
    ("blood_token", r"\bblood\s+token\b"),
    ("map_token", r"\bmap\s+token\b"),
    ("token_many", r"\bcreate\s+(?:two|three|four|five|six|\d{1,2})\b.*\btoken"),
    # ── +1/+1 Counters & Proliferate ──────────────────────────────────────────
    ("plus_counters", r"\+1/\+1\s+counter"),
    ("plus_counters_many", r"\b(?:two|three|four|five|\d{1,2})\s+\+1/\+1\s+counter"),
    ("minus_counters", r"-1/-1\s+counter"),
    ("other_counters", r"\b(?:charge|loyalty|lore|page|verse|time|fading|quest|level)\s+counter"),
    ("proliferate", r"\bproliferate\b"),
    ("counter_doubling", r"\bdouble\b.*\bcounter|\btwice\s+that\s+many\b.*\bcounter"),
    ("counter_moving", r"\bmove\b.*\bcounter"),
    ("counter_removal", r"\bremove\b.*\bcounter"),
    ("modular", r"\bmodular\b"),
    # ── Graveyard ──────────────────────────────────────────────────────────────
    (
        "reanimate_battlefield",
        r"\breturn[s]?\b.*\bfrom\b.*\bgraveyard\b.*\bto\s+the\s+battlefield\b",
    ),
    ("reanimate_to_hand", r"\breturn[s]?\b.*\bfrom\b.*\bgraveyard\b.*\bto\b.*\bhand\b"),
    ("self_mill", r"\bmill\b|\bput\s+the\s+top\b.*\binto\s+your\s+graveyard\b"),
    ("mill_opponent", r"\b(?:target\s+)?(?:player|opponent)\b.*\bmill\b"),
    (
        "exile_from_graveyard",
        r"\bexile\b.*\bfrom\b.*\bgraveyard\b|\bexile\b.*\b(?:card|cards)\b.*\bgraveyard\b",
    ),
    ("graveyard_count", r"\bfor\s+each\b.*\b(?:card|creature)\b.*\bin\b.*\bgraveyard\b"),
    ("graveyard_matters", r"\bgraveyard\b"),
    ("flashback", r"\bflashback\b"),
    ("unearth", r"\bunearth\b"),
    ("escape", r"\bescape\b"),
    ("dredge", r"\bdredge\b"),
    ("delve", r"\bdelve\b"),
    ("embalm", r"\bembalm\b"),
    ("eternalize", r"\beternalize\b"),
    ("encore", r"\bencore\b"),
    ("retrace", r"\bretrace\b"),
    ("disturb", r"\bdisturb\b"),
    # ── Life ───────────────────────────────────────────────────────────────────
    ("gain_life", r"\bgain[s]?\b.*\blife\b"),
    ("gain_life_payoff", r"\bwhenever\b.*\bgain\b.*\blife\b"),
    ("drain", r"\b(?:lose|loses)\b.*\blife\b.*\bgain\b|\bgain\b.*\blife\b.*\blose"),
    ("life_payment", r"\bpay\b.*\blife\b"),
    ("life_total_matters", r"\blife\s+total\b"),
    # ── Combat ─────────────────────────────────────────────────────────────────
    ("pump_temporary", r"\bget[s]?\s+\+\d+/\+\d+\s+until\b"),
    ("pump_permanent", r"\bget[s]?\s+\+\d+/\+\d+(?!\s+until)\b.*\bcounter\b"),
    ("anthem", r"\b(?:other\s+)?creatures\s+you\s+control\s+get\s+\+\d"),
    ("team_buff", r"\ball\s+creatures\b.*\bget\s+\+"),
    ("lord_effect", r"\b(?:other\s+)?\w+(?:\s+\w+)?\s+you\s+control\s+get\s+\+\b"),
    ("combat_trick", r"\buntil\s+end\s+of\s+turn\b"),
    ("extra_combat", r"\badditional\s+combat\s+phase\b"),
    ("goad", r"\bgoad\b"),
    ("must_attack", r"\bmust\s+attack\b|\battacks?\s+each\s+(?:combat|turn)\s+if\s+able\b"),
    ("cant_be_blocked", r"\bcan't\s+be\s+blocked\b"),
    ("power_matters", r"\bpower\s+(?:is|equal|greater)\b|\bwith\s+power\b"),
    ("toughness_matters", r"\btoughness\b.*\b(?:equal|greater|less)\b"),
    # ── Protection / Defense ───────────────────────────────────────────────────
    ("hexproof_grant", r"\bgain[s]?\s+hexproof\b|\bhas\s+hexproof\b|\bhexproof\b"),
    ("indestructible_grant", r"\bgain[s]?\s+indestructible\b|\bindestructible\b"),
    ("shroud", r"\bshroud\b"),
    ("protection_from", r"\bprotection\s+from\b"),
    ("ward", r"\bward\b"),
    ("regenerate", r"\bregenerate\b"),
    ("phase_out", r"\bphase[s]?\s+out\b"),
    ("totem_armor", r"\btotem\s+armor\b"),
    ("prevent_damage", r"\bprevent\b.*\bdamage\b"),
    ("shield_counter", r"\bshield\s+counter\b"),
    # ── Evasion Keywords ──────────────────────────────────────────────────────
    ("flying", r"\bflying\b"),
    ("trample", r"\btrample\b"),
    ("menace", r"\bmenace\b"),
    ("fear_intimidate", r"\b(?:fear|intimidate)\b"),
    ("shadow", r"\bshadow\b"),
    ("skulk", r"\bskulk\b"),
    ("reach", r"\breach\b"),
    # ── Static Keywords ───────────────────────────────────────────────────────
    ("haste", r"\bhaste\b"),
    ("first_strike", r"\bfirst\s+strike\b"),
    ("double_strike", r"\bdouble\s+strike\b"),
    ("deathtouch", r"\bdeathtouch\b"),
    ("lifelink", r"\blifelink\b"),
    ("vigilance", r"\bvigilance\b"),
    ("defender", r"\bdefender\b"),
    ("flash_keyword", r"\bflash\b"),
    ("infect_keyword", r"\binfect\b"),
    ("wither", r"\bwither\b"),
    ("toxic", r"\btoxic\b"),
    ("annihilator", r"\bannihilator\b"),
    ("affinity", r"\baffinity\b"),
    # ── Trigger Types ─────────────────────────────────────────────────────────
    ("etb_trigger", r"\benter[s]?\s+the\s+battlefield\b"),
    ("ltb_trigger", r"\bleave[s]?\s+the\s+battlefield\b"),
    ("death_trigger", r"\bwhen\b.*\bdie[s]?\b|\bwhen\b.*\bput\s+into\b.*\bgraveyard\b"),
    ("attack_trigger", r"\bwhenever\b.*\battack[s]?\b"),
    ("damage_trigger", r"\bwhenever\b.*\bdeal[s]?\b.*\bdamage\b"),
    ("upkeep_trigger", r"\bat\s+the\s+beginning\s+of\b.*\bupkeep\b"),
    ("end_step_trigger", r"\bat\s+the\s+beginning\s+of\b.*\bend\s+step\b"),
    ("landfall", r"\blandfall\b|\bwhenever\s+a\s+land\s+enters\b"),
    ("cast_trigger", r"\bwhenever\s+you\s+cast\b"),
    ("constellation", r"\bwhenever\s+an?\s+enchantment\s+enters\b|\bconstellation\b"),
    ("magecraft", r"\bmagecraft\b|\bwhenever\s+you\s+cast\s+or\s+copy\b"),
    ("whenever_gains_life", r"\bwhenever\b.*\bgain\b.*\blife\b"),
    ("whenever_token", r"\bwhenever\b.*\btoken\b"),
    # ── Equipment / Auras / Vehicles ──────────────────────────────────────────
    ("equip", r"\bequip\b"),
    ("aura_enchant", r"\benchant\s+(?:creature|permanent|player|land)\b"),
    ("living_weapon", r"\bliving\s+weapon\b"),
    ("reconfigure", r"\breconfigure\b"),
    ("bestow", r"\bbestow\b"),
    ("crew_vehicle", r"\bcrew\b"),
    ("for_mirrodin", r"\bfor\s+mirrodin\b"),
    # ── Copy Effects ──────────────────────────────────────────────────────────
    ("copy_spell", r"\bcopy\b.*\b(?:instant|sorcery|spell)\b"),
    ("copy_creature", r"\bcopy\b.*\bcreature\b"),
    ("copy_permanent", r"\bcopy\b.*\bpermanent\b"),
    ("clone_effect", r"\bcopy\s+of\b|\benters\b.*\bas?\s+a?\s*copy\b"),
    # ── Sacrifice / Aristocrats ───────────────────────────────────────────────
    (
        "sacrifice_outlet",
        r"\bsacrifice\s+(?:a|an?other)\s+(?:creature|permanent|artifact|enchantment)\b",
    ),
    ("sacrifice_self", r"\bsacrifice\s+~\b|\bsacrifice\s+this\b|\bsacrifice\s+it\b"),
    ("sacrifice_payoff", r"\bwhenever\b.*\bsacrifice\b"),
    ("blood_artist", r"\bwhenever\b.*\bdie[s]?\b.*\b(?:lose|deal|gain)\b"),
    ("aristocrat", r"\bwhenever\b.*\b(?:creature|permanent)\b.*\bdie[s]?\b"),
    # ── Discard ───────────────────────────────────────────────────────────────
    ("discard_opponent", r"\b(?:target\s+)?(?:player|opponent)\b.*\bdiscard\b"),
    ("discard_self", r"\bdiscard\s+(?:a|your|that|this)\b"),
    ("madness", r"\bmadness\b"),
    ("discard_payoff", r"\bwhenever\b.*\bdiscard\b"),
    ("wheel", r"\beach\s+player\b.*\b(?:draw|discard)\b.*\bhand\b"),
    # ── Extra Turns / Stax / Control ──────────────────────────────────────────
    ("extra_turn", r"\bextra\s+turn\b|\badditional\s+turn\b"),
    ("tax_effect", r"\bcost[s]?\s+\{?\d+\}?\s+more\b|\badditional\s+\{?\d+\}\b"),
    ("cant_cast", r"\bcan't\s+cast\b"),
    ("cant_attack", r"\bcan't\s+attack\b"),
    ("cant_block", r"\bcan't\s+block\b"),
    ("tap_down", r"\btap\s+target\b|\bdoesn't\s+untap\b"),
    (
        "prison_effect",
        r"\bcan't\s+attack\b.*\bor\s+block\b|\bcan't\s+(?:attack|block|be\s+activated)\b",
    ),
    ("rule_setting", r"\b(?:each|all)\s+(?:player|opponent)s?\s+can't\b|\bplayers?\s+can't\b"),
    ("end_the_turn", r"\bend\s+the\s+turn\b"),
    # ── Modality / X-spells ───────────────────────────────────────────────────
    ("choose_modes", r"\bchoose\s+(?:one|two|three|any\s+number|up\s+to)\b"),
    ("x_spell", r"\{X\}"),
    ("kicker", r"\bkicker\b"),
    ("overload", r"\boverload\b"),
    ("entwine", r"\bentwine\b"),
    ("escalate", r"\bescalate\b"),
    ("adventure", r"\badventure\b"),
    ("cycling", r"\bcycling\b"),
    ("channel", r"\bchannel\b"),
    ("ninjutsu", r"\bninjutsu\b"),
    ("cascade", r"\bcascade\b"),
    ("convoke", r"\bconvoke\b"),
    ("emerge", r"\bemerge\b"),
    ("evoke", r"\bevoke\b"),
    ("foretell", r"\bforetell\b"),
    ("suspend", r"\bsuspend\b"),
    ("miracle", r"\bmiracle\b"),
    # ── Blink / Flicker ───────────────────────────────────────────────────────
    ("blink", r"\bexile\b.*\bthen\s+return\b|\bexile\b.*\breturn[s]?\b.*\bbattlefield\b"),
    # ── Spell-matters / Storm ─────────────────────────────────────────────────
    ("storm", r"\bstorm\b"),
    ("spell_copying", r"\bcopy\b.*\bspell\b"),
    ("prowess", r"\bprowess\b"),
    ("spell_matters", r"\binstant[s]?\s+(?:and|or)\s+sorcer|\bwhenever\s+you\s+cast\s+(?:a|an)\b"),
    # ── Tap / Untap ───────────────────────────────────────────────────────────
    ("tap_ability", r"\{T\}"),
    ("untap_creature", r"\buntap\s+(?:target\s+)?(?:creature|permanent)\b"),
    ("untap_land", r"\buntap\s+(?:target\s+)?land\b"),
    ("untap_all", r"\buntap\s+(?:all|each)\b"),
    ("inspired", r"\bwhenever\b.*\bbecomes\s+untapped\b"),
    # ── Tribal / Type-matters ─────────────────────────────────────────────────
    ("tribal_lord", r"\b(?:other\s+)?\w+s?\s+you\s+control\s+(?:get|have|gain)\b"),
    ("changeling", r"\bchangeling\b"),
    ("type_counting", r"\bfor\s+each\s+\w+\s+you\s+control\b"),
    ("kindred", r"\bchoose\s+a\s+creature\s+type\b"),
    # ── Loot / Rummage / Cycling ──────────────────────────────────────────────
    ("loot", r"\bdraw\b.*\bthen\s+discard\b"),
    ("rummage", r"\bdiscard\b.*\bthen\s+draw\b"),
    # ── Transform / Morph ─────────────────────────────────────────────────────
    ("transform", r"\btransform\b"),
    ("morph_manifest", r"\bmorph\b|\bmanifest\b|\bmegamorph\b"),
    ("disguise", r"\bdisguise\b"),
    # ── Win conditions / Special ──────────────────────────────────────────────
    ("alt_wincon", r"\byou\s+win\s+the\s+game\b"),
    ("alt_loss", r"\byou\s+lose\s+the\s+game\b"),
    ("damage_doubling", r"\bdouble\b.*\bdamage\b|\bdeals?\s+(?:twice|double)\b"),
    ("poison_counters", r"\bpoison\s+counter\b"),
    ("commander_matters", r"\bcommander\b"),
    ("experience_counter", r"\bexperience\s+counter\b"),
    ("partner_keyword", r"\bpartner\b"),
    # ── Misc named mechanics ──────────────────────────────────────────────────
    ("amass", r"\bamass\b"),
    ("adapt", r"\badapt\b"),
    ("afflict", r"\bafflict\b"),
    ("backup", r"\bbackup\b"),
    ("bargain", r"\bbargain\b"),
    ("battle_cry", r"\bbattle\s+cry\b"),
    ("blitz", r"\bblitz\b"),
    ("boast", r"\bboast\b"),
    ("bushido", r"\bbushido\b"),
    ("cipher", r"\bcipher\b"),
    ("cleave_mechanic", r"\bcleave\b"),
    ("collect_evidence", r"\bcollect\s+evidence\b"),
    ("connive", r"\bconnive\b"),
    ("conspire", r"\bconspire\b"),
    ("craft", r"\bcraft\b"),
    ("dash", r"\bdash\b"),
    ("decayed", r"\bdecayed\b"),
    ("devour", r"\bdevour\b"),
    ("domain", r"\bdomain\b"),
    ("exploit", r"\bexploit\b"),
    ("exalted", r"\bexalted\b"),
    ("extort", r"\bextort\b"),
    ("fabricate", r"\bfabricate\b"),
    ("ferocious", r"\bferocious\b"),
    ("heroic", r"\bheroic\b|\bwhenever\s+you\s+cast\s+a\s+spell\s+that\s+targets\b"),
    ("hideaway", r"\bhideaway\b"),
    ("investigate", r"\binvestigate\b"),
    ("learn", r"\blearn\b"),
    ("melee", r"\bmelee\b"),
    ("mentor", r"\bmentor\b"),
    ("myriad", r"\bmyriad\b"),
    ("outlast", r"\boutlast\b"),
    ("overrun_effect", r"\bcreatures\s+you\s+control\s+get\s+\+.*\band\s+gain\s+trample\b"),
    ("persist", r"\bpersist\b"),
    ("plot", r"\bplot\b"),
    ("raid", r"\braid\b"),
    ("rally", r"\brally\b"),
    ("rebound", r"\brebound\b"),
    ("renown", r"\brenown\b"),
    ("replicate", r"\breplicate\b"),
    ("riot", r"\briot\b"),
    ("role_token", r"\brole\b.*\btoken\b"),
    ("soulbond", r"\bsoulbond\b"),
    ("spectacle", r"\bspectacle\b"),
    ("splice", r"\bsplice\b"),
    ("squad", r"\bsquad\b"),
    ("threshold", r"\bthreshold\b"),
    ("training", r"\btraining\b"),
    ("undying", r"\bundying\b"),
    ("unleash", r"\bunleash\b"),
]

# Card types used as extra features
TYPE_CATEGORIES = [
    "Creature",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Planeswalker",
    "Land",
    "Battle",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_ability_features(cards: list[dict]) -> tuple[list[dict], np.ndarray, list[str]]:
    """Extract ability feature vectors from card oracle text.

    Returns (filtered_cards, features, feature_names).
    ``features`` is ``(N, n_features)`` float32.
    """
    logger.info("Extracting ability features from %d cards...", len(cards))

    compiled = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in ABILITY_PATTERNS]
    n_ability = len(compiled)
    n_type = len(TYPE_CATEGORIES)
    n_features = n_ability + n_type + 5  # +5 WUBRG

    feature_names = [name for name, _ in ABILITY_PATTERNS]
    feature_names += [f"type_{t.lower()}" for t in TYPE_CATEGORIES]
    feature_names += ["color_W", "color_U", "color_B", "color_R", "color_G"]

    filtered_cards: list[dict] = []
    feature_rows: list[np.ndarray] = []

    for card in cards:
        oracle = (card.get("oracle_text") or "").lower()
        type_cats = set(card.get("type_categories", []))

        vec = np.zeros(n_features, dtype=np.float32)

        for i, (_, regex) in enumerate(compiled):
            if regex.search(oracle):
                vec[i] = 1.0

        for j, tc in enumerate(TYPE_CATEGORIES):
            if tc in type_cats:
                vec[n_ability + j] = 1.0

        ci = set(card.get("color_identity", []))
        for k, color in enumerate("WUBRG"):
            if color in ci:
                vec[n_ability + n_type + k] = 1.0

        filtered_cards.append(card)
        feature_rows.append(vec)

    features = np.array(feature_rows, dtype=np.float32)
    avg_abilities = features[:, :n_ability].sum(axis=1).mean()
    logger.info(
        "Extracted %d features for %d cards (avg %.1f ability matches/card)",
        n_features,
        len(filtered_cards),
        avg_abilities,
    )
    return filtered_cards, features, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic embeddings  (cached to disk)
# ═══════════════════════════════════════════════════════════════════════════════


def _embedding_cache_path(catalog_dir: Path) -> Path:
    model_slug = SEMANTIC_MODEL.replace("/", "_")
    return catalog_dir / f".semantic_cache_{model_slug}.npz"


def compute_semantic_embeddings(
    cards: list[dict],
    catalog_dir: Path,
    batch_size: int = 256,
    no_cache: bool = False,
) -> np.ndarray:
    """Encode oracle text + type line with a sentence-transformer model.

    Results are cached to ``<catalog_dir>/.semantic_cache_<model>.npz``
    so subsequent runs skip the expensive encoding step.
    """
    cache_path = _embedding_cache_path(catalog_dir)
    card_hash = hashlib.md5(
        json.dumps([c["name"] for c in cards], sort_keys=True).encode()
    ).hexdigest()

    if not no_cache and cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            if str(data["card_hash"]) == card_hash:
                emb = data["embeddings"]
                logger.info(
                    "Loaded cached semantic embeddings from %s  shape=%s",
                    cache_path,
                    emb.shape,
                )
                return emb
            logger.info("Cache stale (card list changed), recomputing...")
        except Exception as exc:
            logger.warning("Cache read failed (%s), recomputing...", exc)

    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer '%s'...", SEMANTIC_MODEL)
    model = SentenceTransformer(SEMANTIC_MODEL)

    texts: list[str] = []
    for card in cards:
        type_line = card.get("type_line", "") or ""
        oracle = card.get("oracle_text", "") or ""
        texts.append(f"{type_line}. {oracle}" if oracle else type_line)

    logger.info("Encoding %d cards with '%s'...", len(texts), SEMANTIC_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    logger.info("Semantic embeddings shape: %s", embeddings.shape)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        card_hash=np.array(card_hash),
    )
    logger.info("Cached embeddings to %s", cache_path)
    return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Feature combination
# ═══════════════════════════════════════════════════════════════════════════════


def combine_features(
    regex_features: np.ndarray,
    semantic_embeddings: np.ndarray,
    regex_weight: float = 0.5,
) -> np.ndarray:
    """L2-normalise each block, apply weights, and concatenate."""
    norms = np.linalg.norm(regex_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    regex_normed = regex_features / norms

    sem_weight = 1.0 - regex_weight
    combined = np.concatenate(
        [regex_normed * regex_weight, semantic_embeddings * sem_weight],
        axis=1,
    )
    logger.info(
        "Combined features: %d regex x %.1f + %d semantic x %.1f -> %d dims",
        regex_features.shape[1],
        regex_weight,
        semantic_embeddings.shape[1],
        sem_weight,
        combined.shape[1],
    )
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_display_data(cards: list[dict]):
    """Build display arrays for the cards (extended with Scryfall + EDHREC)."""
    names: list[str] = []
    types: list[str] = []
    mana_costs: list[str] = []
    oracle_texts: list[str] = []
    primary_types: list[str] = []
    color_ids: list[str] = []
    scryfall_ids: list[str] = []
    edhrec_ranks: list[int | None] = []

    for card in cards:
        names.append(card["name"])
        types.append(card.get("type_line", ""))
        mana_costs.append(card.get("mana_cost", ""))
        oracle_text = card.get("oracle_text", "") or ""
        oracle_texts.append(oracle_text[:200] + ("..." if len(oracle_text) > 200 else ""))

        tc = card.get("type_categories", [])
        for ptype in TYPE_CATEGORIES:
            if ptype in tc:
                primary_types.append(ptype)
                break
        else:
            primary_types.append("Other")

        ci = card.get("color_identity", [])
        color_ids.append("".join(sorted(ci)) if ci else "C")

        scryfall_ids.append(card.get("scryfall_id", ""))
        edhrec_ranks.append(card.get("edhrec_rank"))

    return (
        names,
        types,
        mana_costs,
        oracle_texts,
        primary_types,
        color_ids,
        scryfall_ids,
        edhrec_ranks,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UMAP
# ═══════════════════════════════════════════════════════════════════════════════


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.15,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce combined features to 2D with UMAP."""
    logger.info(
        "Running UMAP (n=%d, n_neighbors=%d, min_dist=%.2f, metric=%s)...",
        len(features),
        n_neighbors,
        min_dist,
        metric,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(features)
    logger.info("UMAP complete.  Shape: %s", coords.shape)
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# KNN  (sklearn — fast, parallelised)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_neighbor_indices(features: np.ndarray, k: int = 20) -> np.ndarray:
    """K-nearest neighbors via sklearn cosine metric."""
    logger.info("Computing %d nearest neighbours (sklearn, cosine, n_jobs=-1)...", k)
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(features)
    _, indices = nn.kneighbors(features)
    # Column 0 is the card itself — drop it
    neighbors = indices[:, 1:].astype(np.int32)
    logger.info("Neighbour computation complete.")
    return neighbors


# ═══════════════════════════════════════════════════════════════════════════════
# HDBSCAN cluster labels
# ═══════════════════════════════════════════════════════════════════════════════


def compute_clusters(
    coords: np.ndarray,
    regex_features: np.ndarray,
    feature_names: list[str],
    min_cluster_size: int = 100,
) -> list[tuple[int, np.ndarray, str, int]]:
    """Run HDBSCAN on 2D UMAP coords and label each cluster by top abilities.

    Returns a list of (cluster_id, centroid_xy, label_text, n_cards).
    """
    logger.info("Running HDBSCAN (min_cluster_size=%d)...", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords)

    n_ability = len(ABILITY_PATTERNS)
    cluster_info: list[tuple[int, np.ndarray, str, int]] = []

    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        centroid = coords[mask].mean(axis=0)
        # Mean ability frequency in this cluster
        ability_freq = regex_features[mask, :n_ability].mean(axis=0)
        top_ids = np.argsort(-ability_freq)[:3]
        top_names = [feature_names[i] for i in top_ids if ability_freq[i] > 0.05]
        label = (
            " / ".join(n.replace("_", " ") for n in top_names) if top_names else f"Cluster {cid}"
        )
        cluster_info.append((cid, centroid, label, int(mask.sum())))

    n_clustered = sum(1 for la in labels if la >= 0)
    logger.info(
        "Found %d clusters  (%d/%d cards clustered, %d noise)",
        len(cluster_info),
        n_clustered,
        len(labels),
        len(labels) - n_clustered,
    )
    return cluster_info


# ═══════════════════════════════════════════════════════════════════════════════
# EDHREC rank → marker size
# ═══════════════════════════════════════════════════════════════════════════════


def compute_marker_sizes(
    edhrec_ranks: list[int | None],
    min_size: float = 2.0,
    max_size: float = 10.0,
) -> np.ndarray:
    """Map EDHREC rank to marker size (lower rank = bigger dot)."""
    n = len(edhrec_ranks)
    sizes = np.full(n, min_size, dtype=np.float32)
    valid_idx: list[int] = []
    valid_ranks: list[float] = []
    for i, r in enumerate(edhrec_ranks):
        if r is not None and r > 0:
            valid_idx.append(i)
            valid_ranks.append(float(r))
    if valid_ranks:
        ranks = np.array(valid_ranks, dtype=np.float32)
        max_rank = ranks.max()
        scaled = 1.0 - np.log1p(ranks) / np.log1p(max_rank)
        for j, idx in enumerate(valid_idx):
            sizes[idx] = min_size + (max_size - min_size) * scaled[j]
    return sizes


# ═══════════════════════════════════════════════════════════════════════════════
# HTML builder
# ═══════════════════════════════════════════════════════════════════════════════

TYPE_COLORS = {
    "Creature": "#4CAF50",
    "Instant": "#2196F3",
    "Sorcery": "#F44336",
    "Artifact": "#9E9E9E",
    "Enchantment": "#9C27B0",
    "Planeswalker": "#FF9800",
    "Land": "#795548",
    "Battle": "#00BCD4",
    "Other": "#607D8B",
}

IDENTITY_COLOR_MAP = {
    "W": "#F8F6D8",
    "U": "#0E68AB",
    "B": "#6B3FA0",
    "R": "#D3202A",
    "G": "#00733E",
}


def _identity_color(ci: str) -> str:
    if not ci or ci == "C":
        return "#CAC5C0"
    if len(ci) == 1:
        return IDENTITY_COLOR_MAP.get(ci, "#607D8B")
    return "#DAA520"  # gold for multicolor


def build_html(
    coords: np.ndarray,
    names: list[str],
    types: list[str],
    mana_costs: list[str],
    oracle_texts: list[str],
    primary_types: list[str],
    color_ids: list[str],
    neighbors: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    scryfall_ids: list[str],
    edhrec_ranks: list[int | None],
    marker_sizes: np.ndarray,
    cluster_info: list[tuple[int, np.ndarray, str, int]],
    k_neighbors: int,
):
    """Build the interactive Plotly HTML visualization."""
    logger.info("Building visualization...")

    # ── Hover text ────────────────────────────────────────────────────────────
    hover_texts: list[str] = []
    for i in range(len(names)):
        rank_str = f"  EDHREC #{edhrec_ranks[i]}" if edhrec_ranks[i] else ""
        hover_texts.append(
            f"<b>{names[i]}</b><br>"
            f"{mana_costs[i]}  \u2022  {types[i]}{rank_str}<br>"
            f"Colors: {color_ids[i]}<br>"
            f"<br>{oracle_texts[i]}"
        )

    fig = go.Figure()

    # ── Per-type traces with per-point sizes ──────────────────────────────────
    trace_meta: dict[int, dict] = {}  # trace_index -> {type_color, identity_colors}
    trace_idx = 0

    for ptype in TYPE_CATEGORIES + ["Other"]:
        mask = [i for i, t in enumerate(primary_types) if t == ptype]
        if not mask:
            continue
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(
                    size=[float(marker_sizes[i]) for i in mask],
                    color=TYPE_COLORS[ptype],
                    opacity=0.7,
                ),
                name=ptype,
                text=[names[i] for i in mask],
                customdata=[i for i in mask],
                hovertext=[hover_texts[i] for i in mask],
                hoverinfo="text",
                hoverlabel=dict(bgcolor="white", font_size=11),
            )
        )
        trace_meta[trace_idx] = {
            "type_color": TYPE_COLORS[ptype],
            "identity_colors": [_identity_color(color_ids[i]) for i in mask],
        }
        trace_idx += 1

    n_data_traces = trace_idx

    # ── Neighbour highlight trace ─────────────────────────────────────────────
    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="red",
                symbol="circle-open",
                line=dict(width=2, color="red"),
            ),
            name="Nearest neighbours",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # ── Selected card trace ───────────────────────────────────────────────────
    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=14,
                color="gold",
                symbol="star",
                line=dict(width=2, color="black"),
            ),
            name="Selected card",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # ── HDBSCAN cluster annotations ───────────────────────────────────────────
    for _cid, centroid, label, count in cluster_info:
        font_size = max(9, min(14, 8 + count // 500))
        fig.add_annotation(
            x=float(centroid[0]),
            y=float(centroid[1]),
            text=label,
            showarrow=False,
            font=dict(size=font_size, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=3,
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="MTG Card Ability Map \u2014 Combined Regex + Semantic (cosine UMAP)",
            font=dict(size=18),
        ),
        width=1400,
        height=900,
        template="plotly_dark",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.7)",
            font=dict(size=12),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    # ── Pre-compute ability lists for JS ──────────────────────────────────────
    n_ability = len(ABILITY_PATTERNS)
    card_abilities: list[list[str]] = []
    for i in range(len(names)):
        card_abilities.append([feature_names[j] for j in range(n_ability) if features[i, j] > 0.5])

    neighbor_shared: list[list[list[str]]] = []
    for i in range(len(names)):
        shared_per_nbr: list[list[str]] = []
        for ni in neighbors[i]:
            shared = [
                feature_names[j]
                for j in range(n_ability)
                if features[i, j] > 0.5 and features[ni, j] > 0.5
            ]
            shared_per_nbr.append(shared)
        neighbor_shared.append(shared_per_nbr)

    all_x = coords[:, 0].tolist()
    all_y = coords[:, 1].tolist()

    # ── Serialise everything to HTML ──────────────────────────────────────────
    html_str = fig.to_html(include_plotlyjs=True, full_html=True, div_id="main-plot")

    custom_js = f"""
<style>
    #search-container {{
        position: fixed; top: 15px; left: 20px; z-index: 10000;
        background: rgba(30,30,30,0.95); padding: 12px 16px;
        border-radius: 8px; border: 1px solid #555;
        font-family: 'Segoe UI', Arial, sans-serif;
    }}
    #card-search {{
        width: 320px; padding: 8px 12px; font-size: 14px;
        border: 1px solid #666; border-radius: 4px;
        background: #222; color: #eee; outline: none;
    }}
    #card-search:focus {{ border-color: #4CAF50; }}
    #search-results {{
        max-height: 250px; overflow-y: auto; margin-top: 5px;
    }}
    .search-result {{
        padding: 4px 8px; cursor: pointer; color: #ccc;
        font-size: 13px; border-radius: 3px;
    }}
    .search-result:hover {{ background: #444; color: #fff; }}
    #color-toggle {{
        position: fixed; top: 15px; right: 20px; z-index: 10000;
        background: rgba(30,30,30,0.95); padding: 8px 12px;
        border-radius: 8px; border: 1px solid #555;
        font-family: 'Segoe UI', Arial, sans-serif;
        display: flex; gap: 6px;
    }}
    .color-btn {{
        padding: 5px 12px; cursor: pointer; border: 1px solid #555;
        border-radius: 4px; font-size: 12px; color: #ccc; background: #333;
        transition: all 0.2s;
    }}
    .color-btn:hover {{ background: #444; }}
    .color-btn.active {{ background: #4CAF50; color: #fff; border-color: #4CAF50; }}
    #neighbor-info {{
        position: fixed; bottom: 20px; left: 20px; z-index: 10000;
        background: rgba(30,30,30,0.95); padding: 12px 16px;
        border-radius: 8px; border: 1px solid #555;
        font-family: 'Segoe UI', Arial, sans-serif; color: #eee;
        max-height: 500px; max-width: 600px; overflow-y: auto;
        display: none;
    }}
    #neighbor-info h3 {{ margin: 0 0 6px 0; color: gold; font-size: 15px; }}
    .ability-tag {{
        display: inline-block; padding: 1px 6px; margin: 1px 2px;
        border-radius: 3px; font-size: 10px;
        background: #335; color: #8bf; border: 1px solid #447;
    }}
    .ability-tag.shared {{
        background: #353; color: #8f8; border-color: #474;
    }}
    .neighbor-item {{
        padding: 4px 0; font-size: 12px; color: #ccc;
        cursor: pointer; border-bottom: 1px solid #333;
    }}
    .neighbor-item:hover {{ color: #fff; }}
    .neighbor-rank {{ color: #F44336; font-weight: bold; margin-right: 6px; }}
    .neighbor-type {{ color: #888; font-size: 11px; }}
    .shared-label {{ font-size: 10px; color: #6a6; margin-left: 4px; }}
    .card-header {{ display: flex; gap: 12px; align-items: flex-start; }}
    .card-img {{
        width: 140px; border-radius: 8px; flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }}
</style>

<div id="search-container">
    <input type="text" id="card-search" placeholder="Search for a card..." autocomplete="off">
    <div id="search-results"></div>
</div>

<div id="color-toggle">
    <button class="color-btn active" id="btn-type" onclick="window._toggleColor()">Color by Type</button>
    <button class="color-btn" id="btn-identity" onclick="window._toggleColor()">Color by Identity</button>
</div>

<div id="neighbor-info"></div>

<script>
(function() {{
    const allX = {json.dumps(all_x)};
    const allY = {json.dumps(all_y)};
    const allNames = {json.dumps(list(names))};
    const allTypes = {json.dumps(list(types))};
    const allMana = {json.dumps(list(mana_costs))};
    const allOracle = {json.dumps(list(oracle_texts))};
    const allScryfallIds = {json.dumps(list(scryfall_ids))};
    const allRanks = {json.dumps([r if r else None for r in edhrec_ranks])};
    const neighbors = {json.dumps(neighbors.tolist())};
    const cardAbilities = {json.dumps(card_abilities)};
    const neighborShared = {json.dumps(neighbor_shared)};
    const K = {k_neighbors};
    const nDataTraces = {n_data_traces};
    const traceMeta = {json.dumps(trace_meta)};

    const plotDiv = document.getElementById('main-plot');
    const searchInput = document.getElementById('card-search');
    const searchResultsEl = document.getElementById('search-results');
    const neighborInfo = document.getElementById('neighbor-info');

    let colorMode = 'type';

    function scryfallImg(id) {{
        if (!id) return '';
        return 'https://cards.scryfall.io/normal/front/' + id[0] + '/' + id[1] + '/' + id + '.jpg';
    }}

    function fmt(name) {{ return name.replace(/_/g, ' '); }}

    /* ── Color Toggle ─────────────────────────────────────────────── */
    function toggleColorMode() {{
        colorMode = colorMode === 'type' ? 'identity' : 'type';
        document.getElementById('btn-type').classList.toggle('active', colorMode === 'type');
        document.getElementById('btn-identity').classList.toggle('active', colorMode === 'identity');
        for (let t = 0; t < nDataTraces; t++) {{
            const meta = traceMeta[String(t)];
            if (colorMode === 'identity') {{
                Plotly.restyle(plotDiv, {{'marker.color': [meta.identity_colors]}}, [t]);
            }} else {{
                Plotly.restyle(plotDiv, {{'marker.color': meta.type_color}}, [t]);
            }}
        }}
    }}
    window._toggleColor = toggleColorMode;

    /* ── Select Card ──────────────────────────────────────────────── */
    function selectCard(idx) {{
        if (idx < 0 || idx >= allNames.length) return;
        const nbrs = neighbors[idx];
        const nTraces = plotDiv.data.length;

        Plotly.restyle(plotDiv, {{
            x: [nbrs.map(i => allX[i])],
            y: [nbrs.map(i => allY[i])],
            text: [nbrs.map(i => allNames[i])],
            hoverinfo: 'text',
        }}, [nTraces - 2]);
        Plotly.restyle(plotDiv, {{
            x: [[allX[idx]]], y: [[allY[idx]]],
            text: [[allNames[idx]]], hoverinfo: 'text',
        }}, [nTraces - 1]);

        const imgUrl = scryfallImg(allScryfallIds[idx]);
        const myAbils = cardAbilities[idx];
        let tags = myAbils.map(a => '<span class="ability-tag">' + fmt(a) + '</span>').join('');

        let html = '<div class="card-header">';
        if (imgUrl) {{
            html += '<img class="card-img" src="' + imgUrl + '" onerror="this.style.display=\\'none\\'">';
        }}
        html += '<div>';
        html += '<h3>\\u2B50 ' + allNames[idx] + '</h3>';
        html += '<div style="color:#aaa;font-size:11px;margin-bottom:4px">' + allMana[idx] + ' \\u00B7 ' + allTypes[idx] + '</div>';
        if (allRanks[idx]) {{
            html += '<div style="color:#f80;font-size:10px;margin-bottom:4px">EDHREC Rank: #' + allRanks[idx] + '</div>';
        }}
        html += '<div style="margin-bottom:6px">' + tags + '</div>';
        html += '</div></div>';

        html += '<div style="border-top:1px solid #444;padding-top:6px;margin-top:8px;font-size:12px;color:#F44336;margin-bottom:4px">' + K + ' Most Similar Cards:</div>';

        nbrs.forEach(function(ni, rank) {{
            const shared = neighborShared[idx][rank];
            const sTags = shared.map(a => '<span class="ability-tag shared">' + fmt(a) + '</span>').join('');
            html += '<div class="neighbor-item" onclick="window._sel(' + ni + ')">';
            html += '<span class="neighbor-rank">' + (rank+1) + '.</span> ' + allNames[ni] + ' ';
            html += '<span class="neighbor-type">' + allTypes[ni] + '</span>';
            html += '<span class="shared-label">(' + shared.length + ' shared)</span>';
            html += '<div style="margin:2px 0 2px 20px">' + sTags + '</div></div>';
        }});

        neighborInfo.innerHTML = html;
        neighborInfo.style.display = 'block';
    }}
    window._sel = selectCard;

    /* ── Click handler ────────────────────────────────────────────── */
    plotDiv.on('plotly_click', function(data) {{
        if (data.points && data.points.length > 0) {{
            const g = data.points[0].customdata;
            if (g !== undefined && g !== null) selectCard(g);
        }}
    }});

    /* ── Search ───────────────────────────────────────────────────── */
    let debounce = null;
    searchInput.addEventListener('input', function() {{
        const q = this.value.toLowerCase().trim();
        clearTimeout(debounce);
        if (q.length < 2) {{ searchResultsEl.innerHTML = ''; return; }}
        debounce = setTimeout(function() {{
            const m = [];
            for (let i = 0; i < allNames.length && m.length < 15; i++)
                if (allNames[i].toLowerCase().includes(q)) m.push(i);
            let h = '';
            m.forEach(function(i) {{
                h += '<div class="search-result" onclick="window._sel(' + i + ');document.getElementById(\\'search-results\\').innerHTML=\\'\\';">';
                h += allNames[i] + ' <span style="color:#888;font-size:11px">' + allTypes[i] + '</span></div>';
            }});
            searchResultsEl.innerHTML = h || '<div style="color:#888;padding:4px 8px;font-size:13px">No matches</div>';
        }}, 150);
    }});
    searchInput.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') {{ searchResultsEl.innerHTML = ''; this.blur(); }}
    }});
}})();
</script>
"""

    html_str = html_str.replace("</body>", custom_js + "\n</body>")
    return html_str


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate interactive MTG card embedding visualization",
    )
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Card catalog directory  (default: data/catalog)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("card_embeddings.html"),
        help="Output HTML file  (default: card_embeddings.html)",
    )
    p.add_argument(
        "--k-neighbors",
        type=int,
        default=20,
        help="Nearest neighbours to display  (default: 20)",
    )
    p.add_argument(
        "--regex-weight",
        type=float,
        default=0.5,
        help="0=semantic only, 1=regex only  (default: 0.5)",
    )
    p.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors  (default: 30)",
    )
    p.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.15,
        help="UMAP min_dist  (default: 0.15)",
    )
    p.add_argument(
        "--semantic-model",
        default="all-mpnet-base-v2",
        help="Sentence-transformer model name  (default: all-mpnet-base-v2)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="HDBSCAN min cluster size  (default: 100)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recompute of semantic embeddings",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    global SEMANTIC_MODEL
    SEMANTIC_MODEL = args.semantic_model

    # 1. Load cards
    logger.info("Loading card catalog...")
    with open(args.catalog_dir / "cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)

    # 2. Regex ability features
    filtered_cards, regex_features, feature_names = extract_ability_features(cards)

    # 3. Semantic embeddings (cached)
    semantic_embs = compute_semantic_embeddings(
        filtered_cards,
        args.catalog_dir,
        no_cache=args.no_cache,
    )

    # 4. Combine
    combined = combine_features(
        regex_features,
        semantic_embs,
        regex_weight=args.regex_weight,
    )

    # 5. Display data (extended)
    (
        names,
        types,
        mana_costs,
        oracle_texts,
        primary_types,
        color_ids,
        scryfall_ids,
        edhrec_ranks,
    ) = prepare_display_data(filtered_cards)

    # 6. Marker sizes from EDHREC rank
    marker_sizes = compute_marker_sizes(edhrec_ranks)

    # 7. UMAP
    coords = compute_umap(
        combined,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
    )

    # 8. KNN (sklearn)
    neighbors = compute_neighbor_indices(combined, k=args.k_neighbors)

    # 9. HDBSCAN clusters
    cluster_info = compute_clusters(
        coords,
        regex_features,
        feature_names,
        min_cluster_size=args.min_cluster_size,
    )

    # 10. Build HTML
    html = build_html(
        coords,
        names,
        types,
        mana_costs,
        oracle_texts,
        primary_types,
        color_ids,
        neighbors,
        regex_features,
        feature_names,
        scryfall_ids,
        edhrec_ranks,
        marker_sizes,
        cluster_info,
        args.k_neighbors,
    )

    args.output.write_text(html, encoding="utf-8")
    logger.info("Visualization saved to %s (%d cards)", args.output, len(names))

    n_ab = len(ABILITY_PATTERNS)
    print(f"\nDone! Open {args.output} in your browser.")
    print(f"  - {len(names):,} cards plotted")
    print(
        f"  - {n_ab} ability patterns + {len(TYPE_CATEGORIES)} types + 5 colors"
        f" = {n_ab + len(TYPE_CATEGORIES) + 5} regex features"
    )
    print(f"  - {semantic_embs.shape[1]}-dim semantic embeddings ({SEMANTIC_MODEL})")
    print(f"  - Combined: {combined.shape[1]} dims  (regex_weight={args.regex_weight})")
    print(f"  - UMAP metric: cosine  |  {len(cluster_info)} HDBSCAN clusters")
    print(f"  - Click any card to see image + {args.k_neighbors} nearest neighbours")


if __name__ == "__main__":
    main()
