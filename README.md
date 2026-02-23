# MulliganMachine

AI-powered MTG Commander deck builder using a transformer model in PyTorch.

Given a commander (and optionally some cards you already want to include), MulliganMachine generates a complete 100-card Commander decklist using a GPT-2-style decoder transformer trained on real decklists scraped from EDHREC, Moxfield, and Archidekt.

## How It Works

- **Card Vocabulary**: Each of the ~27,000 Commander-legal cards is a token in a custom vocabulary
- **Sequence Format**: `[BOS] commander [SEP] card_1 card_2 ... card_99 [EOS]`
- **Model**: Decoder-only transformer (8 layers, 512 dim, 8 heads, ~50M parameters)
- **Training**: Next-token prediction with random card permutation (order-invariance)
- **Inference**: Autoregressive generation with constrained decoding:
  - Color identity masking (only legal cards)
  - Singleton enforcement (no duplicates except basic lands)
  - Land-count nudging (targets ~36 lands)

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .
```

For GPU training, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Step 1: Build Card Catalog (Scryfall)

Downloads all Commander-legal card data from Scryfall (~80 MB download):

```bash
python scripts/scrape_scryfall.py
```

### Step 2: Collect Training Data

**Quick start — EDHREC average decklists (~2,000 decks):**
```bash
python scripts/scrape_edhrec.py
```

**Scale up — Moxfield & Archidekt (runs for hours, 10K+ decks each):**
```bash
python scripts/scrape_moxfield.py --max-decks 50000
python scripts/scrape_archidekt.py --max-decks 50000
```

### Step 3: Verify Data

```bash
python scripts/preprocess.py
```

### Step 4: Train

```bash
python scripts/train.py
```

Adjust for your hardware:
```bash
# Smaller model for quick iteration
python scripts/train.py --n-layers 4 --d-model 256 --d-ff 1024 --batch-size 32

# Full model (requires 16+ GB VRAM)
python scripts/train.py --n-layers 8 --d-model 512 --batch-size 64 --max-epochs 50
```

### Step 5: Generate Decks

```bash
# Generate a complete deck for a commander
python scripts/generate.py --commander "Atraxa, Praetors' Voice" --checkpoint checkpoints/last.ckpt

# Start with some cards already chosen
python scripts/generate.py \
  --commander "Krenko, Mob Boss" \
  --cards "Goblin Warchief,Lightning Bolt,Purphoros, God of the Forge" \
  --checkpoint checkpoints/last.ckpt

# Generate 5 variants
python scripts/generate.py \
  --commander "Meren of Clan Nel Toth" \
  --n-decks 5 \
  --temperature 0.9 \
  --checkpoint checkpoints/last.ckpt
```

### Step 6: Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/last.ckpt
```

## Project Structure

```
MulliganMachine/
├── pyproject.toml              # Package config & dependencies
├── scripts/                    # CLI entry points
│   ├── scrape_scryfall.py      # Download card data
│   ├── scrape_edhrec.py        # Scrape EDHREC average decks
│   ├── scrape_moxfield.py      # Scrape Moxfield decklists
│   ├── scrape_archidekt.py     # Scrape Archidekt decklists
│   ├── preprocess.py           # Verify tokenization & splits
│   ├── train.py                # Train the model
│   ├── generate.py             # Generate decklists
│   └── evaluate.py             # Evaluate model quality
├── src/mulligan_machine/
│   ├── data/
│   │   ├── scryfall.py         # Scryfall bulk data → card catalog
│   │   ├── tokenizer.py        # Card name ↔ token ID mapping
│   │   └── dataset.py          # PyTorch Dataset for decklists
│   ├── scraping/
│   │   ├── edhrec.py           # EDHREC scraper
│   │   ├── moxfield.py         # Moxfield scraper
│   │   └── archidekt.py        # Archidekt scraper
│   ├── model/
│   │   ├── config.py           # Model hyperparameters
│   │   └── transformer.py      # Deck transformer (GPT-2-style)
│   ├── training/
│   │   └── trainer.py          # PyTorch Lightning training loop
│   ├── inference/
│   │   └── generator.py        # Constrained deck generation
│   └── evaluation/
│       └── metrics.py          # Deck quality metrics
├── data/                       # Data directory (gitignored)
│   ├── raw/                    # Scraped decklists
│   ├── processed/              # Stats & preprocessed data
│   └── catalog/                # Card catalog & token maps
└── checkpoints/                # Model weights (gitignored)
```

## Architecture Details

The model is a **decoder-only transformer** (similar to GPT-2) with:

| Component | Size |
|-----------|------|
| Token vocabulary | ~27,000 cards + 5 special tokens |
| Embedding dimension | 512 |
| Attention heads | 8 |
| Transformer layers | 8 |
| FFN dimension | 2048 |
| Max sequence length | 103 |
| Total parameters | ~50M |

**Key design choices:**
- **Random permutation training**: Each epoch shuffles the 99-card order, teaching order-invariant generation
- **Weight tying**: Token embeddings shared with output projection (fewer params, better generalization)
- **Pre-LayerNorm**: More stable training than post-norm
- **Constrained decoding**: Color identity, singleton, and land-count rules enforced via logit masking at inference time

## License

MIT
