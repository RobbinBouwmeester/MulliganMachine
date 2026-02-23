"""Quick end-to-end sanity check."""

import sys

sys.path.insert(0, "src")

from pathlib import Path
import torch
from mulligan_machine.data.tokenizer import DeckTokenizer
from mulligan_machine.model.config import ModelConfig
from mulligan_machine.model.transformer import DeckTransformer

t = DeckTokenizer.from_catalog(Path("data/catalog"))
c = ModelConfig(vocab_size=t.vocab_size)
print(f"Vocab: {c.vocab_size}, ~{c.total_params_estimate()/1e6:.1f}M params")

m = DeckTransformer(c)
tokens = t.encode_deck("Krenko, Mob Boss", ["Lightning Bolt", "Goblin Warchief", "Mountain"])
x = torch.tensor([tokens])
loss = m.compute_loss(x)
print(f"Loss on single sample: {loss.item():.4f}")
logits = m.generate_next_token_logits(x[:, :5])
print(f"Next-token logits shape: {logits.shape}")
print("All checks passed!")
