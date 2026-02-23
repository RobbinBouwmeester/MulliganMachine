"""Model configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the Commander deck transformer model."""

    # Vocabulary & sequence
    vocab_size: int = 30_400  # updated after catalog build (special + card tokens)
    max_seq_len: int = 103  # [BOS] commander [SEP] 99 cards [EOS]

    # Transformer architecture
    n_layers: int = 8  # number of decoder blocks
    d_model: int = 512  # embedding / hidden dimension
    n_heads: int = 8  # attention heads
    d_ff: int = 2048  # feed-forward inner dimension
    dropout: float = 0.1  # dropout rate

    # Modern architecture options
    use_swiglu: bool = True  # SwiGLU activation in FFN (LLaMA-style)
    use_rmsnorm: bool = True  # RMSNorm instead of LayerNorm

    # Card feature injection (optional)
    use_card_features: bool = False  # whether to concat card features to embeddings
    n_card_features: int = 14  # number of numeric card features (from scryfall.get_card_features)

    # Regularization
    label_smoothing: float = 0.0  # label smoothing for cross-entropy (0.0 = off, 0.1 recommended)
    stochastic_depth_rate: float = (
        0.0  # max drop rate for stochastic depth (0.0 = off, 0.1 recommended)
    )

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 100
    batch_size: int = 64
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1  # gradient accumulation steps

    # Inference
    temperature: float = 0.8
    top_k: int = 100
    top_p: float = 0.95

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def total_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        # Embeddings
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        # Per transformer block: attention (4 * d² for Q/K/V/O + QK-norm) + FFN + norms
        attn_params = 4 * self.d_model**2 + 2 * self.d_model  # QK-norm weights
        if self.use_swiglu:
            ffn_params = 3 * self.d_model * self.d_ff  # gate + up + down projections
        else:
            ffn_params = 2 * self.d_model * self.d_ff
        per_block = attn_params + ffn_params + 4 * self.d_model
        blocks = self.n_layers * per_block
        # Output head is weight-tied with token embedding
        total = emb + blocks
        return total

    def __post_init__(self):
        assert (
            self.d_model % self.n_heads == 0
        ), f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
