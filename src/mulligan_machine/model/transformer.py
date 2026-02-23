"""Decoder-only transformer for Commander deck generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mulligan_machine.model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Used by LLaMA, Mistral, and other modern architectures.
    More stable and efficient than standard LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Used by LLaMA, PaLM, and other modern architectures.
    Replaces the standard Linear -> GELU -> Linear FFN with a gated variant:
        FFN(x) = (SiLU(xW_gate) * xW_up) W_down
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention with SDPA and QK-Norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.dropout_p = config.dropout

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        # QK-Norm for stable training at scale
        NormClass = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.q_norm = NormClass(self.d_head)
        self.k_norm = NormClass(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V in one matmul
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # QK-Norm for training stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class TransformerBlock(nn.Module):
    """A single transformer decoder block: attention + FFN with pre-norm.

    Supports stochastic depth (layer drop) for regularization during training.
    Uses RMSNorm + SwiGLU when configured (LLaMA-style).
    """

    def __init__(self, config: ModelConfig, drop_rate: float = 0.0):
        super().__init__()
        NormClass = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln1 = NormClass(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = NormClass(config.d_model)

        if config.use_swiglu:
            self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout),
            )
        self.drop_rate = drop_rate  # stochastic depth drop probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stochastic depth: randomly skip this block during training
        if self.drop_rate > 0.0 and self.training:
            if torch.rand(1, device=x.device).item() < self.drop_rate:
                return x  # skip entire block
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DeckTransformer(nn.Module):
    """
    GPT-2-style decoder-only transformer for Commander deck generation.

    Input: sequence of token IDs [BOS, commander, SEP, card_1, ..., card_99, EOS]
    Output: logits over vocabulary at each position (next-token prediction)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        NormClass = RMSNorm if config.use_rmsnorm else nn.LayerNorm

        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Optional: card feature projection
        if config.use_card_features:
            self.feature_proj = nn.Linear(config.n_card_features, config.d_model)
            # We'll add (not concat) the projected features to the token embeddings
        else:
            self.feature_proj = None

        # Transformer blocks with linearly increasing stochastic depth
        drop_rates = [
            config.stochastic_depth_rate * i / max(config.n_layers - 1, 1)
            for i in range(config.n_layers)
        ]
        self.blocks = nn.ModuleList([TransformerBlock(config, drop_rate=dr) for dr in drop_rates])

        # Output
        self.ln_f = NormClass(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with output head
        self.head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual paths for stable deep training
        if config.n_layers >= 6:
            self._apply_residual_scaling()

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"DeckTransformer initialized: {n_params / 1e6:.1f}M parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_residual_scaling(self):
        """Scale residual path weights by 1/sqrt(2*n_layers) for stable deep training."""
        factor = (2 * self.config.n_layers) ** -0.5
        for block in self.blocks:
            with torch.no_grad():
                block.attn.out_proj.weight.mul_(factor)
                if isinstance(block.ffn, SwiGLUFFN):
                    block.ffn.w_down.weight.mul_(factor)
                else:
                    # Scale the last linear in FFN (index -2, before Dropout)
                    block.ffn[-2].weight.mul_(factor)

    def forward(
        self,
        input_ids: torch.Tensor,
        card_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (B, T) integer token IDs
            card_features: (B, T, n_card_features) optional float features per token

        Returns:
            logits: (B, T, vocab_size) — raw scores for next-token prediction
        """
        B, T = input_ids.shape
        device = input_ids.device

        assert (
            T <= self.config.max_seq_len
        ), f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_emb(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_emb(pos)  # (1, T, d_model)

        x = self.emb_dropout(tok_emb + pos_emb)

        # Optional card feature injection
        if self.feature_proj is not None and card_features is not None:
            feat_emb = self.feature_proj(card_features)  # (B, T, d_model)
            x = x + feat_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output logits
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        card_features: torch.Tensor | None = None,
        ignore_index: int = 0,  # PAD token
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for next-token prediction.

        Targets are input_ids shifted by 1 position.
        """
        logits = self.forward(input_ids, card_features)

        # Shift: predict token at position t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_targets = input_ids[:, 1:].contiguous()  # (B, T-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=ignore_index,
            label_smoothing=self.config.label_smoothing,
        )
        return loss

    @torch.no_grad()
    def generate_next_token_logits(
        self,
        input_ids: torch.Tensor,
        card_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get logits for the next token given a prefix.

        Args:
            input_ids: (B, T) current sequence
            card_features: optional features

        Returns:
            logits: (B, vocab_size) — logits for the next token
        """
        logits = self.forward(input_ids, card_features)
        return logits[:, -1, :]  # logits at last position
