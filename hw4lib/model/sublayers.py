import torch.nn as nn
import torch
from typing import Tuple, Optional
from .causal_self_attention import CausalSelfAttention, KVCache


class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, *, use_rope: bool = False, max_len: int = 2048, num_kv_heads: Optional[int] = None, rope_base: float = 10_000.0):
        super().__init__()

        # DO NOT MODIFY
        # Only the rotary path has key/value groups
        self.use_rope = use_rope
        if not use_rope and num_kv_heads not in (None, num_heads):
            raise ValueError("num_kv_heads requires use_rope=True")

        if use_rope:
            self.mha = CausalSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                max_len=max_len,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                rope_base=rope_base,
            )
        else:
            # DO NOT MODIFY
            # This is the additive-positional path used by HW4P2 and is not
            # part of the HW4P1 exercise
            self.mha = nn.MultiheadAttention(
                embed_dim   = d_model,
                num_heads   = num_heads,
                dropout     = dropout,
                batch_first = True
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)

        if self.use_rope:
            x, mha_attn_weights, _ = self.mha(
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # DO NOT MODIFY
            # This is the additive-positional path used by HW4P2 and is not
            # part of the HW4P1 exercise
            x, mha_attn_weights = self.mha(
                query                = x,
                key                  = x,
                value                = x,
                key_padding_mask     = key_padding_mask,
                need_weights         = True,
                attn_mask            = attn_mask,
                average_attn_weights = True,
            )

        x = residual + self.dropout(x)
        return x, mha_attn_weights

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
        # DO NOT MODIFY: KV caching is available only on the rotary path.
        if not self.use_rope:
            raise ValueError("KV caching requires use_rope=True")

        residual = x
        normalized = self.norm(x)

        attention_output, mha_attn_weights, present_kv = self.mha(
            normalized,
            past_kv=past_kv,
            use_cache=True,
        )

        x = residual + self.dropout(attention_output)

        # DO NOT MODIFY: a cached attention call must return cache state.
        if present_kv is None:
            raise RuntimeError("Cached attention did not return a key/value cache")

        return x, mha_attn_weights, present_kv


class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x
