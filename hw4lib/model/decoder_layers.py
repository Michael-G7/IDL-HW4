import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer
from .causal_self_attention import KVCache


class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.
    '''
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        *,
        use_rope: bool = False,
        max_len: int = 2048,
        num_kv_heads: Optional[int] = None,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_len=max_len,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
        )
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mha_attn_weights = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.ffn(x)
        return x, mha_attn_weights

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
        x, mha_attn_weights, present_kv = self.self_attn.forward_cached(
            x,
            past_kv=past_kv,
        )
        x = self.ffn(x)
        return x, mha_attn_weights, present_kv
