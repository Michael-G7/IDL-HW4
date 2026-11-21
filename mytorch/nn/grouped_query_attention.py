from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np
from typing import Optional

class GroupedQueryAttention:
    """
    Grouped Query Attention
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of key/value heads shared across query groups
        """
        # DO NOT MODIFY
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.num_kv_heads  = num_kv_heads
        self.head_dim      = embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # TODO: Implement the attention layer and the projections

        # Initialize your scaled dot product attention layer
        self.attention = NotImplementedError

        # Initialize your linear layers
        # Query is projected to embed_dim, but key and value are compact:
        # they carry num_kv_heads heads instead of num_heads.
        # What output size does that give the key and value projections?
        self.q_proj   = NotImplementedError
        self.k_proj   = NotImplementedError
        self.v_proj   = NotImplementedError
        self.out_proj = NotImplementedError

    def init_weights(
        self,
        Wq: np.ndarray,
        bq: np.ndarray,
        Wk: np.ndarray,
        bk: np.ndarray,
        Wv: np.ndarray,
        bv: np.ndarray,
        Wo: np.ndarray,
        bo: np.ndarray,
    ) -> None:
        """
        Initialize the weights and biases with the given values.
        Note that Wk and Wv are shaped for the compact key/value heads.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        key_padding_mask: Optional[np.ndarray] = None,
        attn_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """

        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # Project inputs
        q = NotImplementedError
        k = NotImplementedError
        v = NotImplementedError

        # Reshape for multiple heads
        # Query and key/value do not use the same head count here
        q = NotImplementedError
        k = NotImplementedError
        v = NotImplementedError

        # Repeat key/value heads for their query groups
        k = NotImplementedError
        v = NotImplementedError

        # Combine padding and causal masks
        mask = NotImplementedError

        # Apply attention
        attn_outputs = NotImplementedError

        # Merge heads
        attn_output = NotImplementedError

        # Final projection
        output = NotImplementedError

        raise NotImplementedError

    def backward(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for grouped query attention.
        """

        # TODO: Implement backward pass

        # Backpropagate through output projection
        d_attn_output = NotImplementedError

        # Undo head splitting
        d_attn_outputs = NotImplementedError

        # Backpropagate through attention
        d_q, d_k, d_v = NotImplementedError

        # Reduce the repeated key/value gradients back to compact heads
        d_k = NotImplementedError
        d_v = NotImplementedError

        # Merge head gradients
        d_q = NotImplementedError
        d_k = NotImplementedError
        d_v = NotImplementedError

        # Backpropagate through input projections
        d_q = NotImplementedError
        d_k = NotImplementedError
        d_v = NotImplementedError

        raise NotImplementedError

    def _merge_masks(
        self,
        key_padding_mask: Optional[np.ndarray],
        attn_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Merge two mask types into a single mask.
        The masks are applied after the key/value heads have been repeated.
        """
        # TODO: Implement mask merging

        # Expand masks for broadcasting
        key_mask = NotImplementedError
        attention_mask = NotImplementedError

        # Combine masks
        combined_mask = NotImplementedError

        raise NotImplementedError

    def _split_heads(
        self,
        x: np.ndarray,
        num_heads: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reshape tensor for multi-head attention.
        Query is split into num_heads, key and value into num_kv_heads.
        :param x: (N, T, num_heads * head_dim)
        :return: (N, num_heads, T, head_dim)
        """
        # Select the head count for this call (DO NOT MODIFY)
        num_heads = self.num_heads if num_heads is None else num_heads

        # TODO: Implement head splitting

        # Reshape and transpose for heads
        x = NotImplementedError
        x = NotImplementedError

        raise NotImplementedError

    def _concat_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        This is used on both query heads and compact key/value heads,
        so do not assume which head count x carries.
        :param x: (N, num_heads, T, embed_dim // num_heads)
        :return: (N, T, embed_dim)
        """
        # TODO: Implement head concatenation

        # Transpose and reshape
        x = NotImplementedError
        x = NotImplementedError

        raise NotImplementedError

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Repeat each key/value head for the query heads that share it.
        :param x: (N, num_kv_heads, S, head_dim)
        :return: (N, num_heads, S, head_dim)
        """
        # TODO: Implement key/value head repetition

        # Each head is needed num_kv_groups times
        # Think about where the copies of one head have to sit so that a
        # query head lines up with the key/value head of its own group
        x = NotImplementedError

        raise NotImplementedError

    def _sum_kv_gradients(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce repeated key/value gradients back to compact heads.
        :param x: (N, num_heads, S, head_dim)
        :return: (N, num_kv_heads, S, head_dim)
        """
        # TODO: Implement the group reduction

        # This undoes _repeat_kv, and the adjoint of a repeat is a sum
        # Think about how to expose the group structure inside the head
        # dimension, and which part of it you should reduce over
        x = NotImplementedError

        raise NotImplementedError
