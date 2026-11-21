from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np


class MultiHeadAttention:
	"""Multi-head attention composed of learned projections and SDPA."""
	def __init__(self, embed_dim, num_heads):
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")

		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.d_k = embed_dim // num_heads
		self.attention = ScaledDotProductAttention()
		self.q_proj = Linear(embed_dim, embed_dim)
		self.k_proj = Linear(embed_dim, embed_dim)
		self.v_proj = Linear(embed_dim, embed_dim)
		self.out_proj = Linear(embed_dim, embed_dim)

	def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
		self.q_proj.init_weights(Wq, bq)
		self.k_proj.init_weights(Wk, bk)
		self.v_proj.init_weights(Wv, bv)
		self.out_proj.init_weights(Wo, bo)

	def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
		q = self._split_heads(self.q_proj.forward(query))
		k = self._split_heads(self.k_proj.forward(key))
		v = self._split_heads(self.v_proj.forward(value))
		mask = self._merge_masks(key_padding_mask, attn_mask)
		attended = self.attention.forward(q, k, v, mask)
		return self.out_proj.forward(self._concat_heads(attended))

	def backward(self, d_output):
		d_attended = self._split_heads(self.out_proj.backward(d_output))
		d_q, d_k, d_v = self.attention.backward(d_attended)
		d_query = self.q_proj.backward(self._concat_heads(d_q))
		d_key = self.k_proj.backward(self._concat_heads(d_k))
		d_value = self.v_proj.backward(self._concat_heads(d_v))
		return d_query, d_key, d_value

	def _merge_masks(self, key_padding_mask, attn_mask):
		if key_padding_mask is None and attn_mask is None:
			return None
		key_mask = None
		attention_mask = None
		if key_padding_mask is not None:
			key_mask = key_padding_mask[:, np.newaxis, np.newaxis, :].astype(bool)
		if attn_mask is not None:
			attention_mask = attn_mask[np.newaxis, np.newaxis, :, :].astype(bool)
		if key_mask is None:
			return attention_mask
		if attention_mask is None:
			return key_mask
		return key_mask | attention_mask

	def _split_heads(self, x):
		batch_size, seq_len = x.shape[:2]
		x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
		return np.ascontiguousarray(x.transpose(0, 2, 1, 3))

	def _concat_heads(self, x):
		x = np.ascontiguousarray(x.transpose(0, 2, 1, 3))
		batch_size, seq_len = x.shape[:2]
		return x.reshape(batch_size, seq_len, self.embed_dim)
