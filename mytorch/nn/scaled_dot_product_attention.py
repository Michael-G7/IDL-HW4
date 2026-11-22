import numpy as np
from .activation import Softmax


class ScaledDotProductAttention:
    """Scaled dot-product attention with manual gradients."""

    def __init__(self):
        self.eps = 1e10
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        self.Q = Q
        self.K = K
        self.V = V
        self.d_k = Q.shape[-1]

        self.scaled_dot_product = np.matmul(Q, np.swapaxes(K, -1, -2))
        self.scaled_dot_product /= np.sqrt(self.d_k)
        if mask is not None:
            mask_bool = mask.astype(bool)
            self.scaled_dot_product = np.where(
                mask_bool, -self.eps, self.scaled_dot_product
            )

        self.attention_scores = self.softmax.forward(self.scaled_dot_product)
        if mask is not None:
            fully_masked = np.all(mask_bool, axis=-1, keepdims=True)
            if np.any(fully_masked):
                self.attention_scores = np.where(
                    fully_masked,
                    np.zeros_like(self.attention_scores),
                    self.attention_scores,
                )
                self.softmax.A = self.attention_scores

        self.output = np.matmul(self.attention_scores, V)
        return self.output

    def backward(self, d_output):
        d_V = np.matmul(np.swapaxes(self.attention_scores, -1, -2), d_output)
        d_attention_scores = np.matmul(d_output, np.swapaxes(self.V, -1, -2))
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        d_scaled_dot_product /= np.sqrt(self.d_k)

        d_Q = np.matmul(d_scaled_dot_product, self.K)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -1, -2), self.Q)
        return d_Q, d_K, d_V
