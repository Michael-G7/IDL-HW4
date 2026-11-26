import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term[: pe[0, :, 1::2].shape[-1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}"
            )
        return x + self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)


class RotaryPositionalEncoding(nn.Module):
    """Apply rotary position information to attention queries and keys."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0):
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary encoding")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if base <= 0:
            raise ValueError("base must be positive")

        self.head_dim = head_dim
        self.max_len = max_len

        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequencies = 1.0 / (base ** (dimensions / head_dim))

        positions = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, heads, sequence, head_dim)")
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f"Expected head_dim {self.head_dim}, but received {x.size(-1)}"
            )
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError("start_pos must be an integer")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        seq_len = x.size(-2)
        end_pos = start_pos + seq_len
        if end_pos > self.max_len:
            raise ValueError(
                f"Positions up to {end_pos - 1} exceed max_len {self.max_len}"
            )

        cos = self.cos[start_pos:end_pos].to(device=x.device, dtype=x.dtype)
        sin = self.sin[start_pos:end_pos].to(device=x.device, dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        output = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
        return output
