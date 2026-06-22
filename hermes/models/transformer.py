"""Transformer encoder over the convolutional token sequence."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from jax import Array


class EncoderLayer(nnx.Module):
  """A single pre-norm transformer encoder layer."""

  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      mlp_dim: int,
      dropout_rate: float,
      *,
      rngs: nnx.Rngs,
  ):
    self.norm_attn = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.attention = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_dim,
        dropout_rate=dropout_rate,
        decode=False,
        rngs=rngs,
    )
    self.norm_mlp = nnx.LayerNorm(embed_dim, rngs=rngs)
    self.linear_in = nnx.Linear(embed_dim, mlp_dim, rngs=rngs)
    self.linear_out = nnx.Linear(mlp_dim, embed_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

  def __call__(self, x: Array) -> Array:
    attended = self.attention(self.norm_attn(x))
    x = x + attended
    hidden = self.dropout(nnx.gelu(self.linear_in(self.norm_mlp(x))))
    return x + self.linear_out(hidden)


class SequenceEncoder(nnx.Module):
  """Projects, position-encodes and transformer-encodes a token sequence."""

  def __init__(
      self,
      in_channels: int,
      sequence_length: int,
      embed_dim: int,
      num_layers: int,
      num_heads: int,
      mlp_dim: int,
      dropout_rate: float,
      *,
      rngs: nnx.Rngs,
  ):
    """Builds the sequence encoder.

    Args:
      in_channels: Channel count of the input tokens.
      sequence_length: Number of tokens (sets the positional-embedding size).
      embed_dim: Transformer embedding width.
      num_layers: Number of encoder layers.
      num_heads: Attention heads per layer.
      mlp_dim: Hidden width of the feed-forward sublayers.
      dropout_rate: Dropout probability.
      rngs: Random-number generators for parameter initialisation.
    """
    self.input_projection = nnx.Linear(in_channels, embed_dim, rngs=rngs)
    self.position_embedding = nnx.Param(
        nnx.initializers.normal(stddev=0.02)(
            rngs.params(), (1, sequence_length, embed_dim)
        )
    )
    self.layers = nnx.List(
        [
            EncoderLayer(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        ]
    )
    self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

  def __call__(self, tokens: Array) -> Array:
    """Encodes a token sequence into a pooled embedding.

    Args:
      tokens: Token sequence of shape ``(batch, sequence_length, in_channels)``.

    Returns:
      Pooled embedding of shape ``(batch, embed_dim)``.
    """
    x = self.input_projection(tokens) + self.position_embedding[...]
    for layer in self.layers:
      x = layer(x)
    return jnp.mean(self.norm(x), axis=1)
