"""1D convolutional stem that returns a token sequence for the transformer."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx
from jax import Array


class ConvStem(nnx.Module):
  """Stack of 1D convolution / ReLU / average-pool blocks."""

  def __init__(
      self,
      channels: Sequence[int],
      kernel_size: int,
      *,
      rngs: nnx.Rngs,
  ):
    """Builds the convolutional stem.

    Args:
      channels: Output channel count of each convolution block.
      kernel_size: Convolution kernel width (``SAME`` padding).
      rngs: Random-number generators for parameter initialisation.
    """
    self.convs = nnx.List([])
    in_channels = 1
    for out_channels in channels:
      self.convs.append(
          nnx.Conv(
              in_channels, out_channels, kernel_size=(kernel_size,), rngs=rngs
          )
      )
      in_channels = out_channels

  def __call__(self, view: Array) -> Array:
    """Applies the stem to a batch of 1D views.

    Args:
      view: Binned flux of shape ``(batch, length)``.

    Returns:
      Token sequence of shape ``(batch, reduced_length, channels[-1])``.
    """
    x = view[..., jnp.newaxis]
    for conv in self.convs:
      x = nnx.relu(conv(x))
      x = nnx.avg_pool(x, window_shape=(2,), strides=(2,))
    return x
