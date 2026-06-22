"""Output heads: a binary classifier and a heteroscedastic regressor."""

from __future__ import annotations

from flax import nnx
from jax import Array


class ClassifierHead(nnx.Module):
  """Two-layer MLP producing a single binary-classification logit."""

  def __init__(
      self,
      in_features: int,
      hidden_features: int,
      dropout_rate: float,
      *,
      rngs: nnx.Rngs,
  ):
    self.linear_in = nnx.Linear(in_features, hidden_features, rngs=rngs)
    self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    self.linear_out = nnx.Linear(hidden_features, 1, rngs=rngs)

  def __call__(self, features: Array) -> Array:
    """Returns a logit of shape ``(batch,)``."""
    hidden = self.dropout(nnx.relu(self.linear_in(features)))
    return self.linear_out(hidden)[..., 0]


class HeteroscedasticHead(nnx.Module):
  """Predicts a mean and a log-variance for each regression target."""

  def __init__(self, in_features: int, num_targets: int, *, rngs: nnx.Rngs):
    self.mean = nnx.Linear(in_features, num_targets, rngs=rngs)
    self.log_variance = nnx.Linear(in_features, num_targets, rngs=rngs)

  def __call__(self, features: Array) -> tuple[Array, Array]:
    """Returns the per-target mean and log-variance."""
    return self.mean(features), self.log_variance(features)
