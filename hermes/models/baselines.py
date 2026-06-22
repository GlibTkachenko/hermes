"""Benchmark baselines: a no-training view-SNR statistic and an AstroNet CNN."""

from __future__ import annotations

import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from flax import nnx
from jax import Array

from hermes.models import cnn


def view_snr_scores(local_views: np.ndarray) -> np.ndarray:
  """Scores candidates by transit-region depth over out-of-transit scatter.

  The depth-normalised views all bottom out at -1, so absolute depth carries no
  information; instead this contrasts the central (in-transit) bins against the
  scatter of the edge (out-of-transit) bins. A real transit has a deep, clean
  centre and flat edges (high score); folded noise does not. This is the
  classical "is there a significant dip at the folded period?" baseline, with no
  learning.

  Args:
    local_views: Normalised local views, shape ``(n, local_bins)``.

  Returns:
    A score per candidate (higher means more transit-like).
  """
  views = np.asarray(local_views)
  num_bins = views.shape[1]
  lower, upper = int(0.4 * num_bins), int(0.6 * num_bins)
  centre = views[:, lower:upper].mean(axis=1)
  edges = np.concatenate([views[:, :lower], views[:, upper:]], axis=1)
  return -centre / (edges.std(axis=1) + 1e-6)


class AstroNetClassifier(nnx.Module):
  """CNN-only dual-view classifier (AstroNet, Shallue & Vanderburg 2018)."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      global_bins: int,
      local_bins: int,
      *,
      rngs: nnx.Rngs,
  ):
    channels = tuple(config.conv_channels)
    self.global_stem = cnn.ConvStem(channels, config.kernel_size, rngs=rngs)
    self.local_stem = cnn.ConvStem(channels, config.kernel_size, rngs=rngs)
    global_tokens = self.global_stem(jnp.zeros((1, global_bins)))
    local_tokens = self.local_stem(jnp.zeros((1, local_bins)))
    flat = (
        global_tokens.shape[1] * global_tokens.shape[2]
        + local_tokens.shape[1] * local_tokens.shape[2]
    )
    self.linear_in = nnx.Linear(flat, config.mlp_dim, rngs=rngs)
    self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
    self.linear_out = nnx.Linear(config.mlp_dim, 1, rngs=rngs)

  def __call__(self, global_view: Array, local_view: Array) -> Array:
    """Returns a logit of shape ``(batch,)``."""
    global_features = self.global_stem(global_view)
    local_features = self.local_stem(local_view)
    fused = jnp.concatenate(
        [
            global_features.reshape(global_features.shape[0], -1),
            local_features.reshape(local_features.shape[0], -1),
        ],
        axis=-1,
    )
    hidden = self.dropout(nnx.relu(self.linear_in(fused)))
    return self.linear_out(hidden)[..., 0]


@nnx.jit(static_argnames=("positive_weight",))
def _astronet_step(model, optimizer, batch, positive_weight):
  """One weighted-BCE gradient step for the AstroNet baseline."""

  def loss_fn(model):
    logits = model(batch["global_view"], batch["local_view"])
    labels = batch["label"].astype(jnp.float32)
    per_example = optax.sigmoid_binary_cross_entropy(logits, labels)
    weight = jnp.where(labels > 0.5, positive_weight, 1.0)
    return jnp.sum(per_example * weight) / jnp.sum(weight)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)
  return loss


def train_astronet(
    model: AstroNetClassifier,
    dataset,
    *,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    positive_weight: float,
) -> AstroNetClassifier:
  """Trains the AstroNet baseline with weighted binary cross-entropy.

  Args:
    model: The `AstroNetClassifier` to train.
    dataset: A `hermes.data.dataset.HermesDataset` training split.
    num_epochs: Number of epochs.
    batch_size: Batch size.
    learning_rate: AdamW learning rate.
    positive_weight: Positive-class up-weighting for the class imbalance.

  Returns:
    The trained model (updated in place and returned for convenience).
  """
  optimizer = nnx.Optimizer(
      model, optax.adamw(learning_rate), wrt=nnx.Param
  )
  model.train()
  for epoch in range(num_epochs):
    for batch in dataset.iterate(batch_size, shuffle=True, seed=epoch):
      _astronet_step(model, optimizer, batch, positive_weight)
  return model
