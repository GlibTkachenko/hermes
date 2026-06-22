"""TransitVetter: dual-view CNN + transformer with calibrated heads."""

from __future__ import annotations

import jax.numpy as jnp
import ml_collections
from flax import nnx
from jax import Array

from hermes.models import cnn, heads, transformer

#: Order of the regression targets predicted by the heteroscedastic head.
REGRESSION_TARGETS = ("log_depth", "log_duration")


class TransitVetter(nnx.Module):
  """Dual-view transit classifier with calibrated parameter regression."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      global_bins: int,
      local_bins: int,
      *,
      rngs: nnx.Rngs,
  ):
    """Builds the vetter from the model configuration.

    Args:
      config: The ``model`` section of the HERMES configuration.
      global_bins: Length of the global view.
      local_bins: Length of the local view.
      rngs: Random-number generators for parameter initialisation.
    """
    channels = tuple(config.conv_channels)
    self.global_stem = cnn.ConvStem(channels, config.kernel_size, rngs=rngs)
    self.local_stem = cnn.ConvStem(channels, config.kernel_size, rngs=rngs)

    # Infer the stem output shapes from a dummy pass so the positional
    # embedding and fusion dimensions match the configured view sizes.
    global_tokens = self.global_stem(jnp.zeros((1, global_bins)))
    local_tokens = self.local_stem(jnp.zeros((1, local_bins)))
    _, global_length, stem_channels = global_tokens.shape
    local_flat = local_tokens.shape[1] * local_tokens.shape[2]

    self.global_encoder = transformer.SequenceEncoder(
        in_channels=stem_channels,
        sequence_length=global_length,
        embed_dim=config.embed_dim,
        num_layers=config.transformer_layers,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        dropout_rate=config.dropout_rate,
        rngs=rngs,
    )
    self.scalar_projection = nnx.Linear(
        config.num_scalar_features, config.embed_dim, rngs=rngs
    )

    fused_features = config.embed_dim + local_flat + config.embed_dim
    self.classifier = heads.ClassifierHead(
        fused_features, config.mlp_dim, config.dropout_rate, rngs=rngs
    )
    self.regressor = heads.HeteroscedasticHead(
        fused_features, len(REGRESSION_TARGETS), rngs=rngs
    )

  def __call__(
      self, global_view: Array, local_view: Array, scalar_features: Array
  ) -> dict[str, Array]:
    """Runs the model.

    Args:
      global_view: Global phase-folded view, shape ``(batch, global_bins)``.
      local_view: Local phase-folded view, shape ``(batch, local_bins)``.
      scalar_features: Standardised scalar features, ``(batch, num_features)``.

    Returns:
      Dict with ``logit`` (``(batch,)``) and the regression ``mean`` and
      ``log_variance`` (each ``(batch, 2)`` over ``REGRESSION_TARGETS``).
    """
    global_embedding = self.global_encoder(self.global_stem(global_view))
    local_tokens = self.local_stem(local_view)
    local_embedding = local_tokens.reshape(local_tokens.shape[0], -1)
    scalar_embedding = nnx.relu(self.scalar_projection(scalar_features))

    fused = jnp.concatenate(
        [global_embedding, local_embedding, scalar_embedding], axis=-1
    )
    mean, log_variance = self.regressor(fused)
    return {
        "logit": self.classifier(fused),
        "mean": mean,
        "log_variance": log_variance,
    }
