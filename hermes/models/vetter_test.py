"""Tests for the TransitVetter model in hermes.models.vetter."""

import jax.numpy as jnp
from flax import nnx

from hermes.configs.default import get_config
from hermes.models.vetter import REGRESSION_TARGETS, TransitVetter


def test_model_forward_shapes():
  """The model returns a logit and per-target mean/log-variance."""
  config = get_config()
  config.model.conv_channels = (8, 16)
  config.model.embed_dim = 32
  config.model.mlp_dim = 64
  config.model.transformer_layers = 1
  config.model.num_heads = 2
  model = TransitVetter(config.model, 201, 51, rngs=nnx.Rngs(0))
  model.eval()

  out = model(jnp.ones((3, 201)), jnp.ones((3, 51)), jnp.ones((3, 8)))
  assert out["logit"].shape == (3,)
  assert out["mean"].shape == (3, len(REGRESSION_TARGETS))
  assert out["log_variance"].shape == (3, len(REGRESSION_TARGETS))
