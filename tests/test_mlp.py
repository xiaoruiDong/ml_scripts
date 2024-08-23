#!/usr/bin/env python3
import pytest

from torch import nn
from model.mlp import MLP
import torch
import torch


@pytest.mark.parametrize("input_dim", [10, 20])
@pytest.mark.parametrize("hidden_dim", [10, 20])
@pytest.mark.parametrize("output_dim", [10, 20])
@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("activation", [nn.ReLU, nn.Tanh])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_init(
    input_dim,
    hidden_dim,
    output_dim,
    num_layers,
    bias,
    activation,
    batch_norm,
    dropout,
):
    """
    Test the initialization of mlp. This test only tests if
    blocks can be created correctly.
    """
    mlp = MLP(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        bias,
        activation,
        dropout=dropout,
        batch_norm=batch_norm,
    )

    layers = mlp.layers._modules

    # check input dim
    assert layers["linear0"].in_features == input_dim
    # infer the number of layers from names
    num_layers = len([k for k in layers.keys() if k.startswith("linear")]) - 1
    # check hidden layers
    for i in range(num_layers):
        assert layers[f"linear{i}"].out_features == hidden_dim
    for i in range(1, num_layers + 1):
        assert layers[f"linear{i}"].in_features == hidden_dim
    # check output dim
    assert layers[f"linear{num_layers}"].out_features == output_dim
    # check bias
    assert (layers[f"linear{i}"].bias is not None) == bias
    # check activation
    assert isinstance(layers[f"act0"], activation)
    # check batch norm
    assert (layers.get("batch_norm0") is not None) == batch_norm
    # check dropout
    assert layers.get("dropout0").p == dropout


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("input_dim", [10, 20])
@pytest.mark.parametrize("output_dim", [5, 10])
def test_forward(batch_size, input_dim, output_dim):
    """Test MLP forward returns the output with expected shape"""
    x = torch.randn((batch_size, input_dim))
    mlp = MLP(input_dim, 10, output_dim, 3)
    y = mlp(x)
    assert y.shape == (batch_size, output_dim)
