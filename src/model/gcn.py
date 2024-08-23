#!/usr/bin/env python3

from typing import Type

from torch import nn
from torch_geometric.nn import (
    GCNConv,
    BatchNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    Sequential,
)

from model.mlp import MLP


class GCN(nn.Module):

    def __init__(
        self,
        input_dim: int = -1,
        gcn_hidden_dim: int = 100,
        n_gcn_layers: int = 2,
        gcn_activation: Type[nn.Module] = nn.ReLU,
        gcn_batch_norm: bool = True,
        pooling: str = "mean",
        ffnn_hidden_dim: int = 50,
        ffnn_output_dim: int = 1,
        ffnn_activation: Type[nn.Module] = nn.ReLU,
        ffnn_dropout: float = 0.0,
        ffnn_batch_norm: bool = True,
        n_ffnn_layers: int = 2,
    ):

        super().__init__()

        # Define the layers in GCN block
        # In each layer GCNConv -> BatchNorm -> Activation
        gcn_modules = []
        for i_layer in range(n_gcn_layers - 1):
            if i_layer:
                gcn_modules.append(
                    (GCNConv(gcn_hidden_dim, gcn_hidden_dim), "x, edge_index -> x"),
                )
            else:
                gcn_modules.append(
                    (GCNConv(input_dim, gcn_hidden_dim), "x, edge_index -> x"),
                )
            gcn_modules += [BatchNorm(gcn_hidden_dim), gcn_activation(inplace=True)]
        gcn_modules += [(GCNConv(gcn_hidden_dim, gcn_hidden_dim), "x, edge_index -> x")]

        self.gcn_layers = Sequential("x, edge_index", gcn_modules)

        # Pooling function to aggregate node embeddings into a graph embedding
        self.pooling = {
            "mean": global_mean_pool,
            "sum": global_add_pool,
            "max": global_max_pool,
        }[pooling]

        # FFNN layer for readout
        self.ffnn_layers = MLP(
            input_dim=gcn_hidden_dim,
            hidden_dim=ffnn_hidden_dim,
            output_dim=ffnn_output_dim,
            n_layers=n_ffnn_layers,
            activation=ffnn_activation,
            dropout=ffnn_dropout,
            batch_norm=ffnn_batch_norm,
        )

    def forward(self, x, edge_index, batch):

        x = self.gcn_layers(x, edge_index)

        x = self.pooling(x, batch)

        return self.ffnn_layers(x)

    @classmethod
    def get_default_hparams(cls):

        return {
            "input_dim": -1,
            "gcn_hidden_dim": 100,
            "n_gcn_layers": 2,
            "gcn_activation": nn.ReLU,
            "gcn_batch_norm": True,
            "pooling": "mean",
            "ffnn_hidden_dim": 50,
            "ffnn_output_dim": 1,
            "ffnn_activation": nn.ReLU,
            "ffnn_dropout": 0.0,
            "ffnn_batch_norm": True,
            "n_ffnn_layers": 2,
        }
