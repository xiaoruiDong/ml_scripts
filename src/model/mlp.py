"""
A module containing multi-layer perceptron (MLP) models.
"""

from torch import nn


class MLP(nn.Module):
    """
    A simplified multi-layer perceptron (MLP) model whose hidden layer dimensions are the same
    across all hidden layers.

    Args:
        input_dim (int): Size of the input data.
        hidden_dim (int): Size of the hidden layer.
        output_dim (int): Size of the output data.
        n_layers (int): Number of hidden layers. It should be an int $>=$ 1.
        bias (bool, optional): If True, adds a learnable bias to the linear layers. Default: True.
        activation (nn.Module, optional): Activation function. Default: nn.ReLU.
        dropout (float, optional): Dropout rate. Default: 0.0.
    Returns:
        torch.Tensor: Output tensor.
    Example:
        >>> mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, n_layers=3)
        >>> input = torch.randn(2, 10)
        >>> output = mlp(input)
        >>> print(output.shape)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        bias: bool = True,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for i_layer in range(n_layers):
            # Each layer consists of
            # linear -> batch_norm -> activation -> dropout
            if i_layer:
                self.layers.add_module(
                    f"linear{i_layer}", nn.Linear(hidden_dim, hidden_dim, bias=bias)
                )
            else:
                self.layers.add_module(
                    f"linear{i_layer}", nn.Linear(input_dim, hidden_dim, bias=bias)
                )

            if batch_norm:
                self.layers.add_module(
                    f"batch_norm{i_layer}", nn.BatchNorm1d(hidden_dim)
                )

            self.layers.add_module(f"act{i_layer}", activation())
            self.layers.add_module(f"dropout{i_layer}", nn.Dropout(dropout))

        self.layers.add_module(
            f"linear{n_layers}", nn.Linear(hidden_dim, output_dim, bias=bias)
        )

    def forward(self, x):
        return self.layers(x)
