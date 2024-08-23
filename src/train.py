#!/usr/bin/env python3

from typing import Type
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from model.gcn import GCN
from utils import get_hparams


default_hparams = {
    "lr": 0.001,  # learning rate
    "batch_size": 64,  # batch size
    "num_epochs": 10,  # number of epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # device
    "loss_fn": torch.nn.MSELoss,  # loss function
    "optimizer": torch.optim.Adam,  # optimizer
}


def _forward(
    model: torch.nn.Module,
    data: Data | tuple,
    is_gcn: bool = False,
    device: str = "cpu",
) -> tuple:
    """
    A helper function to forward the data through the model.

    Args:
        model (torch.nn.Module): The model to be used for forward pass.
        data (torch_geometric.data.Data): The data to be forwarded.
        is_gcn (bool): Whether the model is a GCN model.
        device (str): The device to use for forward pass.

    Returns:
        torch.Tensor: The output of the model.
        torch.Tensor: The target of the data (if the model is a GCN model).
    """
    if is_gcn:
        data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        return output, data.y
    else:
        input, target = data
        input = input.to(device)
        target = target.to(device)
        return model(input), target


def train_one_epoch(
    data_loader: "torch.utils.data.DataLoader",
    model: "torch.nn.Module",
    hparams: dict | None = None,
):
    """
    Train the model for one epoch.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader for the training data.
        model (torch.nn.Module): The model to be trained.
        hparams (dict, optional): hyperparameters used in training. Default to None,
            equivalent to use default configuration. Options are:
            - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training. Default:
                torch.nn.MSELoss
            - lr (float): The learning rate for the optimizer. Default: 0.001
            - device (str): The device to use for training. Default: "cuda" if available, otherwise "cpu"
            - optimizer (torch.optim.Optimizer): The optimizer for the model. Default: torch.optim.Adam
    """
    _hparams = get_hparams(default_hparams, hparams)
    optimizer = _hparams["optimizer"](model.parameters(), lr=_hparams["lr"])
    loss_fn = _hparams["loss_fn"]()

    # Move the model to the specified device
    model = model.to(_hparams["device"])
    is_gcn = isinstance(model, GCN)

    # Set the model to training mode
    model.train()

    for data in data_loader:
        optimizer.zero_grad()
        output, target = _forward(model, data, is_gcn, _hparams["device"])
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def evaluate_model(
    data_loader: "torch.utils.data.DataLoader",
    model: "torch.nn.Module",
    hparams: dict | None = None,
) -> float:
    """
    Evaluate the model on the given data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader for the evaluation data.
        model (torch.nn.Module): The model to be evaluated.
        hparams (dict, optional): hyperparameters used in training. Default to None,
            equivalent to use default configuration. Options are:
            - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training. Default:
                torch.nn.MSELoss
            - device (str): The device to use for training. Default: "cuda" if available, otherwise "cpu"
    Returns:
        float: The average loss on the evaluated dataset
    """
    _hparams = get_hparams(default_hparams, hparams)
    loss_fn = _hparams["loss_fn"]()

    # Move the model to the specified device
    model = model.to(_hparams["device"])
    is_gcn = isinstance(model, GCN)

    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    with torch.no_grad():

        for data in data_loader:
            output, target = _forward(model, data, is_gcn, _hparams["device"])
            total_loss += loss_fn(output, target).item()

    # Return the average loss
    return total_loss / len(data_loader.dataset)


def train_model(
    train_dataset: "torch.utils.data.Dataset",
    val_dataset: "torch.utils.data.Dataset",
    model: "torch.nn.Module",
    hparams: dict | None = None,
    save_model: bool = True,
    model_save_dir: str = ".",
):
    """
    Train the model for the specified number of epochs.

    Args:
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        model (torch.nn.Module): The model to be trained.
        hparams (dict, optional): hyperparameters used in training. Default to None,
            equivalent to use default configuration. Options are:
            - lr (float): The learning rate for the optimizer. Default: 0.001
            - batch_size (int): The batch size for the data loaders. Default: 64
            - num_epochs (int): The number of epochs to train the model. Default: 10
            - device (str): The device to use for training. Default: "cuda" if available, otherwise "cpu"
            - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training. Default:
                torch.nn.MSELoss
            - optimizer (torch.optim.Optimizer): The optimizer for the model. Default: torch.optim.Adam
        save_model (bool, optional): Whether to save the best model. Default: True
        model_save_dir (str, optional): The directory to save the best model. Default: "."

    Returns:
        float: The best validation loss achieved during training
    """
    _hparams = get_hparams(default_hparams, hparams)
    save_path = Path(model_save_dir) / "best_model.pth"

    model = model.to(_hparams["device"])
    is_gcn = isinstance(model, GCN)

    data_loader_class = GraphDataLoader if is_gcn else DataLoader
    train_loader = data_loader_class(
        train_dataset, batch_size=_hparams["batch_size"], shuffle=True
    )
    val_loader = data_loader_class(
        val_dataset, batch_size=_hparams["batch_size"], shuffle=False
    )

    # record the best validation loss
    best_val_loss = float("inf")

    for epoch in range(_hparams["num_epochs"]):

        train_one_epoch(train_loader, model, _hparams)
        val_loss = evaluate_model(val_loader, model, _hparams)
        print(
            f"Epoch {epoch+1}/{_hparams['num_epochs']}, Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), save_path)

    return best_val_loss
