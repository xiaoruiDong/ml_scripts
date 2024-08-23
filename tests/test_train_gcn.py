#!/usr/bin/env python3

import pytest

from torch_geometric.loader import DataLoader

from data.dataset import ESOLDataset
from model.gcn import GCN
from train import train_one_epoch, train_model


@pytest.fixture
def dataset():
    return ESOLDataset()


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=True)


@pytest.fixture
def model():
    return GCN(ffnn_output_dim=1)  # ESOL dataset only has one target


def test_train_one_epoch(dataloader, model):
    train_one_epoch(dataloader, model)


def test_train_overfit(dataset, model):
    loss = train_model(
        dataset,
        dataset,
        model,
        save_model=False,
        hparams={"num_epochs": 100, "lr": 1e-3},
    )
    assert loss < 0.1
