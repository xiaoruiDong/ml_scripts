#!/usr/bin/env python3

import pytest
from torch.utils.data import DataLoader

from data.dataset import CSVDataset
from model.mlp import MLP
from train import train_one_epoch, train_model


@pytest.fixture
def dataset():
    return CSVDataset()


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=True)


@pytest.fixture
def model():
    return MLP(4, 10, 1, 2)


def test_train_one_epoch(dataloader, model):
    train_one_epoch(dataloader, model)


def test_train_overfit(dataset, model):
    loss = train_model(
        dataset,
        dataset,
        model,
        save_model=False,
        hparams={"num_epochs": 100, "lr": 0.1},
    )
    assert loss < 0.1
