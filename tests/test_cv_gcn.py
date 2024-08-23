#!/bin/bash/env python3

import pytest
from sklearn.model_selection import KFold

from data.dataset import ESOLDataset
from model.gcn import GCN
from cv import cross_validation, nested_gridsearch_cv


@pytest.fixture
def dataset():
    return ESOLDataset()


@pytest.fixture
def inner_cv():
    return KFold(n_splits=4, shuffle=True, random_state=42)


@pytest.fixture
def outer_cv():
    return KFold(n_splits=3, shuffle=True, random_state=42)


@pytest.mark.parametrize("model_hparams", [None, {"gcn_hidden_dim": 64}])
@pytest.mark.parametrize("train_hparams", [None, {"lr": 0.2, "num_epochs": 5}])
def test_cross_validation(dataset, inner_cv, model_hparams, train_hparams):
    scores = cross_validation(
        dataset,
        GCN,
        inner_cv,
        model_hparams,
        train_hparams,
    )
    assert len(scores) == 4


@pytest.mark.parametrize("model_hparams_grids", [None, {"gcn_hidden_dim": [5, 10]}])
@pytest.mark.parametrize("train_hparams_grids", [None, {"num_epochs": [1, 3]}])
def test_nested_gridsearch_cv(
    dataset, inner_cv, outer_cv, model_hparams_grids, train_hparams_grids
):
    scores, models, hparams = nested_gridsearch_cv(
        dataset,
        GCN,
        inner_cv,
        outer_cv,
        model_hparams_grids,
        train_hparams_grids,
    )
    assert len(scores) == 3
    assert len(models) == 3
    assert len(hparams) == 3
