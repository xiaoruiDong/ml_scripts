#!/bin/bash/env python3

import pytest
from sklearn.model_selection import KFold

from data.dataset import CSVDataset
from model.mlp import MLP
from cv import cross_validation, nested_gridsearch_cv


@pytest.fixture
def dataset():
    return CSVDataset()


@pytest.fixture
def inner_cv():
    return KFold(n_splits=4, shuffle=True, random_state=42)


@pytest.fixture
def outer_cv():
    return KFold(n_splits=3, shuffle=True, random_state=42)


@pytest.mark.parametrize("model_hparams", [None, {"hidden_dim": 5}])
@pytest.mark.parametrize("train_hparams", [None, {"lr": 0.2}])
def test_cross_validation(dataset, inner_cv, model_hparams, train_hparams):
    scores = cross_validation(dataset, MLP, inner_cv, model_hparams, train_hparams)
    assert len(scores) == 4


@pytest.mark.parametrize("model_hparams_grids", [None, {"hidden_dim": [5, 10]}])
@pytest.mark.parametrize("train_hparams_grids", [None, {"lr": [0.1, 0.2]}])
def test_nested_gridsearch_cv(
    dataset, inner_cv, outer_cv, model_hparams_grids, train_hparams_grids
):
    scores, models, hparams = nested_gridsearch_cv(
        dataset, MLP, inner_cv, outer_cv, model_hparams_grids, train_hparams_grids
    )
    assert len(scores) == 3
    assert len(models) == 3
    assert len(hparams) == 3
