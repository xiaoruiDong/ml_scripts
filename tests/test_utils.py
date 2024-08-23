#!/usr/bin/env python3

import pytest

from utils import get_hparams_combinations
from itertools import combinations
from itertools import combinations
from itertools import combinations


@pytest.fixture
def train_hparams_grid():
    return {
        "batch_size": [32, 64],
        "lr": [1e-1, 1e-2],
    }


@pytest.fixture
def model_hparams_grid():
    return {
        "hidden_size": [32, 64],
        "dropout": [0.1, 0.2],
    }


def test_get_hparams_combinations(train_hparams_grid, model_hparams_grid):
    combinations = list(
        get_hparams_combinations(train_hparams_grid, model_hparams_grid)
    )
    assert len(combinations) == 16

    for batch_size in [32, 64]:
        for lr in [1e-1, 1e-2]:
            for hidden_size in [32, 64]:
                for dropout in [0.1, 0.2]:
                    assert (
                        {
                            "batch_size": batch_size,
                            "lr": lr,
                        },
                        {
                            "hidden_size": hidden_size,
                            "dropout": dropout,
                        },
                    ) in combinations
