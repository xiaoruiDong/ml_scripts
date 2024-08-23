#!/usr/bin/env python3

import itertools
from typing import Sequence, Iterable


def get_hparams(default_hparams: dict, hparams: dict | None = None):
    """
    Get hyperparameters.

    Args:
        default_hparams (dict): Default hyperparameters.
        hparams (dict, optional): Hyperparameters to update. Defaults to None.

    Returns:
        dict: Updated hyperparameters.
    """
    if hparams is None:
        return default_hparams
    else:
        return {**default_hparams, **hparams}


def get_hparams_combinations(
    model_hparams_grid: dict[str, Sequence],
    train_hparams_grid: dict[str, Sequence],
) -> Iterable:
    """
    Get all combinations of hyperparameters.

    Args:
        model_hparams_grid (dict[str, Sequence]): Grid of model hyperparameters.
        train_hparams_grid (dict[str, Sequence]): Grid of training hyperparameters.

    Yields:
        tuple: Model hyperparameters and training hyperparameters.
    """
    if not train_hparams_grid and not model_hparams_grid:
        yield {}, {}

    elif not train_hparams_grid:
        model_keys, model_vals = zip(*model_hparams_grid.items())
        for model_combine in itertools.product(*model_vals):
            model_dict = dict(zip(model_keys, model_combine))
            yield model_dict, {}

    elif not model_hparams_grid:
        train_keys, train_vals = zip(*train_hparams_grid.items())
        for train_combine in itertools.product(*train_vals):
            train_dict = dict(zip(train_keys, train_combine))
            yield {}, train_dict

    else:
        train_keys, train_vals = zip(*train_hparams_grid.items())
        model_keys, model_vals = zip(*model_hparams_grid.items())

        for train_combine in itertools.product(*train_vals):
            train_dict = dict(zip(train_keys, train_combine))

            for model_combine in itertools.product(*model_vals):
                model_dict = dict(zip(model_keys, model_combine))

                yield model_dict, train_dict
