#!/usr/bin/env python3


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
