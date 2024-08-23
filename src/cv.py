#!/usr/bin/env python3

from pathlib import Path
from typing import Type

from sklearn.model_selection import KFold
import torch

from torch.utils.data import Subset
from train import train_model
from utils import get_hparams, get_hparams_combinations


default_cv_parameters = {
    "n_folds_outer_cv": 5,
    "n_folds_inner_cv": 5,
    "batch_size": 32,
    "num_epochs": 10,
    "random_seed": 42,
}


def cross_validation(
    dataset: torch.utils.data.Dataset,
    model_class: Type[torch.nn.Module],
    cv: KFold,
    model_hparams: dict | None = None,
    train_hparams: dict | None = None,
) -> list:
    """Perform k-fold cross-validation on a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to use.
        model_class (Type[torch.nn.Module]): The model class to use.
        cv (KFold): The cross-validation object.
        model_hparams (dict | None, optional): The model hyperparameters. Defaults to None.
        train_hparams (dict | None, optional): The training hyperparameters. Defaults to None.

    Returns:
        list: A list of scores for each fold.
    """
    scores = []
    for train_idx, val_idx in cv.split(dataset):

        # Initialize the model with the provided hyperparameters
        _model_hparams = get_hparams(model_class.get_default_hparams(), model_hparams)
        model = model_class(**_model_hparams)

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        score = train_model(
            train_dataset,
            val_dataset,
            model,
            hparams=train_hparams,
            save_model=False,
        )
        scores.append(score)

    return scores


def nested_gridsearch_cv(
    dataset: torch.utils.data.Dataset,
    model_class: Type[torch.nn.Module],
    inner_cv: KFold,
    outer_cv: KFold,
    model_hparams_grid: dict | None = None,
    train_hparams_grid: dict | None = None,
    save_model: bool = False,
    model_save_dir: str = ".",
):
    if train_hparams_grid is None:
        train_hparams_grid = {}

    if model_hparams_grid is None:
        model_hparams_grid = {}

    scores = []
    hparams = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(dataset)):

        print("Outer Fold {fold}")
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        best_hparams = ()
        best_score = float("-inf")

        hparams_grid = get_hparams_combinations(model_hparams_grid, train_hparams_grid)

        # Generate grid of hyperparameters
        for model_hparams, train_hparams in hparams_grid:

            print("Current hyperparameters:")
            print(f"Model hyperparameters: {model_hparams}")
            print(f"Training hyperparameters: {train_hparams}")

            # Inner cross-validation
            inner_scores = cross_validation(
                train_dataset, model_class, inner_cv, model_hparams, train_hparams
            )
            avg_score = sum(inner_scores) / len(inner_scores)

            # Update the best hyperparameters
            if avg_score > best_score:
                best_hparams = (model_hparams, train_hparams)
                best_score = avg_score

        # Retrain and evaluate the model with the best hyperparameters
        model_hparams, train_hparams = best_hparams
        _model_hparams = get_hparams(
            model_class.get_default_hparams(),
            model_hparams,
        )
        model = model_class(**_model_hparams)
        score = train_model(
            train_dataset,
            test_dataset,
            model,
            hparams=train_hparams,
            save_model=save_model,
            model_save_dir=Path(model_save_dir) / f"fold{fold}",
        )

        scores.append(score)
        models.append(model)
        hparams.append(best_hparams)

    return scores, models, hparams
