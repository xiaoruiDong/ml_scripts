#!/usr/bin/env python3

"""A naive dataset generator for a relationship of y = [1.5, 0.5, -0.5, 1.5] $\cdot$ x with gaussian noise"""

import numpy as np
import pandas as pd


def eval(x):
    """A hardcoded relationship"""
    coeff = np.array([[1.5, 0.5, -0.5, 1.5]])
    return x @ coeff.T


def generate_dataset_with_noise(
    num_samples: int = 100,
    noise_level: float = 0.1,
) -> tuple:
    """
    Generate a dataset with the relationship defined in `eval` and
    a noise following a gaussian distribution N(0, `noise_level`).

    Args:
        num_samples (int, optional): Number of samples to generate. Defaults to 100.
        noise_level (float, optional): Standard deviation of the gaussian noise. Defaults to 0.1.

    Returns:
        np.array: A tuple containing the input samples `X` and the corresponding output samples `y`.
    """
    np.random.seed(42)
    X = np.random.uniform(-10, 10, size=(num_samples, 4))
    y = eval(X) + np.random.normal(0, noise_level, size=(num_samples, 1))
    return X, y


def main():
    X, y = generate_dataset_with_noise()
    pd.DataFrame(
        np.hstack((X, y)),
        columns=[f"x{i}" for i in range(4)] + ["y"],
    ).to_csv(
        "random_1dfeature_dataset.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
