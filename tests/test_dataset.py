#!/usr/bin/env python3

import pytest
import pandas as pd

import numpy as np
from data.dataset import CSVDataset, _random_data_path

def test_dataset():

    dataset = CSVDataset()
    df = pd.read_csv(_random_data_path)

    assert len(dataset) == len(df)
    # randomly select 5 entries for comparison
    indices = np.random.choice(len(dataset), 5, replace=False)
    for i in indices:
        np.testing.assert_array_almost_equal(
            dataset[i][0].numpy(),
            df.iloc[i, :-1].values,
        )
