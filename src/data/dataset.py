#!/usr/bin/env python3

"""A module for dataset"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


_random_data_path = Path(__file__).absolute().parents[2] / 'data' / 'random_1dfeature_dataset.csv'


class CSVDataset(Dataset):
    """A customized Dataset for loading data from a csv file"""
    def __init__(
        self,
        path: str | Path = _random_data_path,
    ):
        df = pd.read_csv(path)
        self.targets = torch.tensor(df['y'].values, dtype=torch.float32)
        self.inputs = torch.tensor(df.drop('y', axis=1).values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
