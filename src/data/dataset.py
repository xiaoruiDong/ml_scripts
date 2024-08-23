#!/usr/bin/env python3

"""A module for dataset"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from torch_geometric.datasets import MoleculeNet


_random_data_path = (
    Path(__file__).absolute().parents[2] / "data" / "random_1dfeature_dataset.csv"
)

_graph_data_path = Path(__file__).absolute().parents[2] / "data" / "esol"


class CSVDataset(Dataset):
    """A customized Dataset for loading data from a csv file"""

    def __init__(
        self,
        path: str | Path = _random_data_path,
    ):
        df = pd.read_csv(path)
        self.targets = torch.tensor(df[["y"]].values, dtype=torch.float32)
        self.inputs = torch.tensor(df.drop("y", axis=1).values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ESOLDataset(MoleculeNet):
    """A customized Dataset for loading ESOL dataset, so that
    node features are float"""

    def __init__(self, root: str | Path = _graph_data_path):
        super().__init__(root=root, name="esol")

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.x = data.x.float()  # force the feature to be float for compatibility
        return data


# def load_esol_dataset():
#     """
#     A helper function to load ESOL dataset.
#     """
#     dataset = MoleculeNet(root=_graph_data_path, name="esol")
#     for data in dataset:
#         data.x = data.x.float()  # force the feature to be float for compatibility
#     return dataset
