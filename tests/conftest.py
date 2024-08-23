#!/usr/bin/env python3

import pytest
import torch


@pytest.fixture(autouse=True)
def set_pytorch_random_seed():
    torch.manual_seed(42)
