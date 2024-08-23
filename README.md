# ml_scripts

This is a toy repository of machine learning related task. It includes a basic framework for training and evaluating machine learning models with a nested cross validation approach. Currently, two type of models, multilayer perceptron regressor (MLP) and graph convolutional neural network (GCN) regressor are supported.

# Installation
First, create a virtual environment or a conda environment (an example is provided as follow)
```
conda create -n ml_env python=3.10
conda activate ml_env
```
Then, clone this repository
```
git clone https://github.com/xiaoruiDong/ml_scripts
```
Install the package from source
```
cd ml_scripts; pip install .
```

# Examples
Examples of using the `ml-scripts` repo are included in the `examples/` for both models.