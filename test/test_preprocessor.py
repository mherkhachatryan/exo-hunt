import pytest
import numpy as np
import yaml
from hunter.preprocessing import ExoHuntDataset
from hunter.validation import RunConfigs

with open("hunter/configs.yaml", "r") as f:
    _config = yaml.load(f, yaml.FullLoader)

configs = RunConfigs(**_config)
dataset = ExoHuntDataset(configs.data_path, configs.random_state)


def test_binary_classification():
    _, y = dataset.load_train_data()

    unique_values = y.unique()
    check_if_binary = np.isin(unique_values, [0, 1])

    assert len(unique_values) == 2, "There should be only 2 classes"
    assert np.all(check_if_binary), "Binary classification should have only 0's and 1'"


def test_oversample():
    _, y = dataset.load_train_data()

    class_counts = y.value_counts()

    assert class_counts[0] == class_counts[1], "Train Data is not oversampled"


def test_pca():
    pca_components = 24
    dataset = ExoHuntDataset(configs.data_path, configs.random_state)
    X, _ = dataset.load_train_data()

    feature_size = X.shape[1]

    assert feature_size == pca_components, "PCA is not performed on dataset"
