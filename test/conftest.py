import pytest
import yaml
from hunter.preprocessing import ExoHuntDataset
from hunter.validation import RunConfigs

with open("hunter/configs.yaml", "r") as f:
    _config = yaml.load(f, yaml.FullLoader)

configs = RunConfigs(**_config)

dataset = ExoHuntDataset(configs.data_path, configs.random_state)


@pytest.fixture(scope="session")
def pipeline_inputs():
    dataset = ExoHuntDataset(configs.data_path, configs.random_state)
    X_train, X_valid, y_train, y_valid = dataset.load_train_val_data(split_size=configs.split_size,
                                                                     random_state=configs.random_state)
    return X_train, X_valid, y_train, y_valid


@pytest.fixture()
def raw_training_data():
    # For larger datasets, here we would use a testing sub-sample.
    X_train, y_train = dataset.load_train_data()
    return X_train, y_train


@pytest.fixture()
def sample_test_data():
    return dataset.load_test_data()
