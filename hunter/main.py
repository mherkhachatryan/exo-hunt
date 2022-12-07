from dataclasses import dataclass

import yaml

from model import ExoHuntModel
from preprocessing import ExoHuntDataset


@dataclass
class RunConfigs:
    save_path: str
    saved_model_path: str
    run_mode: str
    random_state: int
    data_path: str
    model: str
    model_params: dict
    split_size: float


with open("hunter/configs.yaml", "r") as f:
    _configs = yaml.load(f, yaml.FullLoader)

configs = RunConfigs(**_configs)
dataset = ExoHuntDataset(configs.data_path, configs.random_state)

if configs.run_mode == "train":
    model = ExoHuntModel(configs.model, configs.model_params, configs.random_state)
    X_train, X_valid, y_train, y_valid = dataset.load_train_val_data(split_size=configs.split_size,
                                                                     random_state=configs.random_state)
    model.train(X_train, y_train)
    model.eval(X_valid, y_valid)
    model.save(configs.save_path)

elif configs.run_mode == "test":
    model = ExoHuntModel(model_path=configs.saved_model_path)
    X_test, y_test = dataset.load_test_data()
    model.eval(X_test, y_test)
    model.save(configs.save_path, save_eval_only=True)