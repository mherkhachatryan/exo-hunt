import yaml
from pathlib import Path

from hunter.validation import RunConfigs
from hunter.preprocessing import ExoHuntDataset
from hunter.model import ExoHuntModel

TEST_CONFIG_TEXT = """
save_path: "./logs/RandomForest"
run_mode: "train" 
saved_model_path: ""
random_state: 42
data_path: "data/" # Path to the Dataset directory
model: "RandomForest" 

# Below are the parameters for RandomForest, for GradientBoosting they should be adjusted accordingly
model_params:
  criterion: "gini"

# split size for train/val split, defines val size
split_size: 0.3"""


def test_training(tmpdir):
    configs_dir = Path(tmpdir)
    sample_config = configs_dir / "sample_config.yml"
    sample_config.write_text(TEST_CONFIG_TEXT)

    with open(sample_config, "r") as f:
        _configs = yaml.load(f, yaml.FullLoader)

    configs = RunConfigs(**_configs)

    dataset = ExoHuntDataset(configs.data_path, configs.random_state, pca_components=2)
    model = ExoHuntModel(configs.model, configs.model_params, configs.random_state)

    X_train, y_train = dataset._load_mock_data(mode="train", nrows=50)

    model.train(X_train, y_train)

    model.eval(X_train, y_train)
