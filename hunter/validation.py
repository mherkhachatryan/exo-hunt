from dataclasses import dataclass


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
