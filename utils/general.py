import os

import numpy as np
import torch
from hyperpyyaml import dump_hyperpyyaml, load_hyperpyyaml


def load_config(config_path, dataset):
    # Placeholder values
    project_root = os.getcwd()
    runtime_values = {
        "data_dir": os.path.abspath(os.path.join(project_root, "data", dataset)),
        "models_dir": os.path.abspath(os.path.join(project_root, "models")),
        "results_dir": os.path.abspath(os.path.join(project_root, "results")),
    }
    print(runtime_values["data_dir"])

    with open(config_path, "r") as file:
        return load_hyperpyyaml(file, overrides=runtime_values)


def save_config(config_path):
    with open(config_path, "w") as file:
        return dump_hyperpyyaml(file)


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Seeds set to: {seed}")


def get_transforms(config, set):
    return config["val_transform"] if (set == "test") else config["train_transform"]
