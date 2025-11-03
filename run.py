import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


import argparse
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

from trainer.dcase2020_ltsp_trainer import DCASE2020LTSPTrainer
from trainer.dcase2020_trainer import DCASE2020Trainer


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a YAML configuration")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/baseline.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Enable PyTorch profiler for performance debugging.",
    )
    parser.add_argument("--trainer", type=str)
    parser.add_argument("--is_train", type=bool)
    parser.add_argument("--is_test", type=bool)
    parser.add_argument("--test_every_epoch", type=bool)
    parser.add_argument("--return_mode", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--early_stopping", type=int)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--seed", type=int)

    return parser.parse_args()


def select_trainer(config: dict) -> Any:
    trainer_name = config.get("trainer", "base_trainer")

    if trainer_name == "dcase2020_trainer":
        return DCASE2020Trainer(config)
    elif trainer_name == "dcase2020_ltsp_trainer":
        return DCASE2020LTSPTrainer(config)
    else:
        raise ValueError(f"Invalid trainer: {trainer_name}")


def main() -> None:
    args = parse_args()
    try:
        config = load_yaml(args.config)
        logging.info("YAML configuration loaded successfully.")
        logging.info(f"Config: {config}")
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        return

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    set_seed(config.get("seed", 42), config.get("deterministic", True))

    gpus = config.get("gpus", [])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])

    trainer = select_trainer(config)
    trainer.train_test()


if __name__ == "__main__":
    main()
