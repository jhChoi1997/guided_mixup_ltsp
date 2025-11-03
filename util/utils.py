import csv
import os
import shutil
from typing import Literal

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_csv(save_file_path: str, save_data: list) -> None:
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    save_dir: str,
    is_best: bool,
    filename: str = "checkpoint.pth",
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, os.path.join(save_dir, filename))

    if is_best:
        shutil.copyfile(
            os.path.join(save_dir, filename),
            os.path.join(save_dir, "checkpoint_best.pth"),
        )

    return


def save_initial_checkpoint(
    model: Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint_0.pth")
    if not os.path.exists(checkpoint_path):
        state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
        }
        torch.save(state, checkpoint_path)


def load_checkpoint(
    model: Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    load_dir: str,
    return_mode: Literal["best", "last"] | int = "best",
) -> tuple:
    if isinstance(return_mode, int):
        checkpoint_filename = f"checkpoint_{return_mode}.pth"
    elif return_mode == "last":
        checkpoints = [
            f
            for f in os.listdir(load_dir)
            if f.startswith("checkpoint_") and f.endswith(".pth")
        ]
        (
            checkpoints.remove("checkpoint_best.pth")
            if "checkpoint_best.pth" in checkpoints
            else None
        )
        if checkpoints:
            checkpoint_filename = max(
                checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
            )
        else:
            save_initial_checkpoint(model, optimizer, scheduler, load_dir)
            return model, optimizer, scheduler, 0
    else:
        checkpoint_filename = "checkpoint_best.pth"

    checkpoint_path = os.path.join(load_dir, checkpoint_filename)
    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint {checkpoint_filename} not found. Initializing model from scratch."
        )
        save_initial_checkpoint(model, optimizer, scheduler, load_dir)
        return model, optimizer, scheduler, 0

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler, start_epoch
