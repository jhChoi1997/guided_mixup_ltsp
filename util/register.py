import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

NODE_REGISTRY = {
    "nn.Linear": nn.Linear,
    "nn.ReLU": nn.ReLU,
    "nn.CrossEntropyLoss": nn.CrossEntropyLoss,
    "nn.Softmax": nn.Softmax,
    "nn.NLLLoss": nn.NLLLoss,
    "nn.MSELoss": nn.MSELoss,
    "nn.Sigmoid": nn.Sigmoid,
    "nn.GELU": nn.GELU,
    "nn.PReLU": nn.PReLU,
    "nn.Conv2d": nn.Conv2d,
    "nn.Conv1d": nn.Conv1d,
}
OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}
SCHEDULER_REGISTRY = {
    "cosine_annealing": lr_scheduler.CosineAnnealingLR,
    "onecycle_lr": lr_scheduler.OneCycleLR,
    "cosine_annealing_warm_restarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
}
DATASET_REGISTRY = {}
METRIC_REGISTRY = {}
FIGURE_REGISTRY = {}


def register_node(name: str):
    """Registers a network node in NODE_REGISTRY."""

    def decorator(cls):
        if name in NODE_REGISTRY:
            raise ValueError(f"Node {name} is already registered.")
        NODE_REGISTRY[name] = cls
        return cls

    return decorator


def get_node(name: str, **kwargs):
    """Returns a network node from NODE_REGISTRY."""
    if name not in NODE_REGISTRY:
        raise ValueError(f"Node {name} not found in NODE_REGISTRY.")
    return NODE_REGISTRY[name](**kwargs)


def register_dataset(name: str):
    """Registers a dataset class with a given name."""

    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset {name} is already registered.")
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def register_metric(name: str):
    """Registers a metric class with a given name."""

    def decorator(cls):
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric {name} is already registered.")
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def register_figure(name: str):
    """Registers a figure function with a given name."""

    def decorator(cls):
        if name in FIGURE_REGISTRY:
            raise ValueError(f"Figure {name} is already registered.")
        FIGURE_REGISTRY[name] = cls
        return cls

    return decorator
