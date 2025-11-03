from typing import Any, Callable

from util.register import (
    DATASET_REGISTRY,
    FIGURE_REGISTRY,
    METRIC_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
)


def get_figure(name: str) -> Callable:
    """Returns a figure function from FIGURE_REGISTRY."""
    if name not in FIGURE_REGISTRY:
        raise ValueError(f"Figure {name} not found in FIGURE_REGISTRY.")
    return FIGURE_REGISTRY[name]


def get_dataset(name: str, *args, **kwargs) -> Any:
    """Returns a dataset class from DATASET_REGISTRY."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not found in DATASET_REGISTRY.")
    return DATASET_REGISTRY[name](*args, **kwargs)


def get_optimizer(name: str, *args, **kwargs) -> Any:
    """Returns an optimizer class from OPTIMIZER_REGISTRY."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer {name} not found in OPTIMIZER_REGISTRY.")
    return OPTIMIZER_REGISTRY[name](*args, **kwargs)


def get_scheduler(name: str, *args, **kwargs) -> Any:
    """Returns a scheduler class from SCHEDULER_REGISTRY."""
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler {name} not found in SCHEDULER_REGISTRY.")
    return SCHEDULER_REGISTRY[name](*args, **kwargs)


def get_metric(name: str) -> Callable:
    """Returns a metric function from METRIC_REGISTRY."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Metric {name} not found in METRIC_REGISTRY.")
    return METRIC_REGISTRY[name]
