import itertools
import os
from typing import Any, Literal, Optional

import torch
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import dataset as _
import figures as _
import metric as _
import node as _
import util.earlystop as earlystop
import util.utils as utils
from data_utils.data_augmentation import batch_data_augmentation
from model.node_model import NodeModel
from util.registry_utils import (
    get_dataset,
    get_figure,
    get_metric,
    get_optimizer,
    get_scheduler,
)


class DCASE2020Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config: dict[str, Any] = config
        self.device: torch.device = torch.device("cuda", 0)

        self.is_train: bool = config.get("is_train", False)
        self.is_test: bool = config.get("is_test", False)

        self.return_mode: Literal["best", "last"] = config.get("return_mode", "best")
        self.save_path: str = config.get("save_path", "./results/baseline")
        print(f"Save Path: {self.save_path}")
        self.epoch: int = config.get("epoch", 100)
        self.early_stopping: int = config.get("early_stopping", 10)
        self.loss_key: str = config["loss_key"]
        self.eval_key: str = config["eval_key"]
        self.optimizer: dict = config["optimizer"]
        self.scheduler: dict = config["scheduler"]
        self.train_loss_keys: list[str] = config.get("train_loss_keys", [])
        self.data_augmentation: dict = config.get("data_augmentation", {})

        self.model: dict = config["model"]

        self.train_dataset: dict = config["train_dataset"]
        self.validation_dataset: dict | None = config.get("validation_dataset", None)
        self.test_dataset: dict = config["test_dataset"]
        self.train_dataset_for_test: dict = config.get(
            "train_dataset_for_test", self.train_dataset
        )

        self.sample_metrics: list[dict] = config.get("sample_metrics", [])
        self.metrics: list[dict] = config.get("metrics", [])
        self.subset_metrics: list[dict] = config.get("subset_metrics", [])

        self.data_list: list[str] = config.get("data_list", [])

        self.writer: SummaryWriter = SummaryWriter(
            log_dir=os.path.join(self.save_path, "runs")
        )
        self.save_dir: str = os.path.join(self.save_path, "ckpts")
        os.makedirs(self.save_dir, exist_ok=True)

    def upload_sample_metrics(
        self, batch: dict, dataset_mode: str, step: int = 0
    ) -> None:
        assert dataset_mode in ["train", "validation", "test"]

        common_set = set(self.data_list) & set(batch["file_path"])
        if not bool(common_set):
            return

        common_index_list = [
            idx
            for idx, file_path in enumerate(batch["file_path"])
            if file_path in common_set
        ]

        for data_idx in common_index_list:
            if self.mode == "validation":
                dataset_args = self.validation_dataset["args"]  # type: ignore
            else:
                dataset_args = self.test_dataset["args"]
            sample_metrics_list = [
                v for v in self.sample_metrics if dataset_mode in v["dataset_mode"]
            ]
            for sample_metric in sample_metrics_list:
                name_key = sample_metric["name_key"]
                tensorboard_type = sample_metric["tensorboard_type"]
                file_path = batch["file_path"][data_idx]

                tag = f"{name_key}/{self.mode}/{file_path}"
                args = sample_metric.get("args", {})

                if isinstance(sample_metric["data_keys"], str):
                    data = batch[sample_metric["data_keys"]][data_idx]
                else:
                    data = [batch[v][data_idx] for v in sample_metric["data_keys"]]

                if tensorboard_type == "audio":
                    self.writer.add_audio(
                        tag=tag,
                        snd_tensor=data,
                        global_step=step,
                        sample_rate=dataset_args["sampling_rate"],
                    )

                elif tensorboard_type == "image":
                    self.writer.add_image(
                        tag=tag,
                        img_tensor=data,
                        global_step=step,
                    )

                elif tensorboard_type == "figure":
                    figure_fn = get_figure(sample_metric["figure_type"])
                    self.writer.add_figure(
                        tag=tag,
                        figure=figure_fn(*data, **args),
                        global_step=step,
                    )

                elif tensorboard_type == "scalar":
                    self.writer.add_scalar(
                        tag=tag,
                        scalar_value=data,
                        global_step=step,
                    )
                else:
                    raise ValueError(f"Unsupported data type: {type}")

        return

    def upload_metrics(
        self,
        data_for_train_metrics: Optional[dict] = None,
        data_for_val_metrics: Optional[dict] = None,
        data_for_test_metrics: Optional[dict] = None,
        step: int = 0,
        label_dict: dict[str, dict[int, str]] = {},
    ) -> None:
        data_dicts = {
            "train": data_for_train_metrics,
            "validation": data_for_val_metrics,
            "test": data_for_test_metrics,
        }

        for metric in self.metrics:
            name_key = metric["name_key"]
            tensorboard_type = metric["tensorboard_type"]
            dataset_mode = metric["dataset_mode"]
            data_keys = metric["data_keys"]
            args = metric.get("args", {})

            data_values = [None] * len(data_keys)
            if isinstance(dataset_mode, str):
                if data_dicts[dataset_mode] is None:
                    continue
                tag = f"{name_key}/{dataset_mode}"
                for idx, data_key in enumerate(data_keys):
                    if data_key not in data_dicts[dataset_mode]:
                        raise ValueError(f"Key {data_key} not found in metrics_data.")
                    data_values[idx] = data_dicts[dataset_mode][data_key]  # type: ignore
            elif isinstance(dataset_mode, list):
                if any(data_dicts[mode] is None for mode in dataset_mode):
                    continue
                tag = f"{name_key}/{'_'.join(dataset_mode)}"
                for mode in dataset_mode:
                    for idx, data_key in enumerate(data_keys):
                        if data_key not in data_dicts[mode]:
                            raise ValueError(
                                f"Key {data_key} not found in metrics_data."
                            )

                        if data_values[idx] is None:
                            data_values[idx] = data_dicts[mode][data_key]  # type: ignore
                        else:
                            data_values[idx] = torch.cat(  # type: ignore
                                (data_values[idx], data_dicts[mode][data_key]),  # type: ignore
                                dim=0,
                            )
            else:
                raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

            if tensorboard_type == "figure":
                figure_fn = get_figure(metric["figure_type"])

                if args.get("label_key", None) is not None:
                    label_key = args["label_key"]
                    figure_label_dict_list = [label_dict[label_key]]
                else:
                    label_key_list = [key for key in data_keys if key in label_dict]
                    figure_label_dict_list = [label_dict[v] for v in label_key_list]

                args["label_dict_list"] = figure_label_dict_list
                figure = figure_fn(*data_values, **args)
                self.writer.add_figure(tag, figure=figure, global_step=step)
            elif tensorboard_type == "scalar":
                metric_fn = get_metric(metric["metric_type"])
                scalar = metric_fn(*data_values, **args)
                self.writer.add_scalar(tag, scalar_value=scalar, global_step=step)
            elif tensorboard_type == "histogram":
                self.writer.add_histogram(tag, *data_values, global_step=step)
            else:
                raise ValueError(f"Unsupported metric type: {tensorboard_type}")

    def upload_subset_metrics(
        self,
        data_for_subset_metrics: dict,
        step: int,
        label_dict: dict[str, dict[int, str]],
        mode: str,
    ) -> None:
        for subset_metrics in self.subset_metrics:
            for metric in subset_metrics["metrics"]:
                name_key = metric["name_key"]
                tensorboard_type = metric["tensorboard_type"]
                dataset_mode = metric["dataset_mode"]
                data_keys = metric["data_keys"]
                args = metric.get("args", {})

                if isinstance(dataset_mode, str):
                    if dataset_mode != mode:
                        continue
                elif isinstance(dataset_mode, list):
                    if mode != "train_test":
                        continue

                subset_keys = list(subset_metrics["subset_keys"])
                selected_items = {
                    key: list(label_dict[key].keys()) for key in subset_keys
                }
                possible_combinations = list(
                    itertools.product(*selected_items.values())
                )

                num_data = len(data_for_subset_metrics[subset_keys[0]])
                for comb in possible_combinations:
                    indices = torch.arange(num_data)
                    for key, value in zip(subset_keys, comb):
                        indices = indices[
                            torch.where(data_for_subset_metrics[key][indices] == value)[
                                0
                            ]
                        ]
                    if len(indices) == 0:
                        continue

                    label = "_".join(
                        list(label_dict[key].values())[idx]
                        for key, idx in zip(subset_keys, comb)
                    )

                    tag = f"{name_key}/{mode}/{label}"
                    data_values = []
                    for data_key in data_keys:
                        if data_key not in data_for_subset_metrics:
                            raise ValueError(
                                f"Key {data_key} not found in metrics_data."
                            )
                        if isinstance(data_for_subset_metrics[data_key], torch.Tensor):
                            data_values.append(
                                data_for_subset_metrics[data_key][indices]
                            )
                        else:
                            data_values.append(
                                [
                                    data_for_subset_metrics[data_key][i]
                                    for i in indices.tolist()
                                ]
                            )

                    if tensorboard_type == "figure":
                        figure_fn = get_figure(metric["figure_type"])
                        if label_dict is not None:
                            label_key_list = [
                                key
                                for key in metric["data_keys"]
                                if key in label_dict.keys()
                            ]
                            figure_label_dict_list = [
                                label_dict[v] for v in label_key_list
                            ]
                        args["label_dict_list"] = figure_label_dict_list
                        args["title"] = label
                        figure = figure_fn(*data_values, **args)
                        self.writer.add_figure(tag, figure=figure, global_step=step)
                    elif tensorboard_type == "scalar":
                        metric_fn = get_metric(metric["metric_type"])
                        scalar = metric_fn(*data_values, **args)
                        self.writer.add_scalar(
                            tag, scalar_value=scalar, global_step=step
                        )
                    elif tensorboard_type == "histogram":
                        self.writer.add_histogram(tag, *data_values, global_step=step)
                    else:
                        raise ValueError(
                            f"Unsupported metric type: {metric['tensorboard_type']}"
                        )

        return

    def append_keys_to_dict(
        self,
        metrics_list: list[dict],
        data_keys: dict,
    ) -> dict[str, torch.Tensor | list[str] | None]:
        for metric in metrics_list:
            if isinstance(metric["data_keys"], str):
                if not metric["data_keys"] in data_keys:
                    data_keys[metric["data_keys"]] = None
            else:
                for key in metric["data_keys"]:
                    if not key in data_keys:
                        data_keys[key] = None
        return data_keys

    def append_data_from_batch(
        self,
        data_dict: dict,
        batch: dict[str, torch.Tensor | list[str]],
    ) -> dict[str, torch.Tensor | list[str] | None]:
        for key in data_dict:
            if data_dict[key] is None:
                if isinstance(batch[key], torch.Tensor):
                    data_dict[key] = batch[key].detach().cpu()  # type: ignore
                else:
                    data_dict[key] = batch[key]
            else:
                if isinstance(batch[key], torch.Tensor):
                    try:
                        data_dict[key] = torch.cat(
                            (data_dict[key], batch[key].detach().cpu()), dim=0  # type: ignore
                        )
                    except:
                        raise ValueError(
                            f"Error in appending data from batch. Key: {key}"
                        )
                else:
                    data_dict[key] = data_dict[key] + batch[key]  # type: ignore
        return data_dict

    def train_test(self) -> None:
        if self.is_train:
            self.mode = "train"
            self.train()
        if self.is_test:
            self.mode = "test"
            self.test()

    def train(self) -> None:
        train_dataset = get_dataset(
            self.train_dataset["dataset_name"],
            self.train_dataset["args"],
        )
        if self.validation_dataset is not None:
            val_dataset = get_dataset(
                self.validation_dataset["dataset_name"],
                self.validation_dataset["args"],
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_dataset["batch_size"],
            shuffle=True,
            num_workers=self.train_dataset["num_workers"],
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
        if self.validation_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.validation_dataset["batch_size"],
                shuffle=False,
                num_workers=self.validation_dataset["num_workers"],
                collate_fn=val_dataset.collate_fn,
                pin_memory=True,
                persistent_workers=True,
            )

        model = NodeModel(self.model).to(self.device)
        early_stop = earlystop.EarlyStop(patience=self.early_stopping)

        torchinfo.summary(model)

        optim = get_optimizer(
            self.optimizer["name"],
            model.parameters(),
            **self.optimizer["args"],
        )
        scheduler = get_scheduler(
            self.scheduler["name"], optim, **self.scheduler["args"]
        )

        model, optim, scheduler, start_epoch = utils.load_checkpoint(
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            load_dir=self.save_dir,
            return_mode=self.return_mode,
        )
        self.writer.add_text("config", str(self.config))

        for epoch in range(start_epoch + 1, self.epoch + 1):
            self.mode = "train"
            p_bar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch}/{self.epoch}",
                dynamic_ncols=True,
            )

            model.train()
            model.is_train = True
            train_losses = {}

            train_label_dict = train_dataset.get_label_dict()
            data_for_train_metrics = {key: None for key in train_label_dict.keys()}
            train_metrics = [v for v in self.metrics if "train" in v["dataset_mode"]]
            data_for_train_metrics = self.append_keys_to_dict(
                train_metrics, data_for_train_metrics
            )
            for subset_metrics in self.subset_metrics:
                metrics = [v for v in subset_metrics["metrics"]]
                data_for_train_metrics = self.append_keys_to_dict(
                    metrics, data_for_train_metrics
                )
            for i, param_group in enumerate(optim.param_groups):
                self.writer.add_scalar(
                    f"Learning Rate/param_group_{i}", param_group["lr"], epoch
                )

            for batch in p_bar:
                batch = batch_data_augmentation(self.data_augmentation, batch)
                batch = model(batch)
                loss = batch[self.loss_key]

                optim.zero_grad()
                loss.backward()
                optim.step()

                p_bar_log_dict = {}
                for train_loss_key in self.train_loss_keys:
                    if train_loss_key not in train_losses:
                        train_losses[train_loss_key] = 0.0
                    value = batch[train_loss_key].mean().item()
                    train_losses[train_loss_key] += value
                    p_bar_log_dict[train_loss_key] = f"{value:.2e}"

                if isinstance(p_bar, tqdm):
                    p_bar.set_postfix(p_bar_log_dict)

                self.upload_sample_metrics(
                    batch=batch, dataset_mode=self.mode, step=epoch
                )

                data_for_train_metrics = self.append_data_from_batch(
                    data_for_train_metrics, batch
                )

                del batch

            self.upload_metrics(
                data_for_train_metrics=data_for_train_metrics,
                step=epoch,
                label_dict=train_label_dict,
            )
            self.upload_subset_metrics(
                data_for_train_metrics, epoch, train_label_dict, "train"
            )
            for data_key in train_losses:
                train_losses[data_key] /= len(train_loader)
                self.writer.add_scalar(
                    f"{data_key}/{self.mode}", train_losses[data_key], epoch
                )
            if self.validation_dataset is None:
                early_stopping_score = train_losses[self.eval_key]
            else:
                self.mode = "validation"
                model.eval()
                model.is_train = False
                val_losses = {}

                val_label_dict = val_dataset.get_label_dict()
                data_for_val_metrics = {key: None for key in val_label_dict.keys()}
                val_metrics = [
                    v for v in self.metrics if "validation" in v["dataset_mode"]
                ]
                data_for_val_metrics = self.append_keys_to_dict(
                    val_metrics, data_for_val_metrics
                )
                for subset_metrics in self.subset_metrics:
                    metrics = [v for v in subset_metrics["metrics"]]
                    data_for_val_metrics = self.append_keys_to_dict(
                        metrics, data_for_val_metrics
                    )

                for batch in val_loader:
                    batch = batch_data_augmentation(
                        self.data_augmentation, batch, is_test=True
                    )
                    with torch.no_grad():
                        batch = model(batch)

                    for train_loss_key in self.train_loss_keys:
                        if train_loss_key not in val_losses:
                            val_losses[train_loss_key] = 0.0
                        val_losses[train_loss_key] += batch[train_loss_key].item()

                    self.upload_sample_metrics(
                        batch=batch, dataset_mode=self.mode, step=epoch
                    )
                    data_for_val_metrics = self.append_data_from_batch(
                        data_for_val_metrics, batch
                    )
                    del batch

                for data_key in val_losses:
                    val_losses[data_key] /= len(val_loader)
                    self.writer.add_scalar(
                        f"{data_key}/{self.mode}", val_losses[data_key], epoch
                    )
                early_stopping_score: float = val_losses[self.eval_key]

                self.upload_metrics(
                    data_for_val_metrics=data_for_val_metrics,
                    step=epoch,
                    label_dict=val_label_dict,
                )
                self.upload_subset_metrics(
                    data_for_val_metrics, epoch, train_label_dict, "validation"
                )
            utils.save_checkpoint(
                model=model,
                optimizer=optim,
                scheduler=scheduler,
                epoch=epoch,
                save_dir=self.save_dir,
                is_best=early_stop.is_best_score(early_stopping_score),
                filename=f"checkpoint_{epoch}.pth",
            )
            if self.scheduler["name"] == "ReduceLROnPlateau":
                scheduler.step(early_stopping_score)
            else:
                scheduler.step()
            if early_stop(early_stopping_score):
                print(
                    f"Early stopping at epoch {epoch}, Save Path: {self.config['save_path']},  Best score: {early_stop.best_score:.4f}"
                )
                break
        print(
            f"Save Path: {self.config['save_path']},  Best score: {early_stop.best_score:.4f}"
        )

        return

    def test(self, epoch: int = 0, model=None) -> None:
        self.mode = "test"
        train_dataset = get_dataset(
            self.train_dataset_for_test["dataset_name"],
            self.train_dataset_for_test["args"],
        )
        test_dataset = get_dataset(
            self.test_dataset["dataset_name"],
            self.test_dataset["args"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_dataset["batch_size"],
            shuffle=False,
            num_workers=self.train_dataset["num_workers"],
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.test_dataset["batch_size"],
            shuffle=False,
            num_workers=self.test_dataset["num_workers"],
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
        if model is None:
            model = NodeModel(self.model).to(self.device)

            torchinfo.summary(model)
            model, _, _, epoch = utils.load_checkpoint(
                model=model,
                optimizer=None,
                scheduler=None,
                load_dir=self.save_dir,
                return_mode=self.return_mode,
            )
        model.eval()
        model.is_train = False
        test_losses = {}

        train_label_dict = train_dataset.get_label_dict()
        data_for_train_metrics = {key: None for key in train_label_dict.keys()}
        train_metrics = [v for v in self.metrics if "train" in v["dataset_mode"]]
        data_for_train_metrics = self.append_keys_to_dict(
            train_metrics, data_for_train_metrics
        )
        for subset_metrics in self.subset_metrics:
            metrics = [v for v in subset_metrics["metrics"]]
            data_for_train_metrics = self.append_keys_to_dict(
                metrics, data_for_train_metrics
            )

        test_label_dict = test_dataset.get_label_dict()
        data_for_test_metrics = {key: None for key in test_label_dict.keys()}
        test_metrics = [v for v in self.metrics if "test" in v["dataset_mode"]]
        data_for_test_metrics = self.append_keys_to_dict(
            test_metrics, data_for_test_metrics
        )
        for subset_metrics in self.subset_metrics:
            metrics = [v for v in subset_metrics["metrics"]]
            data_for_test_metrics = self.append_keys_to_dict(
                metrics, data_for_test_metrics
            )

        data_for_train_test_subset_metrics = {
            key: None for key in train_label_dict.keys()
        }
        for subset_metrics in self.subset_metrics:
            metrics = [v for v in subset_metrics["metrics"]]
            data_for_train_test_subset_metrics = self.append_keys_to_dict(
                metrics, data_for_train_test_subset_metrics
            )
        for batch in train_loader:
            batch = batch_data_augmentation(self.data_augmentation, batch, is_test=True)
            with torch.no_grad():
                batch = model(batch)
            self.upload_sample_metrics(batch=batch, dataset_mode=self.mode, step=epoch)
            data_for_train_metrics = self.append_data_from_batch(
                data_for_train_metrics, batch
            )
            data_for_train_test_subset_metrics = self.append_data_from_batch(
                data_for_train_test_subset_metrics, batch
            )
            del batch

        for batch in test_loader:
            batch = batch_data_augmentation(self.data_augmentation, batch, is_test=True)
            with torch.no_grad():
                batch = model(batch)

            for train_loss_key in self.train_loss_keys:
                if train_loss_key not in test_losses:
                    test_losses[train_loss_key] = 0.0
                test_losses[train_loss_key] += batch[train_loss_key].mean().item()

            self.upload_sample_metrics(batch=batch, dataset_mode=self.mode, step=epoch)
            data_for_test_metrics = self.append_data_from_batch(
                data_for_test_metrics, batch
            )
            data_for_train_test_subset_metrics = self.append_data_from_batch(
                data_for_train_test_subset_metrics, batch
            )
            del batch

        self.upload_metrics(
            data_for_train_metrics=data_for_train_metrics,
            data_for_test_metrics=data_for_test_metrics,
            step=epoch,
            label_dict=test_label_dict,
        )

        self.upload_subset_metrics(
            data_for_train_test_subset_metrics, epoch, train_label_dict, "train_test"
        )

        self.upload_subset_metrics(
            data_for_train_metrics, epoch, train_label_dict, "train"
        )

        self.upload_subset_metrics(
            data_for_test_metrics, epoch, test_label_dict, "test"
        )

        for data_key in test_losses:
            test_losses[data_key] /= len(test_loader)
            self.writer.add_scalar(
                f"{data_key}/{self.mode}", test_losses[data_key], epoch
            )
