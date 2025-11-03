import csv
import os
import random
import re
from typing import Any

import torch
import torchaudio

from util import utils
from util.register import register_dataset


@register_dataset("new_dcase2020_dataset")
class NewDCASE2020Dataset(torch.utils.data.Dataset):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config: dict[str, Any] = config
        self.dataset_mode: str = config.get("dataset_mode", "train")
        assert self.dataset_mode in ["train", "validation", "test"]
        self.train_val_split: float = config.get("train_val_split", 1.0)
        self.data_path: str = config.get("data_path", "/home/data/DCASE2020")
        self.sampling_rate: int = config.get("sampling_rate", 16000)
        self.duration: float = config.get("duration", 10.0)
        self.machine_list: list[str] = config.get(
            "machine_list", ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]
        )
        self.n_fft: int = config.get("n_fft", 1024)
        self.win_length: int = config.get("win_length", 1024)
        self.hop_length: int = config.get("hop_length", 512)
        self.n_mels: int = config.get("n_mels", 128)
        self.frames: int = config.get("frames", 313)
        self.power: float = config.get("power", 2.0)
        self.target_length: int = config.get("target_length", 1024)
        self.norm_mean: float = config.get("norm_mean", -4.268)
        self.norm_std: float = config.get("norm_std", 4.569)
        self.snr_db: int = config.get("snr_db", 10)

        self.machine_id_count = {
            "ToyCar": 7,
            "ToyConveyor": 6,
            "fan": 7,
            "pump": 7,
            "slider": 7,
            "valve": 7,
        }
        self.total_machine_type_list: list[str] = [
            "ToyCar",
            "ToyConveyor",
            "fan",
            "pump",
            "slider",
            "valve",
        ]

        self.total_classes: int = 41
        self.file_list: list[str] = self.get_file_list(self.dataset_mode)

        if not self.dataset_mode == "test":
            for machine_type in self.machine_list:
                self.split_train_val(machine_type)

        self.machine_type_label_dict = self.get_machine_type_label_dict(
            self.total_machine_type_list
        )
        self.machine_id_label_dict = self.get_machine_id_label_dict(
            self.total_machine_type_list
        )
        self.normal_anomaly_label_dict = self.get_normal_anomaly_label_dict()
        self.dev_eval_label_dict = {0: "dev", 1: "eval"}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_path = self.file_list[idx]

        y, y_pad, sr = self.load_wav(file_path)

        log_mel = self.get_log_mel_spec(y_pad)
        machine_id_label = self.get_machine_id_label(file_path)
        one_hot = self.int_to_one_hot(machine_id_label, self.total_classes)
        anomaly_label = self.get_anomaly_label(file_path)
        machine_type_label = self.get_machine_type_label(file_path)
        machine_one_hot = self.int_to_one_hot(machine_type_label, 6)
        id_str = self.get_id_str(file_path)
        machine_type_str = self.get_machine_type_str(file_path)
        machine_id_str = self.get_machine_id_label_str(machine_type_str, id_str)
        train_test_label = self.get_train_test_label(file_path)
        dev_eval_label = self.get_dev_eval_label(file_path)

        output_dict = {
            "file_path": file_path,
            "y": y_pad,
            "sr": sr,
            "log_mel": log_mel,
            "machine_type_str": machine_type_str,
            "machine_type_label": machine_type_label,
            "machine_id_label": machine_id_label,
            "machine_id_str": machine_id_str,
            "one_hot": one_hot,
            "machine_one_hot": machine_one_hot,
            "id_str": id_str,
            "anomaly_label": anomaly_label,
            "train_test": train_test_label,
            "dev_eval_label": dev_eval_label,
        }
        return output_dict

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | list[str]]:
        collated = {
            "file_path": [data["file_path"] for data in batch],
            "input": torch.stack([data["y"] for data in batch]),
            "sr": torch.tensor([data["sr"] for data in batch]),
            "log_mel": torch.stack([data["log_mel"] for data in batch]),
            "machine_type_str": [data["machine_type_str"] for data in batch],
            "machine_type_label": torch.tensor(
                [data["machine_type_label"] for data in batch]
            ),
            "machine_id_str": [data["machine_id_str"] for data in batch],
            "machine_id_label": torch.tensor(
                [data["machine_id_label"] for data in batch]
            ),
            "id_str": [data["id_str"] for data in batch],
            "one_hot": torch.stack([data["one_hot"] for data in batch]),
            "machine_one_hot": torch.stack([data["machine_one_hot"] for data in batch]),
            "anomaly_label": torch.tensor([data["anomaly_label"] for data in batch]),
            "train_test": torch.tensor([data["train_test"] for data in batch]),
            "dev_eval_label": torch.tensor([data["dev_eval_label"] for data in batch]),
            "one": 1.0,
        }
        return collated

    def split_train_val(self, machine: str) -> None:
        save_path = os.path.join(
            self.data_path, machine, f"train_val_{machine}_{self.train_val_split}.csv"
        )
        if os.path.exists(save_path):
            return
        else:
            data_dir = os.path.join(self.data_path, machine, "dev_data", "train")
            data_list = os.listdir(data_dir)
            data_idx = list(range(len(data_list)))
            train_idx = sorted(
                random.sample(data_idx, int(self.train_val_split * len(data_idx)))
            )
            csv_list = [
                [
                    os.path.join(data_dir, data_list[idx]),
                    "train" if idx in train_idx else "validation",
                ]
                for idx in data_idx
            ]
            utils.save_csv(save_path, csv_list)
            return

    def get_file_list(self, mode: str) -> list[str]:
        data_list = []
        if mode == "test":
            dev_eval_list = ["dev_data", "eval_data"]
            for dev_mode in dev_eval_list:
                for machine in self.machine_list:
                    file_dir = os.path.join(self.data_path, machine, dev_mode, mode)
                    file_list = os.listdir(file_dir)
                    file_list.sort()
                    file_path_list = [
                        os.path.join(file_dir, file) for file in file_list
                    ]
                    data_list.extend(file_path_list)

        else:
            dev_mode = "dev_data"
            for machine in self.machine_list:
                split_csv = os.path.join(
                    self.data_path,
                    machine,
                    f"train_val_{machine}_{self.train_val_split}.csv",
                )

                with open(split_csv, "r") as f:
                    rdr = csv.reader(f)
                    for line in rdr:
                        if line[1] == mode:
                            data_list.append(line[0])
                        else:
                            continue

        return data_list

    def int_to_one_hot(self, value: int, num_classes: int) -> torch.Tensor:
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[value] = 1.0
        return one_hot

    def load_wav(self, file_path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        y, sr = torchaudio.load(file_path)

        n_frames = y.shape[1]
        target_length = int(self.duration * self.sampling_rate)
        diff = target_length - n_frames
        if diff > 0:
            m = torch.nn.ZeroPad1d((0, diff))
            y_pad = m(y)
        elif diff < 0:
            y_pad = y[:, 0:target_length]
        else:
            y_pad = y

        return y, y_pad, sr

    def get_log_mel_spec(self, y: torch.Tensor) -> torch.Tensor:
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )(y)

        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        n_frames = log_mel_spec.shape[2]
        diff = self.frames - n_frames
        if diff > 0:
            m = torch.nn.ZeroPad2d((0, diff, 0, 0))
            log_mel_spec = m(log_mel_spec)
        elif diff < 0:
            log_mel_spec = log_mel_spec[..., 0 : self.frames]
        return log_mel_spec

    def get_machine_id_label(self, file_path: str) -> int:
        machine_type = file_path.split("/")[-4]
        machine_id_list = self.get_machine_id_list(machine_type)
        id_str = self.get_id_str(file_path)

        machine_local_id = machine_id_list.index(id_str)

        start_index = sum(
            self.machine_id_count[m] for m in self.machine_id_count if m < machine_type
        )
        machine_id = start_index + machine_local_id
        return machine_id

    def get_anomaly_label(self, file_path: str) -> int:
        if "normal" in file_path:
            return 0
        else:
            return 1

    def get_id_str(self, file_path: str) -> str:
        match = re.search(r"id_\d+", file_path)
        if match:
            id_str = match.group()
        else:
            raise ValueError(f"No machine ID found in file path: {file_path}")

        return id_str

    def get_machine_id_list(self, machine_type: str) -> list[str]:
        if machine_type == "ToyCar":
            machine_id_list = [
                "id_01",
                "id_02",
                "id_03",
                "id_04",
                "id_05",
                "id_06",
                "id_07",
            ]
        elif machine_type == "ToyConveyor":
            machine_id_list = ["id_01", "id_02", "id_03", "id_04", "id_05", "id_06"]
        else:
            machine_id_list = [
                "id_00",
                "id_01",
                "id_02",
                "id_03",
                "id_04",
                "id_05",
                "id_06",
            ]

        return machine_id_list

    def get_machine_type_str(self, file_path: str) -> str:
        for machine in self.machine_list:
            if machine in file_path:
                return machine
        raise ValueError(f"No machine type found in file path: {file_path}")

    def get_machine_type_label(self, file_path: str) -> int:
        machine_type = file_path.split("/")[-4]
        machine_type_list = ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]
        index = machine_type_list.index(machine_type)
        return index

    def get_machine_id_label_str(self, machine_type: str, machine_id: str) -> str:
        return f"{machine_type}_{machine_id}"

    def get_train_test_label(self, file_path: str) -> int:
        if "train" in file_path:
            return 0
        elif "test" in file_path:
            return 1
        else:
            raise ValueError(
                f"Could not determine train_test label from file path: {file_path}"
            )

    def get_dev_eval_label(self, file_path: str) -> int:
        machine_type = file_path.split("/")[-4]

        if "train" in file_path:
            if machine_type in ["ToyCar"]:
                eval_id_list = ["id_05", "id_06", "id_07"]
                if any(id_str in file_path for id_str in eval_id_list):
                    return 1
                else:
                    return 0

            elif machine_type in ["fan", "pump", "slider", "valve"]:
                eval_id_list = ["id_01", "id_03", "id_05"]
                if any(id_str in file_path for id_str in eval_id_list):
                    return 1
                else:
                    return 0

            else:
                eval_id_list = ["id_04", "id_05", "id_06"]
                if any(id_str in file_path for id_str in eval_id_list):
                    return 1
                else:
                    return 0
        else:
            if "dev_data" in file_path:
                return 0
            elif "eval_data" in file_path:
                return 1
            else:
                raise ValueError(
                    f"Could not determine dev_eval label from file path: {file_path}"
                )

    def get_machine_id_label_dict(self, machine_list: list[str]) -> dict[int, str]:
        machine_id_label_dict = {}
        global_idx = 0
        for machine_type in machine_list:
            machine_id_list = self.get_machine_id_list(machine_type)
            for id_str in machine_id_list:
                key = self.get_machine_id_label_str(machine_type, id_str)
                machine_id_label_dict[global_idx] = key
                global_idx += 1
        return machine_id_label_dict

    def get_machine_type_label_dict(self, machine_list: list[str]) -> dict[int, str]:
        machine_label_dict = {}
        for idx, machine_type in enumerate(machine_list):
            machine_label_dict[idx] = machine_type
        return machine_label_dict

    def get_normal_anomaly_label_dict(self) -> dict[int, str]:
        normal_anomaly_dict = {0: "normal", 1: "anomaly"}
        return normal_anomaly_dict

    def get_train_test_label_dict(self) -> dict[int, str]:
        train_test_dict = {0: "Train", 1: "Test"}
        return train_test_dict

    def get_label_dict(self) -> dict[str, dict[int, str]]:
        label_dict = {
            "machine_id_label": self.machine_id_label_dict,
            "machine_type_label": self.machine_type_label_dict,
            "anomaly_label": self.normal_anomaly_label_dict,
            "train_test": self.get_train_test_label_dict(),
            "dev_eval_label": self.dev_eval_label_dict,
        }
        return label_dict
