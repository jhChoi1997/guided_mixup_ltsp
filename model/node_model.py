from typing import Any

import torch
import torch.nn as nn

from util.register import get_node


class NodeModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.node_dict: nn.ModuleDict = nn.ModuleDict()
        self.node_execution_list: list[dict] = config["node_execution_list"]

        for node in config["node_list"]:
            node_type = node["node_type"]
            node_name = node["node_name"]
            args = node.get("args", {})

            requires_grad = args.pop("requires_grad", True)

            self.node_dict[node_name] = get_node(node_type, **args)

            if not requires_grad:
                # Freeze the parameters of the node if requires_grad is False
                for param in self.node_dict[node_name].parameters():
                    param.requires_grad = False

        self.is_train: bool = True

    def forward(self, batch: dict) -> dict:
        device = next(self.node_dict.parameters()).device
        batch = {
            key: (
                value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

        for execution in self.node_execution_list:
            node_name: str = execution["node_name"]
            input_keys: str | list[str] = execution["input_keys"]
            output_keys: str | list[str] = execution["output_keys"]
            args: dict[str, Any] = execution.get("args", {})

            node = self.node_dict[node_name]
            node.is_train = self.is_train  # type: ignore

            if isinstance(input_keys, str):
                outputs = node(batch[input_keys], **args)
            else:
                outputs = node(*(batch.get(key, None) for key in input_keys), **args)

            if isinstance(output_keys, str):
                batch[output_keys] = outputs
            else:
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                for key, value in zip(output_keys, outputs):
                    batch[key] = value

        return batch
