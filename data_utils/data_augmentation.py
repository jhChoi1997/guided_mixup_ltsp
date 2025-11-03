from typing import Any

from data_utils.mixup import audio_mel_mixup, mixup_data


def select_function(name: str) -> Any:
    """Selects the augmentation function based on the provided dictionary."""
    if name == "mixup":
        return mixup_data
    elif name == "audio_mel_mixup":
        return audio_mel_mixup
    else:
        raise ValueError(f"Invalid data augmentation name: {name}")


def batch_data_augmentation(
    aug_dict: dict, batch: dict[str, Any], is_test: bool = False
) -> dict[str, Any]:
    if not aug_dict:
        return batch

    if is_test:
        input_keys = aug_dict["input_keys"]
        input_keys_for_test = aug_dict.get("input_keys_for_test", input_keys)
        output_keys = aug_dict["output_keys"]

        for key, value in zip(output_keys, input_keys_for_test):
            if value in batch:
                batch[key] = batch[value]
            else:
                raise KeyError(
                    f"Key '{value}' not found in batch for test augmentation."
                )

        return batch

    aug_fn = select_function(aug_dict["name"])

    input_keys = aug_dict["input_keys"]
    output_keys = aug_dict["output_keys"]
    args = aug_dict.get("args", {})

    inputs = [batch[key].cuda() for key in input_keys]
    outputs = aug_fn(*inputs, **args)

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for key, value in zip(output_keys, outputs):
        batch[key] = value

    return batch
