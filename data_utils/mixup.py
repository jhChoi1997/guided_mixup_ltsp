import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def audio_mel_mixup(
    wav: torch.Tensor,
    mel: torch.Tensor,
    label: torch.Tensor,
    alpha: float,
    add_orig: bool = False,
):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = wav.size()[0]
    index = torch.randperm(batch_size)

    mixed_wav = lam * wav + (1 - lam) * wav[index, :]
    mixed_mel = lam * mel + (1 - lam) * mel[index, :]
    y_a, y_b = label, label[index]

    if add_orig:
        return (
            torch.cat([mixed_wav, wav]),
            torch.cat([mixed_mel, mel]),
            torch.cat([y_a, y_a]),
            torch.cat([y_b, y_a]),
            lam,
        )
    else:
        return mixed_wav, mixed_mel, y_a, y_b, lam
