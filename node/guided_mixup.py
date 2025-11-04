import torch
import torch.nn as nn

from util.register import register_node


@register_node("guided_mixup")
class GuidedMixup(nn.Module):
    def __init__(self, s=30, margin=0.7):
        super().__init__()
        self.s = s
        self.margin = margin

    def forward(self, cos_theta: torch.Tensor, mixed_one_hot: torch.Tensor):
        phi = (
            mixed_one_hot * torch.cos(torch.arccos(cos_theta) + self.margin)
            + (1 - mixed_one_hot) * cos_theta
        )
        logits = phi * self.s
        return logits
