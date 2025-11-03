import torch
import torch.nn as nn
import torch.nn.functional as F

from util.register import register_node


@register_node("mixup")
class Mixup(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        return y_a * lam + y_b * (1 - lam)


@register_node("unsqueeze")
class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


@register_node("squeeze")
class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)


@register_node("concatenate")
class Concatenate(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return torch.cat(x, dim=self.dim)


@register_node("cosine_linear")
class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_norm = F.normalize(self.weight, dim=1, eps=1e-8)
        x_norm = F.normalize(x, dim=1, eps=1e-8)

        cosine_similarity = torch.matmul(x_norm, weight_norm.T)

        return cosine_similarity


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


@register_node("multiply")
class Multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


@register_node("sum")
class Sum(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim)


@register_node("subtract")
class Subtract(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y


@register_node("mean")
class Mean(nn.Module):
    def __init__(self, dim: int = 1, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


@register_node("predicted_class")
class PredictedClass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=1)


@register_node("one_minus_input")
class OneMinusInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - x


@register_node("target_indices")
class TargetIndices(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        view_shape = [1] * x.dim()
        view_shape[0] = -1
        index = label.view(view_shape)

        expand_shape = list(x.shape)
        expand_shape[self.dim] = 1
        index = index.expand(expand_shape)

        gathered = torch.gather(x, self.dim, index)

        return gathered.squeeze(self.dim)


@register_node("normalize")
class Normalize(nn.Module):
    def __init__(self, p=2, dim=1, eps=1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


@register_node("wavenet_mse_loss")
class WaveNetMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        receptive_field = target.size(-1) - x.size(-1)
        return F.mse_loss(x, target[..., receptive_field:])


@register_node("wavenet_mse")
class WaveNetMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        receptive_field = target.size(-1) - x.size(-1)
        mse = F.mse_loss(x, target[..., receptive_field:], reduction="none")

        return torch.mean(mse, dim=(1, 2))
