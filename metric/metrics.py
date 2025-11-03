import numpy as np
import sklearn.metrics as metrics
import sklearn.mixture as mixture
import torch
from sklearn import neighbors

from util.register import register_metric


from typing import Literal


def get_knn_scores(
    embeddings: torch.Tensor,
    train_test: torch.Tensor,
    n_neighbors: int = 1,
    return_train: bool = False,
    metric: str = "cosine",
    remove_self: bool = True,
):
    train_indices = torch.where(train_test == 0)[0]
    test_indices = torch.where(train_test == 1)[0]

    knn = neighbors.NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    knn.fit(embeddings[train_indices])  # type: ignore
    score, _ = knn.kneighbors(embeddings)  # type: ignore

    if return_train:
        output = score[:, n_neighbors - 1]  # type: ignore
        if remove_self:
            output[train_indices] = score[train_indices, n_neighbors]
        return output  # type: ignore
    else:
        return score[:, n_neighbors - 1][test_indices]


def get_ensemble_scores(
    score_list: list[torch.Tensor],
    train_test: torch.Tensor,
) -> torch.Tensor:
    assert len(score_list) > 0, "Score list must not be empty"

    shapes = [score.shape for score in score_list]
    assert all(
        shape == shapes[0] for shape in shapes
    ), "All scores must have the same shape"

    normalized_scores = []
    for score in score_list:
        train_score_min = torch.min(score[train_test == 0])
        train_score_max = torch.max(score[train_test == 0])

        normalized_score = (score - train_score_min) / (
            train_score_max - train_score_min + 1e-8
        )
        normalized_scores.append(normalized_score)

    return torch.mean(torch.stack(normalized_scores, dim=0), dim=0)


def get_average_freq_diff(
    target_latent: torch.Tensor, pred_latent: torch.Tensor
) -> torch.Tensor:
    receptive_field = target_latent.shape[-1] - pred_latent.shape[-1]
    freq_diff = target_latent[..., receptive_field:] - pred_latent
    average_freq_diff = torch.mean(freq_diff, dim=2)
    return average_freq_diff


def get_mahalanobis_distance(
    x: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor, squared: bool = False
) -> torch.Tensor:
    diff = x - mean
    dist_sq = torch.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    return dist_sq if squared else torch.sqrt(dist_sq)


def get_ltsp_error_scores(
    gt_latent: torch.Tensor,
    pred_latent: torch.Tensor,
    train_test: torch.Tensor,
    return_train: bool = True,
) -> torch.Tensor:
    train_mask = train_test == 0
    test_mask = train_test == 1

    train_freq_diff = get_average_freq_diff(
        gt_latent[train_mask], pred_latent[train_mask]
    )
    freq_diff_mean = torch.mean(train_freq_diff, dim=0)
    freq_diff_cov = torch.cov(train_freq_diff.T)
    freq_diff_inv_cov = torch.linalg.pinv(freq_diff_cov)

    test_freq_diff = get_average_freq_diff(gt_latent, pred_latent)
    scores = get_mahalanobis_distance(test_freq_diff, freq_diff_mean, freq_diff_inv_cov)

    if return_train:
        return scores
    else:
        return scores[test_mask]


def get_subcluster_gmm_scores(
    embeddings: torch.Tensor,
    train_test: torch.Tensor,
    n_subclusters: int,
    return_train: bool = True,
) -> torch.Tensor:
    train_mask = train_test == 0

    clf1 = mixture.GaussianMixture(
        n_components=n_subclusters,
        covariance_type="full",
        reg_covar=1e-3,
    ).fit(embeddings[train_mask].numpy())

    y_pred = -np.max(clf1._estimate_log_prob(embeddings.numpy()), axis=-1)  # type: ignore

    if return_train:
        return torch.Tensor(y_pred)
    else:
        test_mask = train_test == 1
        return torch.Tensor(y_pred[test_mask])


@register_metric("new_average_performance")
def new_average_performance(
    anomaly_scores: torch.Tensor,
    embeddings: torch.Tensor,
    train_test: torch.Tensor,
    anomaly_label: torch.Tensor,
    machine_id_label: torch.Tensor,
    scores: list[str] = ["prob", "knn", "gmm"],
    output_metric: Literal["auc", "pauc", "mean"] = "auc",
    max_fpr: float = 0.1,
    n_neighbors: int = 1,
    n_subclusters: int = 1,
    **kwargs,
):
    test_mask = train_test == 1
    performance = []

    unique_machine_ids = torch.unique(machine_id_label)
    for machine_id in unique_machine_ids:
        score_list = []

        machine_mask = machine_id_label == machine_id

        y_true = anomaly_label[machine_mask & test_mask]

        if "prob" in scores:
            score = anomaly_scores[machine_mask]
            score_list.append(score)
        if "knn" in scores:
            score = torch.Tensor(
                get_knn_scores(
                    embeddings[machine_mask],
                    train_test[machine_mask],
                    n_neighbors=n_neighbors,
                    return_train=True,
                )
            )
            score_list.append(score)
        if "gmm" in scores:
            score = get_subcluster_gmm_scores(
                embeddings[machine_mask],
                train_test[machine_mask],
                n_subclusters=n_subclusters,
                return_train=True,
            )
            score_list.append(score)

        machine_train_test_label = train_test[machine_mask]
        machine_test_mask = machine_train_test_label == 1
        ensemble_score = get_ensemble_scores(score_list, train_test[machine_mask])
        y_pred = ensemble_score[machine_test_mask]

        auc = metrics.roc_auc_score(y_true, y_pred)
        pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
        avg_auc_pauc = (auc + pauc) / 2

        if output_metric == "auc":
            performance.append(auc)
        elif output_metric == "pauc":
            performance.append(pauc)
        elif output_metric == "mean":
            performance.append(avg_auc_pauc)
        else:
            raise ValueError(f"Unknown metric: {output_metric}")

    return np.mean(performance) if performance else 0.0


@register_metric("new_average_performance_ltsp")
def new_average_performance_ltsp(
    anomaly_scores: torch.Tensor,
    embeddings: torch.Tensor,
    gt_latent: torch.Tensor,
    pred_latent: torch.Tensor,
    train_test: torch.Tensor,
    anomaly_label: torch.Tensor,
    machine_id_label: torch.Tensor,
    scores: list[str] = ["prob", "knn", "gmm", "ltsp_error"],
    output_metric: Literal["auc", "pauc", "mean"] = "auc",
    max_fpr: float = 0.1,
    n_neighbors: int = 1,
    n_subclusters: int = 1,
    **kwargs,
):
    test_mask = train_test == 1
    performance = []

    unique_machine_ids = torch.unique(machine_id_label)
    for machine_id in unique_machine_ids:
        score_list = []

        machine_mask = machine_id_label == machine_id

        y_true = anomaly_label[machine_mask & test_mask]

        if "prob" in scores:
            score = anomaly_scores[machine_mask]
            score_list.append(score)
        if "knn" in scores:
            score = torch.Tensor(
                get_knn_scores(
                    embeddings[machine_mask],
                    train_test[machine_mask],
                    n_neighbors=n_neighbors,
                    return_train=True,
                )
            )
            score_list.append(score)
        if "gmm" in scores:
            score = get_subcluster_gmm_scores(
                embeddings[machine_mask],
                train_test[machine_mask],
                n_subclusters=n_subclusters,
                return_train=True,
            )
            score_list.append(score)
        if "ltsp_error" in scores:
            score = get_ltsp_error_scores(
                gt_latent[machine_mask],
                pred_latent[machine_mask],
                train_test[machine_mask],
                return_train=True,
            )
            score_list.append(score)

        machine_train_test_label = train_test[machine_mask]
        machine_test_mask = machine_train_test_label == 1
        ensemble_score = get_ensemble_scores(score_list, train_test[machine_mask])
        y_pred = ensemble_score[machine_test_mask]

        auc = metrics.roc_auc_score(y_true, y_pred)
        pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
        avg_auc_pauc = (auc + pauc) / 2

        if output_metric == "auc":
            performance.append(auc)
        elif output_metric == "pauc":
            performance.append(pauc)
        elif output_metric == "mean":
            performance.append(avg_auc_pauc)
        else:
            raise ValueError(f"Unknown metric: {output_metric}")

    return np.mean(performance) if performance else 0.0
