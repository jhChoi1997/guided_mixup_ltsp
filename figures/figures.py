import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

import metric.metrics as my_metrics

from metric.metrics import (
    get_ensemble_scores,
    get_knn_scores,
    get_subcluster_gmm_scores,
    get_ltsp_error_scores,
)
from util.register import register_figure


@register_figure("new_two_score_scatter_plot")
def new_two_score_scatter_plot(
    anomaly_scores: torch.Tensor,
    embeddings: torch.Tensor,
    gt_latent: torch.Tensor,
    pred_latent: torch.Tensor,
    train_test: torch.Tensor,
    anomaly_labels: torch.Tensor,
    class_scores: list[str] = ["prob", "knn", "gmm"],
    pred_scores: list[str] = ["ltsp_error"],
    n_neighbors: int = 1,
    n_subclusters: int = 1,
    plot_train_data: bool = False,
    font_scale: float = 2.0,
    **kwargs,
) -> Figure:
    test_mask = train_test == 1
    train_mask = train_test == 0
    normal_mask = anomaly_labels == 0
    anomaly_mask = anomaly_labels == 1

    classification_scores = []
    prediction_scores = []

    if "prob" in class_scores:
        score = anomaly_scores
        classification_scores.append(score)
    if "knn" in class_scores:
        score = torch.Tensor(
            get_knn_scores(
                embeddings,
                train_test,
                n_neighbors=n_neighbors,
                return_train=True,
            )
        )
        classification_scores.append(score)
    if "gmm" in class_scores:
        score = get_subcluster_gmm_scores(
            embeddings,
            train_test,
            n_subclusters=n_subclusters,
            return_train=True,
        )
        classification_scores.append(score)
    if "ltsp_error" in pred_scores:
        gt_score = get_ltsp_error_scores(
            gt_latent, pred_latent, train_test, return_train=True
        )
        prediction_scores.append(gt_score)

    classification_score = get_ensemble_scores(classification_scores, train_test)
    prediction_score = get_ensemble_scores(prediction_scores, train_test)

    base = plt.rcParams.get("font.size", 10.0)
    fs = base * font_scale
    with plt.rc_context(
        {
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
        }
    ):

        figure = plt.figure(figsize=(8, 6))

        if plot_train_data:
            plt.scatter(
                classification_score[train_mask],
                prediction_score[train_mask],
                color="gray",
                label="Train Data",
                alpha=0.3,
                s=20,
            )

        plt.scatter(
            classification_score[test_mask & normal_mask],
            prediction_score[test_mask & normal_mask],
            color=plt.get_cmap("tab10")(0),
            label="Test Normal",
            alpha=0.5,
        )

        plt.scatter(
            classification_score[test_mask & anomaly_mask],
            prediction_score[test_mask & anomaly_mask],
            # color="red",
            color=plt.get_cmap("tab10")(1),
            label="Test Anomaly",
            alpha=0.5,
        )

        if plot_train_data:
            train_classification_mean = torch.mean(
                classification_score[train_mask]
            ).item()
            train_prediction_mean = torch.mean(prediction_score[train_mask]).item()
            plt.scatter(
                train_classification_mean,
                train_prediction_mean,
                color="green",
                marker="x",
                s=200,
                linewidth=3,
                label="Train Mean",
            )

        plt.xlim(kwargs.get("xlim", None))
        plt.ylim(kwargs.get("ylim", None))

        plt.legend()
        plt.xlabel(kwargs.get("xlabel", "Classification Scores"))
        plt.ylabel(kwargs.get("ylabel", "Prediction Scores"))
        plt.title(
            kwargs.get(
                "title", "Classification Scores vs Prediction Scores Scatter Plot"
            )
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return figure


@register_figure("new_dcase2020_all_score_histogram")
def new_dcase2020_all_score_histogram(
    anomaly_scores: torch.Tensor,
    embeddings: torch.Tensor,
    gt_latent: torch.Tensor,
    pred_latent: torch.Tensor,
    anomaly_label: torch.Tensor,
    train_test: torch.Tensor,
    scores: list[str] = ["prob", "knn", "gmm", "ltsp_error"],
    bins: int = 50,
    n_neighbors: int = 1,
    n_subcenters: int = 1,
    xlabel: str = "Ensemble Score",
    ylabel: str = "Frequency",
    font_scale: float = 1.5,
    **kwargs,
) -> Figure:
    score_list = []

    test_mask = train_test == 1
    test_labels = anomaly_label[test_mask]

    if "prob" in scores:
        prob_scores = anomaly_scores
        score_list.append(prob_scores)
    if "knn" in scores:
        knn_scores = torch.Tensor(
            my_metrics.get_knn_scores(
                embeddings,
                train_test,
                n_neighbors=n_neighbors,
                return_train=True,
            )
        )
        score_list.append(knn_scores)
    if "gmm" in scores:
        gmm_scores = get_subcluster_gmm_scores(
            embeddings,
            train_test,
            n_subclusters=n_subcenters,
            return_train=True,
        )
        score_list.append(gmm_scores)
    if "ltsp_error" in scores:
        ltsp_error_scores = get_ltsp_error_scores(
            gt_latent, pred_latent, train_test, return_train=True
        )
        score_list.append(ltsp_error_scores)

    ensemble_scores = get_ensemble_scores(score_list, train_test)
    test_ensemble_scores = ensemble_scores[test_mask]

    label_dict_list = kwargs["label_dict_list"]
    label_dict = label_dict_list[0]
    unique_labels = np.array(list(label_dict.keys()))

    cmap = kwargs.get("cmap", "tab10")
    cmap_sampling = kwargs.get("cmap_sampling", False)
    if cmap_sampling:
        colormap = plt.cm.get_cmap(cmap, len(unique_labels))
    else:
        colormap = plt.cm.get_cmap(cmap)

    xlim = kwargs.get("xlim", None)
    if xlim is not None:
        bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    else:
        bin_edges = bins

    base = plt.rcParams.get("font.size", 10.0)
    fs = base * font_scale
    with plt.rc_context(
        {
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
        }
    ):
        figure = plt.figure(figsize=(6, 3))
        for idx, label_str in label_dict.items():
            plt.hist(
                test_ensemble_scores[test_labels == idx],
                bins=bin_edges,
                alpha=0.5,
                label=label_str,
                edgecolor="black",
                color=colormap(idx),
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(axis="y", alpha=0.5)
        plt.title(kwargs.get("title", "Ensemble Score Histogram"))

        if xlim is not None:
            plt.xlim(xlim)

        ylim = kwargs.get("ylim", None)
        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout()
        return figure
