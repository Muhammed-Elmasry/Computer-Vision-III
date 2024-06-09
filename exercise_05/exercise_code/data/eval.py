from exercise_code.data.utils import PCA
from exercise_code.data.seg_datasets.davis_obj_seg import (
    DavisDataset,
)
from exercise_code.data.unsupervised_segmentation.gaussian_mixture_models import (
    GaussianMixtureModels,
)
from exercise_code.data.unsupervised_segmentation.k_means import (
    KMeans,
)
from exercise_code.data.unsupervised_segmentation.spectral_clustering import (
    SpectralClustering,
    downscale,
)


def evaluate_model(
    data: DavisDataset, model: KMeans | GaussianMixtureModels, pca: PCA
) -> None:
    if isinstance(model, KMeans):
        num_clusters = model.num_clusters
    elif isinstance(model, GaussianMixtureModels):
        num_clusters = model.num_classes
    else:
        raise NotImplementedError

    metrics = [
        {
            "iou": 0,
            "precision": 0,
            "recall": 0,
            "accuracy": 0,
        }
        for _ in range(num_clusters)
    ]

    for sample in data:
        img = sample["image"]
        annotations = sample["annotations_coarse"]
        feature_map = pca.do_pca(sample["feature_map"])

        if isinstance(model, KMeans):
            assignments = model.inference(feature_map.flatten(0, 2)).reshape(
                feature_map.shape[:3]
            )
        elif isinstance(model, GaussianMixtureModels):
            assignments = (
                model.inference(feature_map.flatten(0, 2))
                .reshape((-1,) + feature_map.shape[:3])
                .max(0)
                .indices
            )
        else:
            raise NotImplementedError

        for i in range(assignments.max() + 1):
            mask = assignments == i
            if mask.sum() == 0:
                continue
            intersection = annotations.bool() & mask
            union = annotations.bool() | mask
            iou = intersection.sum() / union.sum()
            precision = intersection.sum() / mask.sum()
            recall = intersection.sum() / annotations.sum()
            accuracy = (annotations == mask).sum() / annotations.numel()

            metrics[i]["iou"] += iou.item()
            metrics[i]["precision"] += precision.item()
            metrics[i]["recall"] += recall.item()
            metrics[i]["accuracy"] += accuracy.item()

    best_iou = 0
    best_cluster = -1
    for i, metric in enumerate(metrics):
        if metric["iou"] > best_iou:
            best_iou = metric["iou"]
            best_cluster = i
    best_metrics = {
        "accuracy": metrics[best_cluster]["accuracy"] / len(data),
        "precision": metrics[best_cluster]["precision"] / len(data),
        "recall": metrics[best_cluster]["recall"] / len(data),
        "iou": metrics[best_cluster]["iou"] / len(data),
    }

    metrics_header()
    print_metrics(best_metrics, "test")


def evaluate_spectral_clustering(data: DavisDataset, model: SpectralClustering) -> None:
    metrics = {
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "accuracy": 0,
    }

    counter = 0
    for batch in data:
        for idx in range(batch["feature_map"].shape[0]):
            img = batch["image"][idx]
            annotations = batch["annotations_coarse"][idx]
            feature_map = batch["feature_map"][idx]

            assignments = downscale(
                model.original_cluster(img, feature_map, image_color_lambda=1.0)
                .reshape(img.shape[-2:])
                .float()[None],
                model.scaling,
            )[0].int()

            best_iou = 0
            best_cluster = -1
            for i in range(assignments.max() + 1):
                mask = assignments == i
                if mask.sum() == 0:
                    continue
                intersection = annotations.bool() & mask
                union = annotations.bool() | mask
                iou = intersection.sum() / union.sum()
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = i
            mask = assignments == best_cluster
            intersection = annotations.bool() & mask
            union = annotations.bool() | mask
            iou = intersection.sum() / union.sum()
            precision = intersection.sum() / mask.sum()
            recall = intersection.sum() / annotations.sum()
            accuracy = (annotations == mask).sum() / annotations.numel()

            metrics["accuracy"] += accuracy.item()
            metrics["precision"] += precision.item()
            metrics["recall"] += recall.item()
            metrics["iou"] += iou.item()
            counter += 1

    metrics = {
        "accuracy": metrics["accuracy"] / counter,
        "precision": metrics["precision"] / counter,
        "recall": metrics["recall"] / counter,
        "iou": metrics["iou"] / counter,
    }

    metrics_header()
    print_metrics(metrics, "test")


def metrics_header():
    print(f"{f'Split' : >5} {f'Acc' : >5} {f'Prcn' : >5} {f'Rcll' : >5} {f'IOU' : >5}")


def print_metrics(metrics: dict[str, float], split_name: str):
    metric_names = ["accuracy", "precision", "recall", "iou"]
    str_metrics: dict[str, str] = {}
    for metric in metric_names:
        value = metrics.get(metric, None)
        if value:
            str_metrics[metric] = f"{value :.2f}"
        else:
            str_metrics[metric] = "-"

    acc = str_metrics["accuracy"]
    m_prcn = str_metrics["precision"]
    m_rcll = str_metrics["recall"]
    m_iou = str_metrics["iou"]

    print(
        f"{f'{split_name}' : >5} {f'{acc}' : >5} {f'{m_prcn}' : >5} {f'{m_rcll}' : >5} {f'{m_iou}' : >5}"
    )
