import torch
from tqdm.notebook import tqdm


def init_experiment_metrics(include_val_metrics=True):
    metrics = {
        "train": {
            "loss": [],
            "acc": [],
            "batch_accs": [],
        }
    }
    if include_val_metrics:
        metrics["val"] = {
            "loss": [],
            "acc": [],
            "batch_accs": [],
        }
    return metrics


def init_train_metrics():
    metrics = {
        "acc": 0.0,
        "loss": 0.0,
        "total_preds": 0,
        "correct_preds": 0,
        "batch_accs": [],
    }
    return metrics


def init_test_metrics():
    metrics = {
        "acc": 0.0,
        "loss": 0.0,
        "labels": [],
        "predictions": [],
    }
    return metrics


def update_train_metrics(labels, predictions, metrics):
    correct_predictions = (predictions == labels).sum().item()
    metrics["total_preds"] += labels.size(0)
    metrics["correct_preds"] += correct_predictions

    if "batch_accs" in metrics:
        batch_accuracy = (correct_predictions * 100) / labels.size(0)
        metrics["batch_accs"].append(batch_accuracy)
    return metrics


def update_test_metrics(labels, predictions, metrics):
    metrics["predictions"].append(predictions.detach().cpu().numpy())
    metrics["labels"].append(labels.detach().cpu().numpy())
    return metrics


def update_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    loss: float,
    metrics: dict,
    phase="train",
):
    """
    Updates metrics
    """
    metrics["loss"] += loss.item()
    predictions = torch.argmax(outputs, dim=1)

    if phase == "train":
        metrics = update_train_metrics(labels, predictions, metrics)
    else:
        metrics = update_test_metrics(labels, predictions, metrics)
    return metrics


def aggregate_metrics(metrics, phase_metrics, phase="train"):
    """
    Aggregates metrics across epochs for training and validation.

    Args:
        metrics (dict): Dictionary to store aggregated metrics.
        phase_metrics (dict): Metrics from the current phase (train/val).
        phase (str): Phase name ("train" or "val").
    """
    metrics[phase]["loss"].append(phase_metrics["loss"])
    metrics[phase]["acc"].append(phase_metrics["acc"])

    if "batch_accs" in phase_metrics:
        metrics[phase]["batch_accs"] += phase_metrics["batch_accs"]
    return metrics


def log_metrics(epoch, num_epochs, train_metrics, val_metrics=None):
    """
    Logs the metrics for the current epoch.
    """
    tqdm.write(
        f"[Epoch {epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_metrics['loss'][epoch]:.4f}, Train Acc: {train_metrics['acc'][epoch]:.2f}% | "
        f"Val Loss: {val_metrics['loss'][epoch]:.4f}, Val Acc: {val_metrics['acc'][epoch]:.2f}%"
    )


def get_classification_scores(metrics, scores: dict):
    if "scores" not in metrics:
        metrics["scores"] = {}

    for key, value in scores.items():
        metrics["scores"][key] = value(metrics["labels"], metrics["predictions"])
    return metrics