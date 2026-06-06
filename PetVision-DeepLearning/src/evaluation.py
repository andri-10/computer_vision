"""Evaluation helpers for classification and segmentation."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Collect labels, predicted labels, and probability vectors."""
    y_true, y_pred, y_prob = [], [], []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
        y_prob.extend(probabilities.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def classification_metrics_table(y_true, y_pred, label_names: list[str]) -> pd.DataFrame:
    """Return precision/recall/F1 report as a DataFrame."""
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def save_classification_outputs(y_true, y_pred, label_names: list[str], output_dir: str | Path):
    """Save confusion matrix and classification report CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion_matrix(y_true, y_pred), index=label_names, columns=label_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    classification_metrics_table(y_true, y_pred, label_names).to_csv(output_dir / "classification_report.csv")


def top_k_accuracy(y_true, y_prob, k: int = 3) -> float:
    """Compute sparse Top-K accuracy from probability vectors."""
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return float(np.mean([truth in preds for truth, preds in zip(y_true, top_k)]))


def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute pixel accuracy for segmentation masks."""
    return float(np.mean(y_true == y_pred))


def mean_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    """Compute mean intersection-over-union over mask classes."""
    scores = []
    for cls in range(num_classes):
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union > 0:
            scores.append(intersection / union)
    return float(np.mean(scores)) if scores else 0.0


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3, smooth: float = 1e-6) -> float:
    """Compute macro Dice coefficient over mask classes."""
    scores = []
    for cls in range(num_classes):
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        intersection = np.logical_and(true_cls, pred_cls).sum()
        denominator = true_cls.sum() + pred_cls.sum()
        scores.append((2.0 * intersection + smooth) / (denominator + smooth))
    return float(np.mean(scores))


def evaluate_segmentation_model(model: tf.keras.Model, dataset: tf.data.Dataset, num_batches: int | None = None):
    """Evaluate segmentation with pixel accuracy, mean IoU, and Dice."""
    y_true_all, y_pred_all = [], []
    for batch_idx, (images, masks) in enumerate(dataset):
        logits = model.predict(images, verbose=0)
        preds = np.argmax(logits, axis=-1)
        y_true_all.append(np.squeeze(masks.numpy(), axis=-1))
        y_pred_all.append(preds)
        if num_batches is not None and batch_idx + 1 >= num_batches:
            break
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return {
        "pixel_accuracy": pixel_accuracy(y_true, y_pred),
        "mean_iou": mean_iou(y_true, y_pred),
        "dice_coefficient": dice_coefficient(y_true, y_pred),
    }


def measure_inference_time(model: tf.keras.Model, sample_batch: tf.Tensor, runs: int = 30) -> float:
    """Return average inference time per batch in milliseconds."""
    model.predict(sample_batch, verbose=0)
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(sample_batch, verbose=0)
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0
