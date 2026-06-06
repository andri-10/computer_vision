"""Visualization helpers for PetVision notebooks and demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


MASK_COLORS = np.array(
    [
        [255, 120, 90],
        [40, 40, 40],
        [90, 180, 255],
    ],
    dtype=np.uint8,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a 2D/3D integer mask to an RGB mask."""
    mask = np.squeeze(mask).astype(np.int32)
    mask = np.clip(mask, 0, len(MASK_COLORS) - 1)
    return MASK_COLORS[mask]


def plot_training_history(history, output_path: str | Path, title: str):
    """Save training and validation curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics.get("accuracy", []), label="train")
    axes[0].plot(metrics.get("val_accuracy", []), label="validation")
    axes[0].set_title(f"{title} Accuracy")
    axes[0].legend()
    axes[1].plot(metrics.get("loss", []), label="train")
    axes[1].plot(metrics.get("val_loss", []), label="validation")
    axes[1].set_title(f"{title} Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, label_names: list[str], output_path: str | Path, max_labels: int = 37):
    """Save a confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig_size = (14, 12) if len(label_names) > 20 else (10, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=label_names[:max_labels],
        yticklabels=label_names[:max_labels],
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def show_segmentation_triplet(image, true_mask, pred_mask, output_path: str | Path | None = None):
    """Show image, ground truth mask, and predicted mask."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Input")
    axes[1].imshow(colorize_mask(true_mask))
    axes[1].set_title("True mask")
    axes[2].imshow(colorize_mask(pred_mask))
    axes[2].set_title("Predicted mask")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    return fig


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """Overlay a normalized heatmap on an RGB image."""
    image = np.asarray(image)
    if image.max() > 1.0:
        image = image / 255.0
    heatmap = np.clip(heatmap, 0, 1)
    color_map = plt.get_cmap(cmap)
    colored = color_map(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colored, 0, 1)
