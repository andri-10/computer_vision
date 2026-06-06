"""Shared training callbacks."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def standard_callbacks(model_path: str | Path, patience: int = 5):
    """Return checkpoint, early stopping, and LR scheduling callbacks."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(2, patience // 2),
            min_lr=1e-6,
        ),
    ]
