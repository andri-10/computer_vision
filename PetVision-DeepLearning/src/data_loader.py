"""Dataset loading and split helpers for Oxford-IIIT Pet."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = "oxford_iiit_pet:4.*.*"
SEED = 42


def load_oxford_pet(with_info: bool = True):
    """Load Oxford-IIIT Pet from TensorFlow Datasets."""
    return tfds.load(DATASET_NAME, with_info=with_info, as_supervised=False)


def get_label_names(info) -> list[str]:
    """Return the 37 breed names from TFDS metadata."""
    return list(info.features["label"].names)


def save_label_mapping(label_names: list[str], path: str | Path = "results/figures/label_mapping.json") -> Dict[int, str]:
    """Save a reusable integer-to-class-name mapping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {idx: name for idx, name in enumerate(label_names)}
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def load_label_mapping(path: str | Path = "results/figures/label_mapping.json") -> Dict[int, str]:
    """Load a saved integer-to-class-name mapping."""
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): value for key, value in mapping.items()}


def train_validation_split(
    train_ds: tf.data.Dataset,
    train_size: int,
    validation_fraction: float = 0.2,
    seed: int = SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create deterministic train/validation datasets from the official train split."""
    validation_size = int(train_size * validation_fraction)
    shuffled = train_ds.shuffle(train_size, seed=seed, reshuffle_each_iteration=False)
    validation_ds = shuffled.take(validation_size)
    training_ds = shuffled.skip(validation_size)
    return training_ds, validation_ds


def get_splits(validation_fraction: float = 0.2, seed: int = SEED):
    """Load dataset and return train, validation, test, metadata, and label names."""
    datasets, info = load_oxford_pet(with_info=True)
    train_size = int(info.splits["train"].num_examples)
    train_ds, val_ds = train_validation_split(datasets["train"], train_size, validation_fraction, seed)
    test_ds = datasets["test"]
    label_names = get_label_names(info)
    return train_ds, val_ds, test_ds, info, label_names


def configure_for_performance(
    dataset: tf.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 1024,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Batch and prefetch a dataset, with optional deterministic shuffling."""
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
