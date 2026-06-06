"""Preprocessing pipelines for classification and segmentation."""

from __future__ import annotations

import tensorflow as tf


CLASSIFICATION_BASELINE_SIZE = (128, 128)
TRANSFER_SIZE = (224, 224)
SEGMENTATION_SIZE = (160, 160)


def normalize_image_01(image: tf.Tensor) -> tf.Tensor:
    """Convert image to float32 in [0, 1]."""
    image = tf.cast(image, tf.float32)
    return image / 255.0


def preprocess_baseline_classification(example, image_size=CLASSIFICATION_BASELINE_SIZE):
    """Prepare one TFDS example for the baseline CNN."""
    image = tf.image.resize(example["image"], image_size)
    image = normalize_image_01(image)
    label = tf.cast(example["label"], tf.int32)
    return image, label


def preprocess_transfer_classification(example, image_size=TRANSFER_SIZE, backbone: str = "mobilenet_v2"):
    """Prepare one TFDS example for a Keras Applications classifier."""
    image = tf.image.resize(example["image"], image_size)
    image = tf.cast(image, tf.float32)
    if backbone.lower() == "mobilenet_v2":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif backbone.lower() == "resnet50":
        image = tf.keras.applications.resnet50.preprocess_input(image)
    elif backbone.lower() == "efficientnetb0":
        pass
    else:
        image = image / 255.0
    label = tf.cast(example["label"], tf.int32)
    return image, label


def preprocess_segmentation(example, image_size=SEGMENTATION_SIZE):
    """Prepare one TFDS example for 3-class trimap segmentation."""
    image = tf.image.resize(example["image"], image_size)
    image = normalize_image_01(image)

    mask = example["segmentation_mask"]
    mask = tf.image.resize(mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.int32) - 1
    mask = tf.clip_by_value(mask, 0, 2)
    return image, mask


def classification_augmentation() -> tf.keras.Sequential:
    """Moderate augmentation that preserves breed identity."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="classification_augmentation",
    )


def apply_segmentation_random_flip(image: tf.Tensor, mask: tf.Tensor, seed: int | None = None):
    """Apply the same left-right flip to an image and mask."""
    if seed is None:
        seed = tf.random.uniform([], maxval=1_000_000, dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed=[seed, 0])
    mask = tf.image.stateless_random_flip_left_right(mask, seed=[seed, 0])
    return image, mask


def make_pet_binary_mask(mask: tf.Tensor) -> tf.Tensor:
    """Convert 3-class trimap prediction/label into a pet-vs-background mask."""
    return tf.cast(tf.equal(mask, 0), tf.float32)
