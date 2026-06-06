"""Classification model builders."""

from __future__ import annotations

import tensorflow as tf

from src.preprocessing import classification_augmentation


NUM_CLASSES = 37


def build_baseline_cnn(
    input_shape=(128, 128, 3),
    num_classes: int = NUM_CLASSES,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """Build a compact CNN baseline from scratch."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if use_augmentation:
        x = classification_augmentation()(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_classifier(
    input_shape=(224, 224, 3),
    num_classes: int = NUM_CLASSES,
    backbone_name: str = "MobileNetV2",
    dropout_rate: float = 0.30,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a pretrained image classifier and return model plus base model."""
    if backbone_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = classification_augmentation()(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name.lower()}_pet_classifier")
    return model, base_model


def compile_classifier(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile a sparse-label classifier with Top-1 and Top-3 metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model


def unfreeze_for_fine_tuning(base_model: tf.keras.Model, fine_tune_at: int | None = None) -> None:
    """Unfreeze the upper part of a pretrained backbone for low-LR fine-tuning."""
    base_model.trainable = True
    if fine_tune_at is None:
        fine_tune_at = int(len(base_model.layers) * 0.75)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
