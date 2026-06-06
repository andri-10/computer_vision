"""U-Net-style segmentation model builders."""

from __future__ import annotations

import tensorflow as tf


NUM_MASK_CLASSES = 3


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def build_unet(input_shape=(160, 160, 3), num_classes: int = NUM_MASK_CLASSES) -> tf.keras.Model:
    """Build a compact U-Net with raw-logit output."""
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    bridge = conv_block(p4, 512)

    u4 = tf.keras.layers.UpSampling2D()(bridge)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)
    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)
    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)
    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding="same")(c8)
    return tf.keras.Model(inputs, outputs, name="petvision_unet")


def compile_segmenter(model: tf.keras.Model, learning_rate: float = 1e-3, from_logits: bool = True) -> tf.keras.Model:
    """Compile a sparse-label segmentation model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
        metrics=["accuracy"],
    )
    return model
