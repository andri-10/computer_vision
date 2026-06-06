"""Grad-CAM explainability utilities."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the last convolutional layer in a possibly nested Keras model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            try:
                return f"{layer.name}/{find_last_conv_layer(layer)}"
            except ValueError:
                continue
    raise ValueError("No Conv2D layer found. Pass last_conv_layer_name manually.")


def _get_layer_output(model: tf.keras.Model, layer_name: str):
    """Resolve nested layer names written as parent/child."""
    if "/" not in layer_name:
        return model.get_layer(layer_name).output
    current = model
    parts = layer_name.split("/")
    for part in parts[:-1]:
        current = current.get_layer(part)
    return current.get_layer(parts[-1]).output


def make_gradcam_heatmap(
    image_batch: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for the first image in a batch."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_output = _get_layer_output(model, last_conv_layer_name)
    grad_model = tf.keras.Model(model.inputs, [last_conv_output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def top_k_predictions(probabilities: np.ndarray, label_names: list[str], k: int = 3):
    """Return top-k class names and probabilities."""
    probs = np.asarray(probabilities)
    top_indices = probs.argsort()[-k:][::-1]
    return [(label_names[idx], float(probs[idx])) for idx in top_indices]


def make_nested_backbone_gradcam_heatmap(
    image_batch: tf.Tensor,
    model: tf.keras.Model,
    backbone_layer_name: str,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    """Generate Grad-CAM for models that contain a nested pretrained backbone.

    This matches the transfer-learning architecture in src.models_classification:
    Input -> augmentation -> pretrained backbone -> pooling/dropout/dense head.
    """
    backbone_index = [layer.name for layer in model.layers].index(backbone_layer_name)
    preprocessing_layers = model.layers[1:backbone_index]
    head_layers = model.layers[backbone_index + 1 :]
    backbone = model.get_layer(backbone_layer_name)
    conv_layer = backbone.get_layer(last_conv_layer_name)
    feature_model = tf.keras.Model(backbone.input, [conv_layer.output, backbone.output])

    with tf.GradientTape() as tape:
        x = image_batch
        for layer in preprocessing_layers:
            x = layer(x, training=False)
        conv_outputs, x = feature_model(x, training=False)
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()
