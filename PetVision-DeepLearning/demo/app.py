from pathlib import Path
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_label_mapping  # noqa: E402
from src.gradcam import make_nested_backbone_gradcam_heatmap, top_k_predictions  # noqa: E402
from src.preprocessing import make_pet_binary_mask  # noqa: E402
from src.visualization import colorize_mask, overlay_heatmap  # noqa: E402


CLASSIFIER_PATH = ROOT / "models" / "best_classifier.keras"
SEGMENTER_PATH = ROOT / "models" / "best_segmenter.keras"
LABEL_MAPPING_PATH = ROOT / "results" / "figures" / "label_mapping.json"


def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def preprocess_for_classifier(image):
    resized = tf.image.resize(image, (224, 224))
    resized = tf.cast(resized, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(resized)[tf.newaxis, ...]


def preprocess_for_segmenter(image):
    resized = tf.image.resize(image, (160, 160))
    resized = tf.cast(resized, tf.float32) / 255.0
    return resized[tf.newaxis, ...]


st.set_page_config(page_title="PetVision Demo", layout="wide")
st.title("PetVision Demo")

if not CLASSIFIER_PATH.exists() or not SEGMENTER_PATH.exists():
    st.warning("Train the classifier and segmenter notebooks first so the saved .keras models exist.")

uploaded = st.file_uploader("Upload a pet image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = load_image(uploaded)
    st.image(image, caption="Input image", use_container_width=True)

    if CLASSIFIER_PATH.exists() and LABEL_MAPPING_PATH.exists():
        classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
        label_mapping = load_label_mapping(LABEL_MAPPING_PATH)
        label_names = [label_mapping[idx] for idx in sorted(label_mapping)]
        classifier_input = preprocess_for_classifier(image)
        probabilities = classifier.predict(classifier_input, verbose=0)[0]
        top3 = top_k_predictions(probabilities, label_names, k=3)

        st.subheader("Breed prediction")
        st.write(f"Predicted breed: **{top3[0][0]}** ({top3[0][1]:.2%})")
        st.table([{"breed": name, "probability": f"{prob:.2%}"} for name, prob in top3])

        try:
            heatmap = make_nested_backbone_gradcam_heatmap(
                classifier_input,
                classifier,
                backbone_layer_name="mobilenetv2_1.00_224",
                last_conv_layer_name="Conv_1",
            )
            heatmap = tf.image.resize(heatmap[..., np.newaxis], image.shape[:2]).numpy().squeeze()
            st.image(overlay_heatmap(image, heatmap), caption="Grad-CAM overlay", use_container_width=True)
        except Exception as exc:
            st.info(f"Grad-CAM could not be generated automatically: {exc}")

    if SEGMENTER_PATH.exists():
        segmenter = tf.keras.models.load_model(SEGMENTER_PATH)
        segmenter_input = preprocess_for_segmenter(image)
        logits = segmenter.predict(segmenter_input, verbose=0)[0]
        pred_mask = np.argmax(logits, axis=-1)
        pet_mask = make_pet_binary_mask(pred_mask).numpy()
        mask_rgb = colorize_mask(pred_mask)
        mask_resized = tf.image.resize(pet_mask[..., np.newaxis], image.shape[:2], method="nearest").numpy()
        extracted = image * mask_resized

        col1, col2 = st.columns(2)
        col1.image(mask_rgb, caption="Predicted segmentation mask", use_container_width=True)
        col2.image(extracted.astype(np.uint8), caption="Extracted pet area", use_container_width=True)
