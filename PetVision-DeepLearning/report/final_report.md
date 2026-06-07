# PetVision Final Report

## 1. Introduction

PetVision is a deep learning computer vision project for pet breed classification, pet segmentation, and model explainability. The final system combines a baseline CNN, a MobileNetV2 transfer learning classifier, a U-Net-style segmentation model, and Grad-CAM visualization utilities.

The main classification requirement is satisfied by two classifiers: a simple CNN trained from scratch and a transfer learning model. The segmentation model and Grad-CAM explainability extend the project beyond the minimum requirement.

## 2. Dataset

The project uses the Oxford-IIIT Pet Dataset from TensorFlow Datasets:

```python
tfds.load("oxford_iiit_pet:4.*.*", with_info=True, as_supervised=False)
```

Dataset details:

- 37 cat and dog breed classes
- 3,680 official training images
- 3,669 official test images
- Approximately 200 images per class
- Breed labels and pixel-level trimap segmentation masks

The trimap masks use labels `1`, `2`, and `3`. For model training, these were converted to `0`, `1`, and `2`, and masks were resized using nearest-neighbor interpolation to preserve class labels.

Key saved dataset figures:

- `results/figures/sample_images.png`
- `results/figures/sample_masks.png`
- `results/figures/class_distribution.png`
- `results/figures/image_size_distribution.png`

## 3. Methodology

The official train and test splits were loaded from TensorFlow Datasets. The training split was deterministically divided into training and validation subsets.

Classification preprocessing:

- Baseline CNN: images resized to `128x128`, normalized to `[0, 1]`
- Transfer learning: images resized to `224x224`, preprocessed with MobileNetV2 preprocessing
- Labels kept as sparse integer labels
- Loss: sparse categorical crossentropy

Segmentation preprocessing:

- Images and masks resized to `160x160`
- Images normalized to `[0, 1]`
- Masks resized with nearest-neighbor interpolation
- Mask labels converted from `1,2,3` to `0,1,2`
- Loss: sparse categorical crossentropy from logits

Training settings used for the final pulled results:

| Component | Setting |
|---|---|
| Baseline CNN | 3 epochs, batch size 32 |
| MobileNetV2 head training | 3 epochs, batch size 32 |
| MobileNetV2 fine-tuning | 3 epochs, batch size 32 |
| U-Net segmentation | 3 epochs, batch size 16, image size 160x160 |

## 4. Baseline CNN Classification

The baseline CNN was trained from scratch using resized `128x128` images. It used convolution, batch normalization, max pooling, dropout, global average pooling, and a dense softmax classifier with 37 outputs.

Results:

| Metric | Value |
|---|---:|
| Test accuracy | 3.62% |
| Top-3 accuracy | 8.99% |
| Macro precision | 0.34% |
| Macro recall | 3.59% |
| Macro F1-score | 0.56% |
| Model size | 3.88 MB |

The baseline performed poorly because it was trained from scratch for only a short run and the dataset contains many visually similar breeds. This result is still useful because it shows why transfer learning is important for this task.

Saved outputs:

- `results/classification/baseline_training_curves.png`
- `results/classification/baseline_confusion_matrix.png`
- `results/classification/baseline_wrong_predictions.png`

## 5. Transfer Learning Classification

The final classifier uses MobileNetV2 pretrained on ImageNet with `include_top=False`. The base model was first frozen while training the classification head, then upper layers were fine-tuned with a low learning rate.

Results:

| Metric | Value |
|---|---:|
| Test accuracy | 79.07% |
| Top-3 accuracy | 94.06% |
| Macro precision | 78.73% |
| Macro recall | 78.76% |
| Macro F1-score | 77.51% |
| Weighted F1-score | 77.81% |
| Model size | 22.43 MB |
| Inference time per batch | 134.24 ms |

The transfer model strongly outperformed the baseline CNN. The high Top-3 accuracy suggests that many remaining mistakes are near-misses among visually similar breeds.

Saved outputs:

- `models/best_classifier.keras`
- `results/classification/transfer_training_curves.png`
- `results/classification/transfer_confusion_matrix.png`
- `results/classification/model_comparison_table.csv`

## 6. U-Net Segmentation

The segmentation model is a compact U-Net-style architecture trained to predict three trimap classes. It outputs raw logits with shape `(H, W, 3)` and uses sparse categorical crossentropy from logits.

Results:

| Metric | Value |
|---|---:|
| Pixel accuracy | 83.97% |
| Mean IoU | 63.59% |
| Dice coefficient | 76.10% |
| Model size | 90.13 MB |

The segmentation model produces visually useful masks after a short training run. Some errors are expected around object boundaries, thin fur regions, and backgrounds with colors or textures similar to the pet.

Saved outputs:

- `models/best_segmenter.keras`
- `results/segmentation/segmentation_training_curves.png`
- `results/segmentation/segmentation_metrics.csv`
- `results/segmentation/mask_examples.png`

## 7. Grad-CAM Explainability

Grad-CAM is implemented in `src/gradcam.py` and is used in the demo inference workflow. It builds a gradient model around the MobileNetV2 convolutional backbone and overlays the resulting heatmap on the input image.

The Grad-CAM component is intended to answer whether the classifier focuses on:

- the pet face
- body shape
- fur texture
- breed-specific regions
- or irrelevant background regions

The demo output in `results/figures/demo_inference.png` includes the Grad-CAM overlay as part of the full inference pipeline. Dedicated Grad-CAM result files can be regenerated by rerunning `notebooks/05_gradcam_interpretability.ipynb`.

## 8. Results Summary

Classification:

| Model | Accuracy | Top-3 Accuracy | Macro F1 | Model Size |
|---|---:|---:|---:|---:|
| Baseline CNN | 3.62% | 8.99% | 0.56% | 3.88 MB |
| MobileNetV2 transfer | 79.07% | 94.06% | 77.51% | 22.43 MB |

Segmentation:

| Model | Pixel Accuracy | Mean IoU | Dice | Model Size |
|---|---:|---:|---:|---:|
| U-Net | 83.97% | 63.59% | 76.10% | 90.13 MB |

The final project demonstrates a clear improvement from a simple CNN baseline to a pretrained transfer learning model. The segmentation results add a second dense prediction task and make the project more complete visually and technically.

## 9. Error Analysis

The baseline CNN mostly failed to learn meaningful breed-level features in the short run. It predicted only a small number of classes with any frequency and had very low macro F1-score.

The transfer model showed much better performance, but some classes remained difficult. The class `Bombay` had zero recall in the saved classification report, and several cat breeds such as `Bengal`, `Maine_Coon`, `Russian_Blue`, and `Ragdoll` were more challenging than many dog breeds. This is likely because these categories can share similar color, face shape, or fur patterns.

The high Top-3 accuracy indicates that even when the Top-1 prediction is wrong, the correct breed is often still among the strongest candidates.

For segmentation, the most likely failure cases are:

- fuzzy boundaries around fur
- pets blending into similar backgrounds
- small or thin body parts
- uncertain trimap border regions

## 10. Demo

The demo inference notebook loads both saved models and shows:

1. input image
2. predicted breed
3. Top-3 breed probabilities
4. Grad-CAM overlay
5. predicted segmentation mask
6. extracted pet area

Saved demo output:

- `results/figures/demo_inference.png`

The Streamlit app in `demo/app.py` provides a lightweight interactive version after the saved models are available.

## 11. Conclusion

PetVision successfully implements breed classification, pet segmentation, and explainability on the Oxford-IIIT Pet Dataset. The MobileNetV2 transfer learning classifier is the strongest classification model, reaching 79.07% Top-1 accuracy and 94.06% Top-3 accuracy on the test set. The U-Net segmenter provides useful pet masks with 83.97% pixel accuracy and 76.10% Dice coefficient.

The project satisfies the classification requirement and adds segmentation, Grad-CAM, clean visual outputs, reusable source code, saved models, and a demo workflow.

## 12. Future Work

Potential improvements:

- train MobileNetV2 and U-Net for more epochs
- try EfficientNetB0 for classification
- evaluate segmented-pet-only classification
- generate more Grad-CAM examples for correct and incorrect predictions
- use stronger segmentation augmentation
- deploy the Streamlit demo publicly

## 13. References

- Christian Mata, PhD: Computer Vision, EMSSE
- Oxford-IIIT Pet Dataset
- TensorFlow Datasets
- Keras Applications
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
