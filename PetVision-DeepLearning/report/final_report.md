# PetVision Final Report

## 1. Introduction

PetVision investigates pet breed classification, pet segmentation, and model explainability using the Oxford-IIIT Pet Dataset. The project combines a baseline CNN, a transfer learning classifier, a U-Net-style segmenter, and Grad-CAM visualization.

## 2. Dataset

The Oxford-IIIT Pet Dataset contains 37 cat and dog breed categories with roughly 200 images per class. The current TensorFlow Datasets release exposes this dataset as `oxford_iiit_pet:4.0.0`, with breed labels and pixel-level trimap masks.

Include:

- Dataset size and splits
- Class distribution plot
- Example images
- Example trimap masks
- Notes on image size variation

## 3. Methodology

Describe:

- TensorFlow Datasets loading
- Train/validation/test split
- Classification preprocessing
- Segmentation preprocessing
- Data augmentation
- Model saving and reproducibility settings

## 4. Baseline CNN Classification

Describe:

- Architecture
- Input size
- Loss and optimizer
- Training settings
- Test accuracy, precision, recall, and F1-score
- Confusion matrix and wrong predictions

## 5. Transfer Learning Classification

Describe:

- MobileNetV2 or EfficientNetB0 backbone
- ImageNet initialization
- Frozen-head training
- Fine-tuning strategy
- Top-1 and Top-3 accuracy
- Comparison with baseline CNN

## 6. U-Net Segmentation

Describe:

- Trimap label conversion from `1,2,3` to `0,1,2`
- Nearest-neighbor mask resizing
- U-Net architecture
- Loss function
- Pixel accuracy, mean IoU, and Dice coefficient
- Visual mask examples

## 7. Grad-CAM Explainability

Describe:

- Last convolutional layer used
- Correct predictions
- Incorrect predictions
- Whether the model focuses on face, body, fur, or background

## 8. Results

Classification table:

| Model | Accuracy | Top-3 Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | TBD | TBD | TBD | TBD | TBD |
| Transfer learning | TBD | TBD | TBD | TBD | TBD |

Segmentation table:

| Model | Pixel Accuracy | Mean IoU | Dice |
|---|---:|---:|---:|
| U-Net | TBD | TBD | TBD |

## 9. Error Analysis

Discuss:

- Most confused breeds
- Cat vs dog errors
- Visual similarities between breeds
- Background bias revealed by Grad-CAM
- Segmentation failures near borders or occlusions

## 10. Conclusion

Summarize the strongest result and explain how transfer learning, segmentation, and interpretability improved the project beyond a basic classifier.

## 11. Future Work

Possible extensions:

- Stronger backbone
- More segmentation augmentations
- Segmented-pet-only classification
- Test-time augmentation
- Deployment-ready web demo

## 12. References

- Oxford-IIIT Pet Dataset
- TensorFlow Datasets
- Keras Applications
- Grad-CAM paper
