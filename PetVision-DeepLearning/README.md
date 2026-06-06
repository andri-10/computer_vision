# PetVision: Pet Breed Classification, Segmentation, and Explainability

PetVision is a university Computer Vision / Deep Learning final project built on the Oxford-IIIT Pet Dataset. It classifies 37 cat and dog breeds, segments pets from the background with a U-Net-style model, and visualizes classifier attention with Grad-CAM.

## Team Members

- Student 1: Data exploration, baseline CNN, evaluation plots
- Student 2: Transfer learning, fine-tuning, Grad-CAM
- Student 3: Segmentation, demo, README/report integration

## Dataset

The project uses the Oxford-IIIT Pet Dataset through TensorFlow Datasets:

```python
tfds.load("oxford_iiit_pet:4.*.*", with_info=True, as_supervised=False)
```

The dataset contains 37 pet categories with roughly 200 images per class. The current TensorFlow Datasets release exposes this dataset as `oxford_iiit_pet:4.0.0`, including breed labels, species labels, and pixel-level trimap segmentation masks.

## Main Features

- Breed classification with a baseline CNN
- Transfer learning with MobileNetV2 or EfficientNetB0
- Top-1 and Top-3 classification accuracy
- U-Net-style trimap segmentation
- Grad-CAM explainability for classifier predictions
- Error analysis with confusion matrix and wrong predictions
- Optional Streamlit demo for inference

## Repository Structure

```text
PetVision-DeepLearning/
|-- README.md
|-- requirements.txt
|-- report/
|-- notebooks/
|-- src/
|-- models/
|-- results/
`-- demo/
```

## Local Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Running on Google Colab with Git

This project is designed to work with a GitHub-based Colab workflow:

- Push changes from your PC to GitHub.
- Pull the newest changes inside Colab before running notebooks.
- Commit and push Colab changes only when you want to keep updated notebooks, reports, or result files.

First, push this project to GitHub from your PC:

```bash
git add PetVision-DeepLearning
git commit -m "Add PetVision project scaffold"
git remote add origin https://github.com/YOUR_USERNAME/PetVision-DeepLearning.git
git push -u origin main
```

In Colab, either open a notebook from GitHub or create a blank notebook and clone the repository:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/PetVision-DeepLearning.git"
PROJECT_DIR = "/content/PetVision-DeepLearning"

!git clone $REPO_URL $PROJECT_DIR
%cd $PROJECT_DIR
!pip install -q -r requirements.txt
```

At the top of every project notebook there is a **Google Colab Git Setup** section. Replace the `REPO_URL` placeholder with your real GitHub repository URL, then run the setup cell. It will:

- clone the repository into `/content/PetVision-DeepLearning` if it is missing
- run `git pull` if the repository already exists
- install dependencies from `requirements.txt`
- switch the notebook working directory to the project root

If you are running locally, skip the Colab Git setup cell and continue with the normal notebook cells.

Before training, enable a GPU in Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU -> Save
```

Useful Colab Git commands:

```bash
# Pull latest work from GitHub
git pull

# Check changed files
git status

# Commit and push Colab changes
git add notebooks results report README.md
git commit -m "Update training results"
git push
```

For private repositories, Colab may ask for GitHub authentication. Use a GitHub personal access token or keep the repository public during the class project.

## How to Run

Run notebooks in this order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_baseline_cnn_classification.ipynb`
3. `notebooks/03_transfer_learning_classification.ipynb`
4. `notebooks/04_unet_segmentation.ipynb`
5. `notebooks/05_gradcam_interpretability.ipynb`
6. `notebooks/06_demo_inference.ipynb`

The transfer notebook saves `models/best_classifier.keras`. The segmentation notebook saves `models/best_segmenter.keras`.

Notebooks 05 and 06 depend on the saved models from notebooks 03 and 04, so run the training notebooks first.

For quick testing in Colab, reduce epochs in the training notebooks:

```python
EPOCHS = 3
HEAD_EPOCHS = 3
FINE_TUNE_EPOCHS = 3
```

For final results, increase them again:

```python
EPOCHS = 15
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 8
```

The first run downloads the Oxford-IIIT Pet Dataset through TensorFlow Datasets, so notebook 01 may take a few minutes the first time.

## Expected Results Table

Fill this table after training:

| Model | Input Size | Test Accuracy | Top-3 Accuracy | Macro F1 | Notes |
|---|---:|---:|---:|---:|---|
| Baseline CNN | 128x128 | TBD | TBD | TBD | Trained from scratch |
| MobileNetV2 transfer | 224x224 | TBD | TBD | TBD | ImageNet pretrained |
| Segmented-pet classifier | 224x224 | Optional | Optional | Optional | Extra comparison |

Segmentation:

| Model | Input Size | Pixel Accuracy | Mean IoU | Dice |
|---|---:|---:|---:|---:|
| U-Net | 160x160 | TBD | TBD | TBD |

## Example Outputs

Add generated figures after running notebooks:

- `results/figures/sample_images.png`
- `results/figures/sample_masks.png`
- `results/classification/transfer_confusion_matrix.png`
- `results/segmentation/mask_examples.png`
- `results/gradcam/correct_predictions_gradcam.png`

## Demo

After training both models:

```bash
streamlit run demo/app.py
```

The demo predicts breed, shows Top-3 probabilities, creates a Grad-CAM overlay, predicts the segmentation mask, and extracts the pet area.

In Colab, the easiest demo path is `notebooks/06_demo_inference.ipynb`. Running the Streamlit app from Colab requires extra tunneling setup, so the notebook demo is recommended for the final presentation unless a web app is specifically required.

## Credits and References

- Oxford-IIIT Pet Dataset
- TensorFlow Datasets
- Keras Applications
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
