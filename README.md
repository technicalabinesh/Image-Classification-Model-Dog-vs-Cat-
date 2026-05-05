# 🐶🐱 Image Classification Model — Dog vs Cat

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies images as either a **dog** or a **cat**. The model is trained on the [`cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) dataset from TensorFlow Datasets and achieves strong validation accuracy using data augmentation and regularisation techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Training](#training)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

| Property        | Value                          |
|-----------------|--------------------------------|
| Task            | Binary Image Classification    |
| Classes         | Cat (0) / Dog (1)              |
| Input Size      | 64 × 64 × 3 (RGB)              |
| Framework       | TensorFlow / Keras             |
| Dataset         | `cats_vs_dogs` (TFDS v4.0.1)   |
| Training Split  | 80 % train / 20 % validation   |
| Total Samples   | ~23 262 (after corrupt removal)|

---

## Model Architecture

A Sequential CNN with three convolutional blocks followed by a dense classifier head:

```
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Flatten → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

**Total parameters:** ~1.2 M  
**Loss:** Binary Cross-Entropy  
**Optimiser:** Adam  
**Output:** Sigmoid probability (> 0.5 → Dog, ≤ 0.5 → Cat)

---

## Dataset

The [`cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) dataset is loaded automatically via `tensorflow_datasets`:

```python
import tensorflow_datasets as tfds
ds = tfds.load("cats_vs_dogs", split="train", as_supervised=True)
```

- ~25 000 labelled images of cats and dogs
- ~1 738 corrupt images are skipped automatically by TFDS
- No separate test split is provided by TFDS; an 80/20 in-dataset split is used

---

## Data Preprocessing & Augmentation

**Preprocessing (train + validation):**
- Resize to 64 × 64
- Normalise pixel values to [0, 1]

**Augmentation (train only):**
- Random horizontal & vertical flip
- Random brightness adjustment (±0.2)
- Random contrast adjustment (0.8 – 1.2)
- Random saturation adjustment (0.8 – 1.2)
- Random zoom via crop + resize (up to 20 % zoom)

---

## Training

```python
history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data,
    callbacks=callbacks
)
```

**Callbacks used:**

| Callback              | Purpose                                           |
|-----------------------|---------------------------------------------------|
| `EarlyStopping`       | Stops training if `val_loss` doesn't improve for 3 epochs and restores best weights |
| `ReduceLROnPlateau`   | Halves learning rate if `val_loss` plateaus for 2 epochs |
| `ModelCheckpoint`     | Saves the best model to `best_model.keras` based on `val_accuracy` |

---

## Results

Training curves (accuracy & loss) are saved to `training_curves.png` after running the notebook.

---

## How to Run

### On Kaggle (recommended)
1. Open `cat-vs-dog-classification-model.ipynb` in a Kaggle Notebook with **GPU** accelerator enabled.
2. Run all cells — the dataset is downloaded automatically via TFDS.
3. After training, use the upload widget at the bottom of the notebook to classify your own image.

### Locally
```bash
# 1. Install dependencies
pip install tensorflow tensorflow-datasets matplotlib pillow ipywidgets

# 2. Launch the notebook
jupyter notebook cat-vs-dog-classification-model.ipynb
```

### Predict on a custom image
An interactive `ipywidgets` upload widget is included at the end of the notebook:
1. Click **Upload** and select a `.jpg` / `.png` image of a cat or dog.
2. Click **Predict**.
3. The model displays the image with the predicted label and confidence score.

---

## Requirements

| Package               | Version  |
|-----------------------|----------|
| Python                | ≥ 3.10   |
| TensorFlow            | ≥ 2.15   |
| tensorflow-datasets   | ≥ 4.9    |
| NumPy                 | ≥ 1.24   |
| Matplotlib            | ≥ 3.7    |
| Pillow                | ≥ 10.0   |
| ipywidgets            | ≥ 8.0    |

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
