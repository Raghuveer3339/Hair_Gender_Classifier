# 💇 Task 1 — Long Hair Identification (Hair-Based Gender Classifier)

> **Internship Task** | Machine Learning | Custom CNN from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raghuveer3339/Hair_Gender_Classifier/blob/main/Task1_Hair_Gender_Classifier_UPDATED.ipynb)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-CelebA-green)](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

---

## 📌 Problem Statement

Build a machine learning model that classifies gender based on **hair length** with the following conditional logic:

| Age Range | Classification Rule |
|---|---|
| **20–30 years** | Long hair → **Female** (even if biologically male) |
| **20–30 years** | Short hair → **Male** (even if biologically female) |
| **Outside 20–30** | Classify by **actual biological gender** regardless of hair |

The model must include a **Graphical User Interface (GUI)** for uploading and testing images.

---

## 🧠 Approach

### Multi-Task CNN with 3 Output Heads

A single custom CNN backbone predicts all three required attributes simultaneously:

```
Input Image (128×128×3)
         │
         ▼
┌─────────────────────────────┐
│   Custom CNN Backbone       │
│   4 Conv Blocks             │
│   (No pre-trained weights)  │
└──────────┬──────────────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  HEAD1  HEAD2  HEAD3
  Age    Hair   Gender
  (20-30?) (Long?) (Female?)
           │
           ▼
    Decision Logic:
    if age_in_range (20-30):
        Female if long hair else Male
    else:
        Female if actual gender else Male
```

### Decision Logic (Inference)
```python
if age_in_range:   # age 20-30
    gender = 'Female' if is_long_hair else 'Male'
    reason = 'Hair-length rule applied (age 20–30)'
else:              # outside 20-30
    gender = 'Female' if is_female else 'Male'
    reason = 'Actual gender used (outside 20–30)'
```

---

## 🏗️ Model Architecture

**Custom CNN — Built entirely from scratch. Zero pre-trained weights. Zero transfer learning.**

```
Input (128×128×3)
→ Conv Block 1: Conv2D(32)×2 + BatchNorm + MaxPool + Dropout(0.25)
→ Conv Block 2: Conv2D(64)×2 + BatchNorm + MaxPool + Dropout(0.25)
→ Conv Block 3: Conv2D(128)×2 + BatchNorm + MaxPool + Dropout(0.30)
→ Conv Block 4: Conv2D(256)×2 + BatchNorm + MaxPool + Dropout(0.30)
→ GlobalAveragePooling2D
→ Dense(512) + BatchNorm + Dropout(0.5)
→ HEAD 1: age_output    (sigmoid) — Is person aged 20–30?
→ HEAD 2: hair_output   (sigmoid) — Is hair long?
→ HEAD 3: gender_output (sigmoid) — Is person female?
```

**Training strategy:** Multi-task learning with weighted losses, early stopping, and learning rate scheduling.

---

## 📊 Dataset

**CelebA** (Large-scale CelebFaces Attributes Dataset)
- **202,599** celebrity face images
- **40 binary attributes** per image
- Attributes used:

| Attribute | Used For |
|---|---|
| `Long_Hair` | Primary hair label |
| `Wavy_Hair` + `Straight_Hair` - `Bald` - `Receding_Hairline` | Derived hair label (fallback) |
| `Young` | Proxy for age 20–30 range |
| `Male` | Actual biological gender label |

> **Note on Age:** CelebA does not provide exact age values. The `Young` attribute serves as a proxy for the 20–30 age range. This is a dataset limitation — the model correctly implements the task logic using this best available signal.

- Download via Kaggle: `jessicali9530/celeba-dataset`

---

## 🔬 Training Details

| Parameter | Value |
|---|---|
| Input size | 128×128×3 |
| Max samples | Configurable (default 30,000) |
| Train/Val split | 80/20 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss | Binary Crossentropy (per head) |
| Callbacks | EarlyStopping · ReduceLROnPlateau · ModelCheckpoint |
| Augmentation | Rotation · Shift · Flip · Brightness · Zoom |

---

## 📈 Visual Outputs

The notebook generates and saves to Google Drive:

| Output | Description |
|---|---|
| `training_curves.png` | Accuracy + loss curves for all 3 heads |
| `confusion_matrices.png` | Hair / Age / Final gender confusion matrices |
| `sample_predictions.png` | 12 validation samples with predictions |
| `sample_images.png` | CelebA sample images with labels |
| `prediction_result.png` | GUI prediction result visualization |
| `hair_gender_model.h5` | Saved trained model |

---

## 📁 Project Structure

```
Hair_Gender_Classifier/
│
├── Task1_Hair_Gender_Classifier_UPDATED.ipynb   # Main Colab notebook
└── README.md
```

**Google Drive** (auto-created by notebook):
```
Internship_Datasets/Task1_Gender_Classification/
├── Models/
│   └── hair_gender_model.h5
└── Outputs/
    ├── training_curves.png
    ├── confusion_matrices.png
    ├── sample_predictions.png
    ├── sample_images.png
    └── prediction_result.png
```

---

## 🚀 How to Run

1. Click **Open in Colab** badge above
2. `Runtime → Change runtime type → T4 GPU`
3. Run **Step 1** → install dependencies
4. Run **Step 2b** → mount Google Drive
5. Run **Step 2** → upload `kaggle.json` and download CelebA
6. Run all remaining cells top to bottom
7. **Step 12** → GUI: upload your own face image to test

---

## 🖥️ GUI Features

Built with **ipywidgets** inside Google Colab:
- 📂 Upload any face image (jpg/png)
- ⚡ Instant prediction with decision logic applied
- 📊 Shows: Gender · Confidence · Age Range · Hair Length · Rule Used
- 🎨 Visual result with annotated image display

---

## 📦 Requirements

```
tensorflow>=2.21.0
opencv-python-headless
numpy
pandas
scikit-learn
matplotlib
seaborn
pillow
kaggle
```

---

## 📋 Results Summary

| Head | Task | Classes |
|---|---|---|
| Age Head | Is person 20–30? | In Range / Outside |
| Hair Head | Hair length? | Long / Short |
| Gender Head | Biological gender? | Female / Male |
| **Final Output** | **Task logic applied** | **Female / Male** |

---

## 👤 Author

**Raghuveer** | Internship Project — Task 1
Built with Python · TensorFlow · OpenCV · CelebA
