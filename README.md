# Hair-Based Gender Classifier

## Problem Statement

Build a model that classifies gender based on the following logic:
- **Age 20–30**: Long hair → **Female**, Short hair → **Male** (regardless of actual gender)
- **Age outside 20–30**: Classify by **actual gender** (ignore hair length)

---

## Model Architecture (Custom CNN — No Pre-Trained Weights)

Built entirely from scratch using TensorFlow/Keras:

```
Input (128×128×3)
    │
    ├── Conv Block 1: Conv2D(32) × 2 → BN → MaxPool → Dropout(0.25)
    ├── Conv Block 2: Conv2D(64) × 2 → BN → MaxPool → Dropout(0.25)
    ├── Conv Block 3: Conv2D(128) × 2 → BN → MaxPool → Dropout(0.30)
    ├── Conv Block 4: Conv2D(256) × 2 → BN → MaxPool → Dropout(0.30)
    │
    └── GlobalAveragePooling → Dense(512) → BN → Dropout(0.5)
              │
              ├── HEAD 1: Dense(128) → age_output   (sigmoid) → Is age 20-30?
              ├── HEAD 2: Dense(128) → hair_output  (sigmoid) → Is hair long?
              └── HEAD 3: Dense(128) → gender_output(sigmoid) → Is person female?
```

**Decision Logic:**
```python
if age_in_range (20-30):
    prediction = "Female" if long_hair else "Male"
else:
    prediction = actual_gender_from_head3
```

---

## Dataset

**CelebA** (Large-scale CelebFaces Attributes Dataset)
- 202,599 face images with 40 attributes
- Key attributes used: `Long_Hair`, `Male`, `Young`
- `Young` attribute used as proxy for age range 20–30
- Downloaded via Kaggle: `jessicali9530/celeba-dataset`

---

## Setup & Run

### Option 1 — Google Colab (Recommended)
1. Open `Hair_gender_classifier_FINAL.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells: Runtime → Run all (Ctrl+F9)
4. Upload `kaggle.json` when prompted (get it from kaggle.com → Settings → API)
5. Upload your own photo when prompted in Step 12

### Option 2 — Local
```bash
pip install tensorflow numpy opencv-python Pillow scikit-learn matplotlib seaborn pandas gdown
python train.py
python gui.py
```

---

## Results

| Metric              | Value  |
|---------------------|--------|
| Hair Length Acc.    | ~88%   |
| Age Range Acc.      | ~82%   |
| Final Gender Acc.   | ~85%   |

*(Results after training on CelebA samples)*

---

## Project Files

| File                                    | Description                                      |
|-----------------------------------------|--------------------------------------------------|
| `Hair_gender_classifier_FINAL.ipynb`    | Main notebook — full pipeline end to end         |
| `hair_gender_model.h5`                  | Trained model weights (generated after training) |
| `training_curves.png`                   | Loss & accuracy plots (generated after training) |
| `confusion_matrices.png`                | Evaluation confusion matrices                    |
| `sample_predictions.png`                | Sample predictions on validation set             |
| `prediction_result.png`                 | Result from your own uploaded photo              |
| `README.md`                             | This file                                        |

---

## Methodology

1. **Preprocessing**: Resize to 128×128, normalize to [0,1]
2. **Augmentation**: Rotation, flips, brightness, zoom
3. **Multi-task Learning**: 3 output heads trained simultaneously
4. **Loss Weighting**: Hair head weighted 1.5× (most critical for task logic)
5. **Regularization**: L2 + BatchNorm + Dropout
6. **Callbacks**: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau

---

## Task Logic Explained

The classifier follows a rule-based decision on top of the CNN outputs:

```
Person detected as Age 20–30?
    YES → Use hair length to decide gender
            Long hair  → Female
            Short hair → Male
    NO  → Use the actual gender prediction head directly
```

This means for the 20–30 age group, a male with long hair is classified as Female,
and a female with short hair is classified as Male — by design.

---

## GUI Features (Colab)

- Upload any face image (jpg/png)
- Instant gender prediction with confidence score
- Shows: age range detected, hair length detected, decision rule used
- Dark-themed result visualization

---

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab (T4 GPU)
