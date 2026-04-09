# VisualGuard: Deepfake Detection using EfficientNet

## Overview

VisualGuard is a deep learning-based system designed to detect deepfake images by analyzing spatial inconsistencies in facial features. The current implementation focuses on image-level classification using a Convolutional Neural Network built on EfficientNetB0.

The system leverages transfer learning and data augmentation to improve generalization and achieve robust performance in distinguishing between real and fake facial images.

---

## Problem Statement

With the rapid advancement of deepfake generation techniques, identifying manipulated media has become increasingly important. Traditional detection methods struggle to capture subtle visual artifacts introduced by modern deepfake models.

This project aims to build an efficient and scalable deepfake detection model using deep learning techniques that can identify such manipulations from images.

---

## Approach

The current system focuses on spatial feature extraction using a pretrained convolutional neural network.

### Key Steps:
1. Input images are resized and normalized
2. Data augmentation is applied to improve generalization
3. EfficientNetB0 is used as a feature extractor
4. A custom classification head is added for binary classification (Real vs Fake)
5. The model is trained using binary cross-entropy loss

---

## Model Architecture

- Base Model: EfficientNetB0 (pretrained on ImageNet)
- Trainable Layers: Classification head only (base model frozen)
- Layers:
  - Global Average Pooling
  - Batch Normalization
  - Dense (128 units, ReLU)
  - Dropout (0.3)
  - Output Layer (Sigmoid)

---

## Dataset

- Dataset used: Celeb-DF (preprocessed)
- Classes:
  - Real
  - Fake

Dataset structure:
```
dataset/
 ├── train/
 │    ├── real/
 │    └── fake/
 ├── val/
 │    ├── real/
 │    └── fake/
```

---

## Training Pipeline

- Image size: 224 × 224
- Batch size: 32
- Optimizer: Adam (learning rate = 0.0001)
- Loss function: Binary Crossentropy
- Metrics: Accuracy

### Data Augmentation:
- Random horizontal flip
- Random rotation
- Random zoom

### Callback:
- Early Stopping (based on validation loss)

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the model
```
python train_pipeline.py
```

### 3. Run prediction
Update the image path inside `predict.py`, then run:
```
python predict.py
```

---

## Project Structure

```
VisualGuard-Deepfake_detector/
 ├── predict.py
 ├── spatial_model.py
 ├── train_pipeline.py
 ├── requirements.txt
 └── models/
      └── deepfake_spatial_model.keras
```

---

## Results

Model performance should be evaluated using:
- Accuracy
- Validation Loss
- Confusion Matrix (recommended future addition)

(Note: Add your actual results here after training)

---

## Limitations

- Currently supports only image-based detection
- Temporal inconsistencies in videos are not analyzed
- Base model is frozen, limiting fine-tuning potential
- Performance depends on dataset quality and diversity

---

## Future Work

- Incorporate temporal analysis for video-based detection
- Add attention-based fusion of spatial and temporal features
- Implement explainability techniques such as Grad-CAM
- Fine-tune EfficientNet for improved accuracy
- Deploy as a web-based application

---

## Author

Ashtami RS
Artificial Intelligence and Data Science Student

---

## License

This project is open-source and available for educational and research purposes.
