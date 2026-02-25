# Garbage Classification using Custom CNN

## Overview

This project implements a custom Convolutional Neural Network (CNN) for garbage image classification.

The model is trained using stratified 10-fold cross-validation and compared against a VGG16 baseline model.

---

## Objectives

- Design and train a CNN from scratch
- Apply data augmentation
- Perform 10-fold cross-validation
- Evaluate using multiple metrics
- Compare with transfer learning baseline (VGG16)

---

## Model Architecture

Custom CNN structure:

- Conv → ReLU → MaxPool
- Conv → ReLU → MaxPool
- Fully Connected Layers
- Softmax Output

(Modify according to your actual structure)

---

## Training Strategy

- Stratified 10-fold cross-validation
- Data augmentation
- Optimizer: (e.g., Adam)
- Loss: CrossEntropyLoss
- Early stopping (if applicable)

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Baseline Comparison

The performance of the custom CNN is compared with a pre-trained VGG16 model.

Comparison includes:
- Validation accuracy
- F1-score
- Convergence behavior

---

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

---

## Example Results

(Add training curve and confusion matrix image here)

Example:
- Average Accuracy (10-fold): XX%
- Average F1-score: XX%

---

## Notes

This project focuses on evaluating model robustness using cross-validation and systematic comparison with a transfer learning baseline.
