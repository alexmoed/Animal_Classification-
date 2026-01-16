# Animal Image Classification with CNN

A convolutional neural network for classifying animal photographs into 7 categories: Squirrel, Lion, Horse, Elephant, Chicken, Camel, and Bear. Achieved **91.43% test accuracy** through iterative architecture refinement and augmentation tuning.

## Overview

This project implements a CNN-based image classifier trained from scratch (no pre-trained weights). The development process involved 7 iterations, progressively improving from 58% to 91.43% accuracy through systematic experimentation with architecture, augmentation strategies, and optimizer selection.

## Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 99.46% |
| Test Accuracy | 89.06% - 91.43% |
| Training Loss | 0.0165 |

## Dataset

- **Training set:** 2,392 images across 7 categories
- **Test set:** 70 images across 7 categories
- **Categories:** Squirrel, Lion, Horse, Elephant, Chicken, Camel, Bear

## Model Architecture

```
Conv2d (3 → 32, kernel=4, stride=2, padding=1)
BatchNorm2d → ReLU → MaxPool2d
    ↓
Conv2d (32 → 64, kernel=3, padding=1)
BatchNorm2d → ReLU → MaxPool2d
    ↓
Conv2d (64 → 128, kernel=3, padding=1)
BatchNorm2d → ReLU → MaxPool2d
    ↓
Flatten → Dense → ReLU → Dropout(0.5)
    ↓
Dense (7 classes)
```

## Key Configuration

- **Input Resolution:** 256×256
- **Optimizer:** AdamW (lr=0.00110, weight_decay=1e-5)
- **Loss Function:** CrossEntropyLoss
- **Learning Rate Scheduler:** StepLR (gamma=0.9785)

## Data Augmentation

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(25)
transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
transforms.RandomErasing()  # Simulates partial occlusion
```

## Installation

```bash
git clone https://github.com/yourusername/animal-classification.git
cd animal-classification
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
torchvision
numpy
matplotlib
```

## Usage

### Training

```python
from train import train_model

model, history = train_model(
    train_dir='data/train',
    test_dir='data/test',
    epochs=300,
    batch_size=32
)
```

### Inference

```python
from predict import classify_image

result = classify_image(model, 'path/to/image.jpg')
print(f"Predicted: {result}")
```

## Development Journey

### Version 1 (Baseline)
Simple fully connected network with 128×128 input. Reached 58% accuracy and showed clear overfitting.

### Version 2
Upgraded to CNN architecture with 512×512 input and light augmentation. Peaked at 77% accuracy around epoch 41 before degrading.

### Version 3
Switched from SGD to Adam optimizer and added learning rate scheduler. Broke through 80% and reached 84% accuracy.

### Version 4
Key improvements:
- Reduced resolution to 256×256 (more efficient, similar performance)
- Added BatchNorm after each conv layer
- Switched to AdamW with weight decay (1e-4)
- Increased ColorJitter strength to 0.25
- Added stride=2 in first conv layer for broader feature extraction

Result: 88.57% accuracy

### Version 5
Added RandomErasing to handle partially visible animals in training images. Increased dropout to 0.5 and rotation to 25 degrees.

### Version 6 (Experimental)
Tested RandAugment as an automated augmentation strategy. Did not improve over manual augmentations. Abandoned this approach.

### Version 7 (Final)
Returned to Version 5 settings and extended training to 300 epochs. Reached 91.43% test accuracy at epoch 290. The model hit 90% around epoch 120 which may be a more practical stopping point.

## Design Decisions

**Why CNN over Vision Transformers?**

ViT requires large-scale datasets (often millions of images) to train effectively from scratch. With only 2,392 training images and a requirement to avoid pre-trained models, CNN was the clear choice. CNNs also perform well with limited data and have been thoroughly validated across many applications.

**Why AdamW?**

AdamW separates weight decay from the gradient update. Standard Adam applies weight decay inside the gradient computation where it gets distorted by adaptive learning rates. AdamW applies it directly to the weights, providing more effective regularization.

**Why RandomErasing?**

Many training images showed only parts of animals (natural cropping). RandomErasing simulates partial occlusion and forces the model to recognize animals from incomplete visual information.

## Sample Predictions

![Animal Classification Predictions](https://storage.googleapis.com/anmstorage/animal%20prediction.png)

## Limitations

Some misclassifications occurred with unusual images:
- A flying squirrel was classified as a lion (visually distinct from other squirrels in the dataset)
- Occasional confusion between unrelated categories (bear → chicken, elephant → squirrel)

These edge cases suggest the model relies heavily on typical animal appearances and struggles with unusual poses or subspecies not well represented in training data.

## References

- Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Trigka, M. and Dritsas, E. (2025). A Comprehensive Survey of Deep Learning Approaches in Image Processing. *Sensors*, 25(2).
- van der Werff, T. (2024). CNN vs. Vision Transformer: A Practitioner's Guide to Selecting the Right Model.

## Full Report

[View the complete project report (PDF)](https://storage.googleapis.com/anmstorage/Image_classification_Amoed.pdf)
