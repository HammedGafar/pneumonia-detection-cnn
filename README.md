# Pneumonia Detection from Chest X-rays using Deep Learning

## 📌 Overview
This project applies **deep convolutional network** to classify chest X-ray images into **pneumonia** or **normal** categories. The network architecture choice was to take advantage of residual connections for good information propagation, avoiding the gradient vanishing problem.

Two state-of-the-art convolutional neural networks were fine-tuned:
- **ResNet50**
- **DenseNet161**

The work demonstrates how transfer learning, progressive training, and visualization techniques can improve medical image classification.

To know what region in the image is significant in the model's decision making, the project integrates **Class Activation Maps (CAMs)**, which highlight the regions of the X-ray that most influenced the model’s prediction. This step is critical in medical AI applications, making deep learning models transparent(i.e Not viewed as a blackbox function )and clinical relevance.


---

##  Methods
### Data Preparation
- Preprocessed over **6,000 chest X-ray images** downloaded from kaggle
- Applied **data augmentation** (flips, rotations, normalization) to improve generalization
- Split into training, validation, and test sets

### Model Training
- **Transfer Learning**: Used pretrained ImageNet weights for ResNet50 & DenseNet161
- **Progressive Unfreezing**: Gradually unfroze deeper layers for fine-tuning
- **Regularization**: Early stopping, dropout, and learning rate scheduling
- **Optimization**: Adam optimizer with tuned hyperparameters
- **Hardware**: GPU acceleration for efficient training

### Visualization
- Implemented **Class Activation Maps (CAMs)** using PyTorch forward hooks
- Highlighted **clinically relevant regions** influencing model predictions

---

## 📊 Results
- **DenseNet161** achieved **93.75% validation accuracy**
- **ResNet50** produced strong classification performance with interpretable CAMs
- Demonstrated robust ability to detect pneumonia patterns in X-rays

---



