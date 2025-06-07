# Deep Learning for Aerial Scene Classification

This project is part of the COMP9517 group assignment, focusing on aerial image scene classification using deep convolutional neural networks (CNNs) implemented in PyTorch.

## Dataset

- **Name**: SkyView Aerial Landscape Dataset  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- **Description**: Contains 15 balanced classes of aerial landscape images (~12,000 images in total), including scenes like forests, airports, rivers, deserts, etc.

##  Model Architectures Implemented

We implemented and compared the performance of the following CNN architectures:

| Model             | Description                              |
|------------------|------------------------------------------|
| **ResNet-18**     | Classic residual network (baseline)       |
| **EfficientNet-B0** | Lightweight and accurate network         |
| **SENet-ResNet18**  | Adds SE (Squeeze-and-Excitation) blocks |

All models are trained using pretrained ImageNet weights with the final classification layer adapted to 15 classes.

##  Training Setup

- **Framework**: PyTorch
- **Epochs**: 5
- **Input Size**: 224x224
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Hardware**: Google Colab GPU (Tesla T4)

##How to Run (on Colab)

1. Clone or upload this project to your Colab
2. Make sure you have `kaggle.json` API key uploaded
3. Run the following in order:

```python
# 1. Install dependencies
!pip install -r requirements.txt

# 2. Download dataset
import kagglehub
path = kagglehub.dataset_download("ankit1743/skyview-an-aerial-landscape-dataset")

# 3. Train ResNet18
from models.resnet import build_resnet18
from dataset import get_dataloaders
from train import train_model

train_loader, val_loader = get_dataloaders(path + "/Aerial_Landscapes")
model = build_resnet18(num_classes=15)
train_model(model, train_loader, val_loader, num_epochs=5, device='cuda')
