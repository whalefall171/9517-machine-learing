# 9517-project

# Aerial Landscape Emotion Classification (LBP + KNN and SVM)

This project implements machine learning classification using **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)** for classifying multi-class data. It includes model training, evaluation, and optimization, and provides a mechanism for saving and loading trained models.

## Requirements

Before running the code, make sure you have the following software and libraries installed.
```
LBP/
├── data/
│   └── Aerial_Landscapes/         # Dataset folder (each subfolder is a class, not in package)
├── models/
│   └── knn_model.pkl              # Saved model after training
├── src/
│   ├── __init__.py
│   ├── config.py                  # Parameter settings
│   ├── features.py                # Implements LBP feature extraction
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── train.py                   # Training, evaluation
│   ├── utils.py                   # Visual functions and other auxiliary tools
│   └── inference.py               # Load the model and reason
├── main.py                        # Entry point for the project
└── requirements.txt               # Project dependencies
```

### Environment Setup

1. **Python 3.6+**
2. OpenCV
3. scikit-learn: For machine learning algorithms and model evaluation
4. numpy: For numerical operations
5. pandas: For handling data structures and datasets
6. joblib: For saving and loading models

Install dependencies with:
```
pip install -r requirements.txt
```
   
## Models Used

1. K-Nearest Neighbors (KNN)
Model Description: KNN is a non-parametric method used for classification. It classifies data points based on the majority vote from the k-nearest neighbors in the feature space.

        Performance Metrics:
        
        Accuracy: 35.42%
        
        Precision: 38.51%
        
        Recall: 32.54%
        
        F1-Score: 34.87%

    The classification report shows the precision, recall, and F1-score for each class (0–14), with the average performance across all classes being relatively low, indicating room for improvement.

2. Support Vector Machine (SVM)
Model Description: SVM is a supervised machine learning algorithm used for classification tasks. It constructs a hyperplane in a high-dimensional space to separate data points from different classes.

        Performance Metrics:
        
        Accuracy: 47.46%
        
        Precision: 47.04%
        
        Recall: 47.46%
        
        F1-Score: 47.07%

## How to Run

1. Place your dataset in the `data/Aerial_Landscapes/` folder. The structure should be:
```
├── data/
│   └── Aerial_Landscapes/  
│        ├── class1/
│        │   ├── image1.jpg
│        │   ├── image2.jpg
│        ├── class2/
│        │   ├── image1.jpg
│        │   ├── image2.jpg
│        ...
```
2. Run the main script with the desired classifier (`knn` or `svm`):
```
python main.py --classifier knn --save_model
```
Optional arguments:

`--classifier`: Choose between `knn` or `svm`

`--save_model`: Whether to save the trained model to a `.pkl` file


# Aerial Landscape Scene Classification (SIFT + KNN and SVM)

This project implements machine learning classification using **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)** for multi-class scene classification. It is based on classical image processing and BoW (Bag of Visual Words) representations using SIFT features. The pipeline includes feature extraction, vocabulary construction, training, and evaluation.

## Requirements

Before running the code, make sure you have the following software and libraries installed.

```
SIFT/
├── Aerial_Landscapes/             # Dataset folder (15 subfolders for each class)
├── src/                           # Source code modules
│   ├── config.py
│   ├── data_loader.py
│   ├── sift_bow_extractor.py
│   ├── classifier.py
│   ├── utils.py
├── main.ipynb                     # Main notebook to run training and evaluation
├── requirements.txt               # Python dependencies             
```

### Environment Setup

1. Python 3.6+
2. OpenCV
3. scikit-learn: For machine learning algorithms and model evaluation
4. numpy: For numerical operations
5. tqdm: For progress bar and status
6. joblib (optional): For saving and loading models

Install dependencies with:

```
pip install -r requirements.txt
```

## Models Used

1. K-Nearest Neighbors (KNN)  
Model Description: KNN is a non-parametric method used for classification. It classifies data points based on the majority vote from the k-nearest neighbors in the BoW feature space.

        Performance Metrics:
        
        Accuracy: ~47.96%
        
        Precision: ~48.79%
        
        Recall: ~47.96%
        
        F1-Score: ~44.83%

   *Detailed confusion matrix is available in the notebook output or final report.*

    The classification report includes metrics per class. Accuracy is moderate, depending on the training ratio and BoW quality.

3. Support Vector Machine (SVM)  
Model Description: SVM is a supervised machine learning algorithm used for classification tasks. It constructs a linear hyperplane to separate BoW feature vectors in a high-dimensional space.

        Performance Metrics:
        
        Accuracy: ~59.58%
        
        Precision: ~61.4%
        
        Recall: ~59.58%
        
        F1-Score: ~58.63%

    *Detailed confusion matrix is available in the notebook output or final report.*

## How to Run

1. Place your dataset in the `Aerial_Landscapes/` folder. The structure should be:
```
├── Aerial_Landscapes/  
│    ├── class1/
│    │   ├── image1.jpg
│    │   ├── image2.jpg
│    ├── class2/
│    │   ├── image1.jpg
│    │   ├── image2.jpg
│    ...
```

2. Open the `main.ipynb` notebook and run each cell in sequence to:
   - Load the dataset
   - Extract SIFT features
   - Build the BoW vocabulary
   - Train the classifiers
   - Evaluate the results



# Deep learning methods（ResNet, EfficientNet）

This module applies deep learning methods to multi-class classification of aerial images using Convolutional Neural Networks (CNNs). The implementation is based on the PyTorch framework and compares different architectures including ResNet18, EfficientNet-B0, and SENet-enhanced ResNet.

## Requirements

Before running the code, make sure you have the following software and libraries installed.
```
DeepCV/
├── data/                               # Dataset folder with 15 subfolders (one per class)
├── models/
│   ├── resnet.py                       # ResNet18 model builder
│   ├── efficientnet.py                 # EfficientNet-B0 model builder
│   └── senet.py                        # SE-ResNet18 model builder
├── train.py                            # Training and evaluation loop
├── dataset.py                          # DataLoader and augmentation
├── requirements.txt                    # Dependencies
└── README.md                           # This documentation
```
### Environment Setup
1. **Python 3.6+**
2. PyTorch
3. torchvision
4. matplotlib: For plotting training curves
5. tqdm: For training progress bars
6. scikit-learn: For evaluation metrics
7. OpenCV (optional for Grad-CAM)



## Models Used
Three architectures are compared for image classification performance:
1. ResNet-18
A classic residual network with skip connections.

        Performance Metrics:
        
        Accuracy: 71.32% 
        
        Precision: 71.45%  
        
        Recall: 71.32%
        
        F1-Score: 71.11%
3. EfficientNet-B0
A highly efficient architecture balancing network depth, width, and resolution using compound scaling.

        Performance Metrics:
        
        Accuracy: 74.65% 
        
        Precision: 75.29% 
        
        Recall: 74.65% 
        
        F1-Score: 74.33%
5. SENet-ResNet18
An enhanced version of ResNet-18 using Squeeze-and-Excitation (SE) blocks to adaptively recalibrate feature responses.

        Performance Metrics:
        
        Accuracy: 73.59%
        
        Precision: 73.82%
        
        Recall: 73.59%
        
        F1-Score: 73.40%
## How to Run
1. Before running the code, make sure you have the following software and libraries installed.
```
├── Aerial_Landscapes/
│   ├── Agriculture/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── Airport/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── ...
```
2. Open the training notebook or script and run each step in order to:
   - Load the dataset
   - Load the dataset using
   - Select and initialize a model
   - Train the model
   - Save and evaluate the model performance
---

## ℹ️ Notes

- This `README.md` file is **not included** in the original project folder and is intended as post-development documentation.
- The `models/` folder is **not required to be manually created**. It will be **automatically generated** during runtime when you run the `main.ipynb` notebook. Trained models will be saved there.
