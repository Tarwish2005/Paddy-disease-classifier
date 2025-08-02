# Paddy-disease-classifier
Of course\! Here is a comprehensive README file for your GitHub repository based on the provided Jupyter Notebook, complete with a detailed workflow chart.

-----

# Paddy Disease Classification using PyTorch and ResNet-18

This repository contains a deep learning project for classifying diseases in paddy (rice) plants. The model is built using PyTorch and leverages transfer learning with a pre-trained ResNet-18 architecture to achieve high accuracy in identifying various paddy leaf diseases from images.

## 📋 Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Workflow Chart](https://www.google.com/search?q=%23workflow-chart)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Model Architecture](https://www.google.com/search?q=%23model-architecture)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Results](https://www.google.com/search?q=%23results)
  - [Code Overview](https://www.google.com/search?q=%23code-overview)

## 📖 Project Overview

The goal of this project is to accurately classify paddy leaf diseases from a given image dataset. This is a common challenge in agriculture where automated systems can help in early detection and prevention of crop diseases. This notebook implements a complete pipeline for an image classification task:

1.  **Data Acquisition** from Kaggle.
2.  **Image Preprocessing and Augmentation** to create a robust training set.
3.  **Model Training** using a powerful pre-trained convolutional neural network (CNN).
4.  **Model Evaluation** to assess its performance on unseen data.

The final model achieves a high validation accuracy, demonstrating the effectiveness of transfer learning for this computer vision task.

## 📊 Workflow Chart

The project follows a systematic workflow from data collection to model evaluation, as illustrated below.

```
+--------------------------+
|   1. Data Acquisition    |
| (Kaggle API)             |
+--------------------------+
            |
            ▼
+--------------------------+
| 2. Preprocessing & Augment|
|  - Resize to 128x128     |
|  - Random Rotations      |
|  - Flips, Affine Transforms|
|  - Normalization         |
+--------------------------+
            |
            ▼
+--------------------------+
|    3. Data Loading       |
|  - Split (80% Train /    |
|    20% Validation)       |
|  - Create DataLoaders    |
+--------------------------+
            |
            ▼
+--------------------------+
|   4. Model Preparation   |
|  - Load Pre-trained      |
|    ResNet-18             |
|  - Replace Final Layer   |
|  - Add Dropout           |
+--------------------------+
            |
            ▼
+--------------------------+
|      5. Training         |
|  - GPU/CPU Agnostic      |
|  - Adam Optimizer        |
|  - CrossEntropyLoss      |
|  - 50 Epochs             |
+--------------------------+
            |
            ▼
+--------------------------+
|     6. Evaluation        |
|  - Monitor Validation    |
|    Loss & Accuracy       |
+--------------------------+
            |
            ▼
+--------------------------+
|       7. Results         |
|  - Log metrics per epoch |
|  - Final Accuracy: 96.73%|
+--------------------------+
```

## 🌾 Dataset

The dataset used for this project is from the **Paddy Disease Classification** competition on Kaggle. You can find it [here](https://www.kaggle.com/competitions/paddy-disease-classification).

  - **Source**: Kaggle
  - **Classes**: 10 (9 disease categories and 1 healthy category)
  - **Image Size**: Resized to 128x128 pixels for training.
  - **Data Split**: The training data is split into an 80% training set and a 20% validation set.

A visualization of the class distribution in the training dataset is shown below:

## 🧠 Model Architecture

This project employs **transfer learning** to leverage the feature extraction capabilities of a model pre-trained on a large dataset (ImageNet).

  - **Base Model**: **ResNet-18** (`torchvision.models.resnet18`).
  - **Classifier Head**: The original fully connected layer of the ResNet-18 model is replaced with a custom `nn.Sequential` block to adapt it for our 10-class classification problem. This new head consists of:
    1.  A **Dropout** layer with a rate of 0.1 to regularize the model and prevent overfitting.
    2.  A **Linear** layer that maps the features extracted by the ResNet backbone to the 10 output classes.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

  - Python 3.x
  - Jupyter Notebook or JupyterLab
  - A Kaggle account and API key (`kaggle.json`)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install the required packages.** It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:

    ```
    opendatasets
    torch
    torchvision
    matplotlib
    numpy
    tqdm
    opencv-python
    pickle-mixin
    ```

3.  **Set up your Kaggle API key:**

      - Download your `kaggle.json` file from your Kaggle account settings.
      - When you run the notebook for the first time, `opendatasets` will prompt you for your Kaggle username and API key.

## 💻 Usage

1.  Launch Jupyter Notebook or JupyterLab.
2.  Open the `Untitled9.ipynb` notebook.
3.  Execute the cells sequentially from top to bottom.

The notebook will automatically download the dataset, preprocess the images, define the model, and start the training process. The progress for each epoch will be displayed in the output.

## 📈 Results

The model was trained for **50 epochs**. The training process monitored both the training loss and the validation loss/accuracy at the end of each epoch.

  - **Final Validation Accuracy**: **96.73%**
  - **Final Validation Loss**: **0.1895**

This high accuracy demonstrates the model's effectiveness in correctly identifying different paddy diseases.

A sample batch of the augmented training images:

## 🔍 Code Overview

  - **Data Loading & Preprocessing**: Cells 1-8 handle library imports, hyperparameter definitions, data augmentation transforms, and the creation of `ImageFolder` datasets and `DataLoader`s.
  - **Model Definition**: Cell 11 defines the ResNet-18 model and modifies the final layer for transfer learning.
  - **Training Setup**: Cells 12-14 set up the loss function, optimizer, and helper functions/classes for device management (CPU/GPU).
  - **Training Loop**: Cell 15 contains the main training loop that iterates for 50 epochs, trains the model on the training data, evaluates it on the validation data, and logs the results.
