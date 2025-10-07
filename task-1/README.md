# Test task 1

### About
###### This project demonstrates and compares the performance of three different machine learning models — Random Forest (RF), Feed-Forward Neural Network (NN), and Convolutional Neural Network (CNN) — on the MNIST handwritten digit classification task.  

The implementation follows a **factory-style pattern** (`MnistClassifier` class) to manage initialization, data preprocessing, model training, and evaluation for each classifier consistently.  
A reduced dataset of **5,000 training** and **5,000 test** samples is used to ensure fast runtime for demonstration purposes.

---

### Features
**Unified Training Interface**:  
Implements a base class (`MnistClassifierInterface`) that enforces `train` and `predict` methods across all model implementations.

**Factory Pattern Architecture**:  
The `MnistClassifier` class selects and initializes the correct model type (`'rf'`, `'nn'`, or `'cnn'`) and applies the proper preprocessing:
- **Flattening** for RF and NN  
- **Normalization** (divide by 255.0) for NN and CNN  
- **Reshaping** to `(N, 28, 28, 1)` for CNN

**Consistent Evaluation**:  
Each model is trained and evaluated on the same MNIST subset, ensuring a fair comparison of results.

**Lightweight Demonstration Mode**:  
Runs quickly on standard hardware with reduced sample sizes, suitable for educational and prototype testing.

---

### Requirements
Before running, ensure you have the following installed:
- Python 3.8+
- scikit-learn
- TensorFlow or PyTorch
- numpy
- matplotlib

Install dependencies:
`pip install -r requirements.txt`

---

### Classification Overview

| Algorithm | Model Class     | Input Data Requirement                | Training Details                                  | Accuracy (1 epoch) |
|------------|-----------------|----------------------------------------|---------------------------------------------------|--------------------|
| **RF**     | `RFClassifier`  | Flattened `(N, 784)`                   | 10 estimators, no normalization needed            | **0.8498**         |
| **NN**     | `NNClassifier`  | Flattened `(N, 784)`, Normalized (0–1) | 2 Dense layers (128, 64), 1 epoch                | **0.8760**         |
| **CNN**    | `CNNClassifier` | Reshaped `(N, 28, 28, 1)`, Normalized (0–1) | 2 Conv2D layers, 1 epoch                        | **0.9124**         |

---

### Setup

###### 1. Load MNIST Dataset
The dataset is automatically downloaded via TensorFlow/Keras when running the script.

###### 2. Run all model comparisons
`python mnist_comparison.py`

###### 3. Run an individual model
Run a specific classifier:
- `python mnist_comparison.py --model rf`
- `python mnist_comparison.py --model nn`
- `python mnist_comparison.py --model cnn`

---

### Expected Output
After execution, the script will train each model and display accuracy results.

Example:
~~~
[INFO] Training Random Forest...
Accuracy: 0.8498

[INFO] Training Neural Network...
Accuracy: 0.8760

[INFO] Training Convolutional Neural Network...
Accuracy: 0.9124
~~~

### Results & Analysis
The **Convolutional Neural Network (CNN)** achieved the highest test accuracy (**0.9124**) on the MNIST dataset, outperforming both the **Feed-Forward Neural Network (NN)** and the **Random Forest (RF)** classifier — as expected for an image-based classification problem.

---

### Project Structure
~~~
├──main.ipynb
├──README.md
└──requirements.txt
~~~