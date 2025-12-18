# I_AI: Multi-Class Ocular Disease Classification

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 👁️ Project Overview
**I_AI** is a comprehensive research and development project focused on classifying **8 distinct ocular diseases** using retinal fundus images. The project systematically evaluates the performance gap between **Classical Machine Learning algorithms** and state-of-the-art **Deep Convolutional Neural Networks (CNNs)**.

The primary goal is to tackle common challenges in medical imaging:
- **Class Imbalance**
- **High Dimensionality**
- **Subtle Pathological Features** (e.g., micro-aneurysms in Diabetic Retinopathy)

---

## 🎯 Objectives
- Compare **Classical Machine Learning** models (KNN, Random Forest, Logistic Regression) across different image resolutions (**32^2**, **64^2**, **128^2**).
- Optimize **EfficientNet (B0 & B3)** architectures to reduce overfitting using **L2 Regularization** and **Label Smoothing**.
- Improve **ResNet50** sensitivity to minority classes using **CLAHE** preprocessing and **Categorical Focal Loss**.

---

## 📂 Dataset & Preprocessing
The dataset consists of retinal fundus images belonging to the following 8 classes:

`AMD`, `Cataract`, `Diabetes`, `Glaucoma`, `Hypertension`, `Myopia`, `Normal`, `Other`

### Key Preprocessing Steps
1. **Resolution Standardization**  
   Images are resized to **224 × 224** for CNN-based models to balance computational cost and feature richness.

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
   Applied to the **L-channel in LAB color space** to enhance vessels and hemorrhages.  
   This step is critical for improving ResNet50 performance on subtle pathological patterns.

3. **Caching Mechanism**  
   A serialized caching pipeline is implemented for classical ML experiments, reducing data loading time by approximately **90%** during iterative training.

---

## 🏗️ Methodologies & Architectures

### 1. Classical Machine Learning
Classical models were trained on **flattened image vectors**, highlighting the curse of dimensionality.

- **Models:** KNN, Decision Tree, Logistic Regression, Random Forest
- **Key Findings:**
  - Logistic Regression fails at **128x128** due to feature explosion (~49k features).
  - **Random Forest** is the most robust classical model, achieving the best trade-off between noise tolerance and accuracy.

---

### 2. EfficientNet Family (B0 & B3)

- **Initial Observation:**  
  EfficientNet-B0 achieved **79% Test Accuracy** but only **64% Validation Accuracy**, indicating severe overfitting.

- **Ultimate Pipeline (Regularized Setup):**
  - **L2 Regularization:** Prevents weight explosion and improves generalization.
  - **Label Smoothing:** Replaces hard targets (1.0) with soft labels (e.g., 0.9) to reduce overconfidence.

- **Outcome:**
  - Validation and training accuracy gap reduced to **<2%**
  - Final balanced accuracy of approximately **60%**
  - EfficientNet-B3 offers higher capacity but incurs **~40% higher RAM and CPU usage**

---

### 3. ResNet50 (The Specialist Model)

- **Main Focus:** Class imbalance (dominance of the *Normal* class)

- **Techniques Used:**
  - **Categorical Focal Loss:**  
    Down-weights easy examples and emphasizes hard minority classes such as *Diabetes* and *Hypertension*.
  - **Transfer Learning:**  
    Initialized with ImageNet weights using a **Warmup → Fine-Tuning** training strategy.

- **Result:**  
  Lower overall accuracy but **significantly higher recall** on minority disease classes.

---

## 📊 Performance Results

| Model Strategy | Resolution | Test Accuracy | Key Observation |
|---------------|-----------:|--------------:|----------------|
| KNN (k=7) | 32 × 32 | ~30% | Unable to capture spatial hierarchy |
| Random Forest | 64 × 64 | **46%** | Best classical ML performance |
| EfficientNet-B0 (Push) | 224 × 224 | 79% | Severe overfitting (Val: 64%) |
| EfficientNet-B0 (Ultimate) | 224 × 224 | **60%** | Balanced, strong generalization |
| ResNet50 (Focal Loss) | 224 × 224 | 51% | High recall on minority classes |

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for CNN training)
- Required libraries:
  - `tensorflow`
  - `scikit-learn`
  - `pandas`
  - `opencv-python`
  - `matplotlib`

---

### 1. Clone the Repository
```bash
git clone https://github.com/u-bharaki/I_AI.git
cd I_AI
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Training

You can run different training pipelines depending on the selected **model architecture** and **training strategy**.  
Make sure you are in the **root directory** of the project before running any command.

---

#### ResNet50 (Focal Loss & CLAHE)

Train the ResNet50 model with CLAHE preprocessing and Categorical Focal Loss:

```bash
python resnet50/train.py
````

**Resume Training:**
The same script automatically detects the best checkpoint and resumes training if it exists.

---

#### EfficientNet-B0 (The “Ultimate” Pipeline)

Run the optimized EfficientNet-B0 training with **L2 Regularization** and **Label Smoothing**:

```bash
python efficientb0/train_ultimate_b0.py
```

Other available training strategies:

* `train_push.py`
* `train_stabilize.py`

---

#### EfficientNet-B3 (High-Capacity Model)

Train the higher-capacity EfficientNet-B3 model:

```bash
python efficientb3/train.py
```

---

#### Classical Machine Learning Models

Classical ML algorithms can be trained individually.
For example, to train a **Random Forest** model:

```bash
python classicalModels/model_forest.py
```

Other available scripts:

* `model_knn.py`
* `model_logistic.py`
* `model_tree.py`

---

#### MobileNetV3-Large (Fast Training)

Train the lightweight MobileNetV3-Large model with fast convergence and logging enabled:

```bash
python MobileNetV3-Large/train_fast_with_logs.py
```

### Resume Training from a Checkpoint

The system automatically checks for the file:

```bash
best_resnet50_model.keras
```

* If it exists, training resumes from the last best epoch.
* If it does not exist, training starts from scratch.

```bash
python train_continue.py
```

---

## 🧪 Evaluation

To generate:

* Confusion Matrices
* Classification Reports
* Visual evaluation metrics

Run:

```bash
python evaluate_results.py
```

---

## 📈 Visualizations

* Efficiency vs. Overfitting
* Resource Utilization Analysis
* Impact of Regularization on EfficientNet-B0 (Push vs Ultimate)
* EfficientNet-B3 RAM and CPU Usage Analysis

---

## 🤝 Contributors

* **Ali Emre YENİHAYAT** — EfficientNet B3 & Data Analysis
* **Berk ÜLKER** — ResNet50, CLAHE Implementation, Pipeline Design
* **Duygu AKMAN** — EfficientNet B0 Optimization & Classical ML Benchmarks

---

## 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

Developed at **TOBB University of Economics and Technology**
Department of Computer Engineering


