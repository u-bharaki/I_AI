
# I_AI: Multi-Class Ocular Disease Classification

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👁️ Project Overview
**I_AI** is a comprehensive research and development project focused on classifying **8 distinct ocular diseases** using retinal fundus images. The project systematically evaluates the performance gap between **Classical Machine Learning algorithms** and state-of-the-art **Deep Convolutional Neural Networks (CNNs)**.

The primary goal is to address core challenges in medical imaging:
- **Class Imbalance** (dominance of *Normal* samples)
- **High Dimensionality**
- **Subtle Pathological Features** (e.g., micro-aneurysms in Diabetic Retinopathy)

---

## 🎯 Objectives
- Compare **Classical Machine Learning** models (KNN, Random Forest, Logistic Regression) across multiple resolutions (**32×32**, **64×64**, **128×128**).
- Optimize **EfficientNet (B0 & B3)** using **Progressive Resizing** (up to **300×300**) and regularization techniques.
- Improve **ResNet50** sensitivity to minority classes using **CLAHE** preprocessing and **Categorical Focal Loss**.
- Evaluate **MobileNetV3-Large** for rapid prototyping and efficiency benchmarking.

---

## 📂 Dataset & Preprocessing
The dataset consists of retinal fundus images categorized into 8 classes:

`AMD`, `Cataract`, `Diabetes`, `Glaucoma`, `Hypertension`, `Myopia`, `Normal`, `Other`

### Key Preprocessing Steps
1. **Progressive Resolution Standardization**  
   Initial CNN training is performed at **224 × 224** for computational efficiency.  
   During final fine-tuning of EfficientNet models, resolution is increased to **300 × 300** to capture finer details without the heavy cost of 512 × 512.

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
   Applied to the **L-channel in LAB color space** to enhance vessels and hemorrhages.  
   This step is critical for improving ResNet50 performance on subtle pathological patterns.

3. **Caching Mechanism**  
   A serialized caching pipeline is implemented for classical ML experiments, reducing data loading time by approximately **90%** during iterative training.

---

## 🏗️ Methodologies & Architectures

### 1. Classical Machine Learning
Models were trained on **flattened image vectors**, highlighting the *Curse of Dimensionality*.

- **Models:** KNN, Decision Tree, Logistic Regression, Random Forest
- **Key Finding:**  
  **Random Forest** at **64×64** provided the best balance between robustness and accuracy, while Logistic Regression failed at **128×128** due to feature explosion.

---

### 2. EfficientNet Family (B0 & B3)

- **Challenge:**  
  EfficientNet-B0 initially suffered from severe overfitting (**79% Test vs. 64% Validation**).

- **The “Ultimate” Pipeline:**
  - **L2 Regularization**
  - **Label Smoothing**  
  These techniques penalize large weights and reduce model overconfidence.

- **Result:**  
  The generalization gap was reduced to **<2%**, achieving a stable accuracy of approximately **60%**.

- **EfficientNet-B3:**  
  Utilized for higher capacity learning. Due to I/O bottlenecks between CPU and GPU at **300×300** resolution, a custom **CUDA backend selection mechanism** was implemented to manage memory pressure.

---

### 3. ResNet50 (The Specialist)
- **Focus:** Class imbalance mitigation.
- **Technique:**  
  **Categorical Focal Loss** (γ = 2.0) was integrated to down-weight easy samples (*Normal*) and emphasize hard minority classes (*Diabetes*, *Hypertension*).
- **Result:**  
  Lower overall accuracy but significantly **higher recall** for minority disease classes.

---

### 4. MobileNetV3-Large (The Sprinter)
- **Focus:** Speed and efficiency.
- **Performance:**
  - **Training Speed:** ~40 seconds per epoch
  - **Convergence:** Stable validation within 15–20 minutes
- **Use Case:**  
  Ideal for rapid experimentation and hyperparameter tuning, though less effective for capturing very subtle features compared to EfficientNet-B3 or ResNet50.

---

## 📊 Performance Results Summary

| Model Strategy | Resolution | Test Accuracy | Key Observation |
|---------------|------------|---------------|----------------|
| **KNN (k=7)** | 32 × 32 | ~30% | Failed to capture spatial hierarchy |
| **Random Forest** | 64 × 64 | **46%** | Best classical ML performance |
| **MobileNetV3-Large** | 224 × 224 | ~55% | Extremely fast convergence |
| **ResNet50 (Focal Loss)** | 224 × 224 | 51% | High recall on minority classes |
| **EfficientNet-B0 (Push)** | 224 × 224 | 79% | Severe overfitting |
| **EfficientNet-B0 (Ultimate)** | 300 × 300 | **60%** | Best generalization |

---

## 💻 Hardware and Setup
The experimental infrastructure was designed to support varying computational requirements:

- **Primary Workstation:**  
  NVIDIA GeForce **RTX 4080 (16GB VRAM)**  
  Used for ResNet50, MobileNetV3, and EfficientNet-B0 experiments.

- **Secondary Environment:**  
  NVIDIA GeForce **GTX 1070 Ti**  
  Used for EfficientNet-B3 with custom CUDA kernel allocation to handle memory bottlenecks.

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- NVIDIA GPU (highly recommended)
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

## 🚀 Training

Run the script corresponding to the desired architecture.
Ensure you are in the **root directory**.

### ResNet50 (Focal Loss & CLAHE)

```bash
python resnet50/train.py
```

### EfficientNet-B0 (The “Ultimate” Pipeline)

```bash
python efficientb0/train_ultimate_b0.py
```

### EfficientNet-B3 (High Capacity)

```bash
python efficientb3/train.py
```

### MobileNetV3-Large (Fast Training)

```bash
python MobileNetV3-Large/train_fast_with_logs.py
```

### Classical Models (Example: Random Forest)

```bash
python classicalModels/model_forest.py
```

---

## 🔄 Resume Training

To resume training from the last automatically detected checkpoint:

```bash
python train_continue.py
```

---

## 🧪 Evaluation

To generate **Confusion Matrices** and **Classification Reports**:

```bash
python evaluate_results.py
```

---

## 📈 Visualizations

The project includes scripts for generating:

* Confusion Matrix comparisons (Push vs. Stabilize strategies)
* Accuracy and loss evolution graphs
* RAM vs. GPU utilization plots for identifying I/O bottlenecks

---

## 🤝 Contributors

* **Ali Emre YENİHAYAT** — EfficientNet-B3 Architecture & Data Analysis
* **Berk ÜLKER** — ResNet50, CLAHE Implementation, Pipeline Design
* **Duygu AKMAN** — EfficientNet-B0 Optimization & Classical ML Benchmarks

---

Developed at **TOBB University of Economics and Technology**
Department of Computer Engineering
