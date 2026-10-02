# Offline Handwritten Signature Verification using Metric Learning
### Artificial Intelligence & Data Science II (AIDS II) Lab Project

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

Offline signature verification is a critical biometric task across banking, forensics, and legal administration. Unlike online verification (which captures pen speed, pressure, and stroke trajectory), **offline verification** operates strictly on static raster scans. 

### Core Challenges
1. **High Intra-Writer Variance**: Genuine signatures from the same individual exhibit natural morphological variations.
2. **Subtle Inter-Writer Discrepancies**: Skilled forgers intentionally imitate stroke trajectories, loops, and aspect ratios.
3. **Open-Set Deployment**: Real-world biometric systems must verify signatures from **unseen signers** never encountered during model training.

### The Paradigm Shift: From Classification to Metric Learning
Conventional CNNs pose verification as single-image binary classification (*Genuine vs. Forged*). In open-set settings, this approach fails because the network memorizes specific signatures rather than learning structural invariance. 

To overcome this, we designed a **Contrastive Siamese Neural Network (SNN)**:
* **Twin Shared-Weight Branches**: Two identical convolutional backbones map signature pairs $(X_1, X_2)$ onto a shared **128-dimensional unit hypersphere** ($L_2$-normalized).
* **Contrastive Loss Optimization**:
  $$\mathcal{L}(y, d) = \frac{1}{2} (1 - y) d^2 + \frac{1}{2} y \max(0, m - d)^2$$
  - **Genuine Pairs ($y = 0$)**: Pushes Euclidean distance $d \to 0$.
  - **Skilled Forgery Pairs ($y = 1$)**: Enforces a margin separation $d \ge m = 1.0$.

### 99% Architectural Footprint Reduction
In legacy implementations, spatial feature maps were flattened directly into a 1,024-unit dense layer, resulting in over **111.4 million parameters** and a bloated **455.88 MB** weights file (`model_weights33.h5`). 

We replaced the dense parameter bottleneck with **Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`)**, reducing the model checkpoint from **455.88 MB down to 4.85 MB (99% reduction)**, preventing spatial overfitting, and enabling sub-40ms CPU inference.

---

## 🔬 Empirical Benchmarks on Unseen Writers

The model was trained strictly on **Writers 1–35** (Validation: 36–40) and evaluated on **1,200 signature pairs from unseen Writers 41–55** (never seen during training):

| Metric | Result | Benchmark Significance |
| :--- | :--- | :--- |
| **Area Under ROC Curve (AUC)** | **0.9349** (93.49%) | Strong discrimination between authentic signatures and skilled forgeries |
| **Verification Accuracy** | **89.83%** | Evaluated at the optimal decision threshold $\tau^* = 0.5332$ |
| **Recall (Forgery Detection)** | **98.50%** | Intercepts $98.5\%$ of skilled forgery attempts |
| **Precision (Forgery)** | **83.95%** | High confidence when flagging a signature as counterfeit |
| **False Acceptance Rate (FAR)**| **1.50%** | Only $1.5\%$ of forged signatures mistakenly accepted |
| **False Rejection Rate (FRR)**| **18.83%** | Rejections attributable to natural intra-writer variation |
| **Equal Error Rate (EER)** | **14.33%** | Balanced operational threshold at $\tau = 0.6495$ |

### Visual Artifacts
| Distance Distribution (Unseen Signers) | Receiver Operating Characteristic (ROC) |
| :---: | :---: |
| ![Distance Distribution](evaluation_output/distance_distribution.png) | ![ROC Curve](evaluation_output/roc_curve.png) |

---

## 📂 Repository Structure

```
├── best_siamese_model.pth    # Optimized PyTorch checkpoint (4.85 MB)
├── dataset.py                # Writer-independent CEDAR parser & balanced pair generator
├── model.py                  # Siamese CNN architecture with Global Average Pooling
├── train.py                  # Contrastive loss training pipeline
├── evaluate.py               # Biometric evaluation module (FAR, FRR, EER, ROC)
├── verify.py                 # CLI inference verification tool
├── training_history.json     # Loss and accuracy logs across epochs
├── evaluation_output/        # Generated ROC and distance distribution plots
├── frontend/                 # Interactive Streamlit dashboard
│   ├── app.py                # Multi-tab demonstration dashboard
│   └── requirements.txt      # UI dependencies
├── AIDS II Project Analysis.md # Detailed laboratory investigation report
└── README.md                 # Project documentation
```

---

## 💾 Dataset Setup Instructions

This project benchmarks on the **CEDAR Signature Dataset** (Center of Excellence for Document Analysis and Recognition, University at Buffalo):
* **55 Writers**, 24 genuine signatures and 24 skilled forgeries per writer (2,640 images total).

### How to Download & Set Up the Dataset:
1. Download the CEDAR dataset from [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) or the [University at Buffalo CEDAR Repository](https://cedar.buffalo.edu/).
2. Extract the dataset into the `data/` folder following this structure:
   ```
   data/
   └── signatures/
       ├── full_org/
       │   ├── original_1_1.png
       │   └── ... (original_55_24.png)
       └── full_forg/
           ├── forgeries_1_1.png
           └── ... (forgeries_55_24.png)
   ```
*(Note: Datasets are excluded from git via `.gitignore` to comply with repository size guidelines).*

---

## 🚀 Reproduction & Usage Guide

### 1. Installation & Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Anish-Shinde-1/AIDS-II-Siamese-CNN-for-Signature-Forgery-Detection.git
cd AIDS-II-Siamese-CNN-for-Signature-Forgery-Detection
pip install torch torchvision matplotlib scikit-learn pillow streamlit
```

### 2. Live Verification via CLI (`verify.py`)
Run inference directly between any reference signature and query signature:

```bash
# Example 1: Genuine Pair Test
python verify.py --img1 path/to/original_A.png --img2 path/to/original_B.png

# Example 2: Forgery Detection Test
python verify.py --img1 path/to/original_A.png --img2 path/to/forged_A.png
```

**CLI Output Format:**
```
==================================================
      OFFLINE SIGNATURE VERIFICATION RESULT       
==================================================
 Reference Image:  path/to/original_A.png
 Query Image:      path/to/forged_A.png
 Euclidean Dist:   1.3710
 Active Threshold: Auto (optimal EER/Acc)
 Similarity Score: 31.45%
--------------------------------------------------
 VERDICT:          [-] FORGERY DETECTED
 Detail:           Discrepancy exceeds threshold; signature flagged as a forgery.
==================================================
```

### 3. Launching the Interactive Web Dashboard (`frontend/app.py`)
To launch the full Streamlit demonstration dashboard:
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`. The dashboard includes:
* **Tab 1: Project Overview & Architecture**: Mathematical formulation, architectural comparisons, and layer blueprints.
* **Tab 2: Data & Evaluation Benchmarks**: Metric scorecards and high-resolution evaluation plots.
* **Tab 3: Live Verification Demo**: Interactive drag-and-drop dual image verifier with built-in presets and real-time confidence scores.

### 4. Retraining the Siamese Network (Optional)
To retrain the model from scratch on Writers 1–35:
```bash
python train.py --epochs 10 --batch_size 32 --pairs_per_writer 30 --lr 0.001
```

### 5. Running Full Biometric Evaluation
To compute FAR, FRR, EER, and regenerate the ROC curves on unseen test writers:
```bash
python evaluate.py --checkpoint best_siamese_model.pth --pairs_per_writer 40
```

---

## 👥 Contributors & Academic Acknowledgments
* **Project**: AIDS II Miniproject (B.Tech Artificial Intelligence & Data Science)
* **Domain**: Biometric Security & Deep Metric Learning
