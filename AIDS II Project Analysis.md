# AIDS II Project Analysis: Offline Handwritten Signature Verification

## Executive Overview

This repository contains an academic Artificial Intelligence & Data Science (AIDS II) project focused on **Offline Handwritten Signature Verification**. The objective of offline signature verification is to determine whether an input handwritten signature image is **genuine** (authentic) or a **forgery** (counterfeit/skilled forgery), without relying on dynamic/temporal signals like pen pressure, stroke speed, or trajectory.

The workspace is composed of three primary assets:
1. **Interactive Notebook (`signature-verification-using-cnn-snn-csnn.ipynb`)**: A multi-experiment notebook implementing and benchmarking three architectural paradigms: standard Convolutional Neural Networks (**CNN**), Siamese Neural Networks (**SNN**), and Contrastive Siamese Neural Networks (**CSNN**) across multiple distance metrics and datasets.
2. **Multi-Dataset Corpus (`/data`)**: A rich, multilingual collection of benchmark and crowdsourced signature datasets comprising over **26,000 signature images** across English, Hindi, and Bengali scripts, as well as real-world crowdsourced samples.
3. **Pretrained Weights (`model_weights33.h5`)**: A ~455.88 MB HDF5 checkpoint corresponding to a trained Siamese feature extractor network trained on genuine and forged signature pairs from the CEDAR dataset using contrastive loss.

---

## 1. Notebook Analysis (`signature-verification-using-cnn-snn-csnn.ipynb`)

### 1.1 Project Objective and Methodology
The primary challenge of offline signature verification is that genuine signatures from the same individual naturally possess intra-class variations, while skilled forgeries closely emulate the spatial trajectory and visual structure of genuine signatures.

To address this, the notebook investigates two distinct paradigms:
- **Direct Classification (Standard CNN)**: Posing the problem as single-image binary classification ($\text{Class } 0 = \text{Genuine}$, $\text{Class } 1 = \text{Forged}$).
- **Metric Learning / Similarity Learning (Siamese Neural Networks - SNN & CSNN)**: Posing the problem as pair-wise distance metric learning. Two twin subnetworks with shared weights extract low-dimensional embeddings for a pair of signatures $(X_1, X_2)$ and compute a distance metric $d = D(f(X_1), f(X_2))$. If $d < \tau$ (threshold), the signatures are verified as genuine pairs; if $d \ge \tau$, they are flagged as forgeries.

```
+----------------------------------------------------------------------------------------------------+
|                                    PROJECT ARCHITECTURE ROADMAP                                    |
+----------------------------------------------------------------------------------------------------+
                                      [ Signature Images ]
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
             [ Single Image Approach ]                     [ Pairwise Siamese Approach ]
                        │                                             │
             Standard CNN Classifier                       Twin Shared Subnetworks
                        │                                             │
             Binary Classification                         Feature Vector Embeddings (128-d)
             (Genuine vs Forged)                                      │
                                                   ┌──────────────────┴──────────────────┐
                                                   ▼                                     ▼
                                          Distance Metrics                       Contrastive Loss
                                     (Euclidean, Manhattan,                 L = y*d² + (1-y)*max(m-d,0)²
                                      Hamming, Minkowski)                                │
                                                   │                            model_weights33.h5
                                                   └──────────────────┬──────────────────┘
                                                                      ▼
                                                          Verification / Decision
                                                          (Similarity Thresholding)
```

---

### 1.2 Data Preprocessing Pipeline

The notebook implements several data ingestion and preprocessing workflows across different experimental phases:

| Phase / Cell Range | Target Size | Normalization | Color Space | Pairing / Data Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Initial CNN** (Cells 3–6) | $224 \times 224 \times 3$ | Pixel division by `255.0` | RGB (via OpenCV) | Datasets 1–3 as Train (540 images), Dataset 4 as Test (180 images) |
| **Siamese Exploration** (Cells 34–86) | $224 \times 224 \times 3$ | Random / Standard scaling | RGB | Synthetic random arrays used to test distance layer mechanics |
| **CEDAR SNN** (Cells 87–120) | $155 \times 220 \times 3$ | Pixel division by `245.0` | RGB (via Keras `load_img`) | Exhaustive intra-writer pairwise combination (genuine vs genuine, genuine vs forged) |
| **ICDAR CNN** (Cells 130–135) | $224 \times 224 \times 1$ | Pixel division by `255.0` | Grayscale (via OpenCV) | 80/20 Train-validation random split |

#### Pairwise Generation Mechanics for Siamese Training (Cell 114)
For the CEDAR experiment, the notebook constructs pairs from the first 10 writers ($i \in [0, 9]$):
- **Genuine Pairs (Positive, label 0)**: Takes 24 genuine samples per writer and forms all unique pairwise combinations $\binom{24}{2} = 276$ pairs per writer $\times 10$ writers = **2,760 genuine pairs**.
- **Forged Pairs (Negative, label 1)**: Pairs genuine signatures with forgery signatures of that same writer, yielding **2,760 forged pairs**.
- **Total Training Pairs**: $2,760 + 2,760 = \mathbf{5,520}$ pairs ($X_{\text{train1}}, X_{\text{train2}}$).

---

### 1.3 Model Architectures Explored

The notebook implements 4 major model architectures:

#### A. Standard Baseline CNN (Cells 7–28)
- **Input**: $(224, 224, 3)$
- **Layers**:
  - `Conv2D(32, (3, 3))` + `MaxPooling2D((2, 2))` + `Dropout(0.25)`
  - `Conv2D(64, (3, 3))` + `MaxPooling2D((2, 2))` + `Dropout(0.25)`
  - `Flatten()` + `Dense(128)` + `Dropout(0.5)` + `Dense(2, activation='softmax')`
- **Activation Function Comparison**: Evaluates ReLU, Sigmoid, and Tanh activation functions on hidden dense layers.

#### B. Siamese Network with Distance Variations (Cells 30–86)
Tests two twin subnetworks extracting 128-dimensional representations and merges them using different distance functions:
1. **Euclidean Distance**: $d(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$
2. **Manhattan ($L_1$) Distance**: $d(u, v) = \sum_{i=1}^n |u_i - v_i|$
3. **Hamming Distance**: Continuous approximation using absolute element difference.
4. **Minkowski Distance**: $d(u, v) = \left(\sum_{i=1}^n |u_i - v_i|^p\right)^{1/p}$ with $p = 3$.

#### C. CEDAR Deep Siamese Subnetwork (Cells 108–109) — *Source of Saved Weights*
A deep convolutional backbone tailored for signature feature extraction:
- **Input**: $(155, 220, 3)$
- **Conv Block 1**: `Conv2D(96, (11, 11), activation='relu')` + `BatchNormalization(axis=1)` + `MaxPooling2D((3, 3), strides=(2, 2), padding='same')`
- **Conv Block 2**: `Conv2D(256, (5, 5), activation='relu')` + `BatchNormalization(axis=1)` + `MaxPooling2D((3, 3), strides=(2, 2), padding='same')` + `Dropout(0.3)`
- **Conv Block 3**: `Conv2D(384, (3, 3), activation='relu', padding='same')`
- **Conv Block 4**: `Conv2D(256, (3, 3), activation='relu')` + `MaxPooling2D((3, 3), strides=(2, 2), padding='same')`
- **Fully Connected Head**:
  - `Flatten()`: Vector of size $108,800$
  - `Dense(1024, L2=0.0005, glorot_uniform, activation='relu')` + `Dropout(0.3)`
  - `Dense(128, L2=0.0005, glorot_uniform, activation='relu')`
- **Distance Head**: Custom `Lambda` layer calculating Euclidean distance between the twin 128-d outputs.

```
       Input A (155, 220, 3)                         Input B (155, 220, 3)
                 │                                             │
                 ▼                                             ▼
    ┌─────────────────────────┐                   ┌─────────────────────────┐
    │     Shared ConvNet      │                   │     Shared ConvNet      │
    │  - Conv2D (96, 11x11)   │                   │  - Conv2D (96, 11x11)   │
    │  - BatchNorm + MaxPool  │                   │  - BatchNorm + MaxPool  │
    │  - Conv2D (256, 5x5)    │                   │  - Conv2D (256, 5x5)    │
    │  - BatchNorm + MaxPool  │                   │  - BatchNorm + MaxPool  │
    │  - Conv2D (384, 3x3)    │                   │  - Conv2D (384, 3x3)    │
    │  - Conv2D (256, 3x3)    │                   │  - Conv2D (256, 3x3)    │
    │  - Dense (1024)         │                   │  - Dense (1024)         │
    │  - Dense (128)          │                   │  - Dense (128)          │
    └─────────────────────────┘                   └─────────────────────────┘
                 │                                             │
          Embedding Vector A (128)                      Embedding Vector B (128)
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                                [ Lambda Layer ]
                                Euclidean Distance
                                        │
                                        ▼
                                Distance Scalar (d)
```

---

### 1.4 Training and Evaluation Logic

1. **Loss Function**:
   - For standard CNNs: `binary_crossentropy`.
   - For the CEDAR Siamese model (Cell 106): **Contrastive Loss**:
     $$\mathcal{L}(y, d) = \frac{1}{2N} \sum_{i=1}^N \left[ y_i \cdot d_i^2 + (1 - y_i) \cdot \max(0, m - d_i)^2 \right]$$
     where $m = 1.0$ is the contrastive margin, $d$ is Euclidean distance, and $y$ denotes pair similarity.
2. **Optimizer**: RMSprop (`learning_rate=1e-4, rho=0.9, epsilon=1e-8`) and Adam (`learning_rate=0.001`).
3. **Inference & Scoring Logic (Cell 128)**:
   - Euclidean distance $d$ is converted into similarity:
     $$S = \frac{1}{1 + d}$$
   - Threshold decision rule: If $S > 0.5$ ($d < 1.0$), genuine pair; otherwise, forgery.

---

## 2. Dataset In-Depth Analysis (`/data`)

The `/data` folder contains **6 distinct datasets / structures** totaling more than **26,000 files**. This is a combination of standard academic benchmarks, Kaggle subsets, and real-world crowdsourced signature data:

```
data/
├── BHSig260-Bengali/         # 5,400 Bengali signature images (.tif)
├── BHSig260-Hindi/           # 8,640 Hindi signature images (.tif)
├── CEDAR/                    # 2,640 English signatures grouped by writer (1-55)
├── signatures/               # 2,640 English signatures grouped by full_org / full_forg
├── Dataset_Signature_Final/  # 720 images organized into dataset1..4 (real & forge)
├── sample_Signature/         # 301 images (30 subjects, genuine & forged, NFI naming)
├── data/                     # 6,175 crowdsourced images & CSV task logs
└── assignments_07-02-2022.tsv# Metadata / crowd annotation assignments
```

### 2.1 Dataset Inventory and Features

| Dataset | Total Files | Script / Language | Subjects | Samples per Subject | Format & Dimensions | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CEDAR** (`signatures/` & `CEDAR/`) | 2,640 | English / Latin | 55 writers | 24 Genuine + 24 Forged = 48 | PNG, ~600x300, Grayscale/Palette | Benchmark standard. Highly skilled forgeries where forgers practiced replicating genuine signatures. |
| **BHSig260-Bengali** | 5,400 | Bengali script | 100 writers | 24 Genuine + 30 Forged = 54 | TIFF, ~950x275, 8-bit Grayscale | Non-Latin Indic script benchmark. Features continuous cursive headline strokes (*Matra*). |
| **BHSig260-Hindi** | 8,640 | Devanagari (Hindi) | 160 writers | 24 Genuine + 30 Forged = 54 | TIFF, ~980x280, 8-bit Grayscale | Largest regional script dataset. High visual complexity in ligatures and compound characters. |
| **Dataset_Signature_Final** | 720 | Mixed | Multi-user | 4 splits (dataset1–4), real & forge | PNG, ~220x100 to ~270x90, RGB | Pre-split dataset used in Cells 3–6 of the notebook for baseline testing. |
| **sample_Signature** | 301 | English | 30 writers | 5 Genuine + 5 Forged = 10 | PNG, ~170x60 to ~1350x530 | Systematic naming: `NFI-XXXYYZZZ`. If writer $XXX == ZZZ$, genuine; if $XXX \neq ZZZ$, forged. |
| **Crowdsourced Toloka** (`data/data` + TSV) | 6,175 | Real-world global | 272 real / 252 forged batches | 10–12 samples per batch | JPEG / CSV, high resolution | Collected via mobile cameras/scanners on the Toloka crowdsourcing platform in Feb 2022. |

### 2.2 Forgery Taxonomy Present in the Data
1. **Skilled Forgery (CEDAR, BHSig260)**: The forger has access to the genuine signature and attempts to copy stroke dynamics, proportions, and shape. This represents the most difficult machine learning challenge.
2. **Simple / Unskilled Forgery**: The forger knows the spelling/name of the writer but not the signature style.
3. **Random Forgery**: Signatures from person A tested against genuine signatures of person B (simulated in pairwise negative sampling).

---

## 3. Analysis of the Weights File (`model_weights33.h5`)

### 3.1 Provenance and Direct Mapping
By directly inspecting the HDF5 layer hierarchy and cross-referencing with notebook execution history:
- **Origin**: Generated in **Cell 120** of `signature-verification-using-cnn-snn-csnn.ipynb`:
  ```python
  import h5py
  model.save_weights('model_weights33.h5')
  ```
- **Source Model**: The CEDAR Contrastive Siamese Neural Network defined in Cells 108–109.
- **Training Epochs**: Trained for 5 epochs on 5,520 pairs from CEDAR writers 1–10.

### 3.2 Weight File Anatomy & Layer Breakdown
- **File Size**: **455,880,424 bytes (~455.88 MB)**
- **Root HDF5 Groups**: `['input_36', 'input_37', 'lambda_14', 'sequential_10']`

```
Layer Breakdown in model_weights33.h5:
├── sequential_10 (Shared Siamese Subnetwork)
│   ├── conv11 (Conv2D)
│   │   ├── kernel: (11, 11, 3, 96)       --> 34,848 parameters
│   │   └── bias: (96,)
│   ├── batch_normalization (BatchNorm)
│   │   ├── gamma, beta, moving_mean, moving_variance: (145,) each
│   ├── conv12 (Conv2D)
│   │   ├── kernel: (5, 5, 96, 256)       --> 614,400 parameters
│   │   └── bias: (256,)
│   ├── batch_normalization_1 (BatchNorm)
│   │   ├── gamma, beta, moving_mean, moving_variance: (69,) each
│   ├── conv13 (Conv2D)
│   │   ├── kernel: (3, 3, 256, 384)      --> 884,736 parameters
│   │   └── bias: (384,)
│   ├── conv14 (Conv2D)
│   │   ├── kernel: (3, 3, 384, 256)      --> 884,736 parameters
│   │   └── bias: (256,)
│   ├── dense_34 (Dense)                  <-- DOMINANT LAYER (~445 MB)
│   │   ├── kernel: (108800, 1024)        --> 111,411,200 parameters
│   │   └── bias: (1024,)
│   └── dense_35 (Embedding Output Dense)
│       ├── kernel: (1024, 128)           --> 131,072 parameters
│       └── bias: (128,)
```

> [!NOTE]
> **Why is the weights file ~455 MB?**
> The feature map before `dense_34` has dimension $17 \times 25 \times 256 = 108,800$. Connecting this directly to a 1024-unit dense layer requires $108,800 \times 1,024 = 111,411,200$ float32 weights, which accounts for **~445.6 MB** on its own!

### 3.3 Where and How These Weights Can Be Used
1. **Immediate Pairwise Verification Inference**:
   Instantiate the twin Siamese model using the exact architecture in Cell 108/109 with input shape $(155, 220, 3)$, call `model.load_weights('model_weights33.h5')`, and feed any two preprocessed signature images to obtain the distance scalar.
2. **Transfer Learning / Feature Extractor**:
   Load `sequential_10` as a standalone backbone to convert any signature image into a dense 128-dimensional embedding vector. These vectors can be classified with Cosine Similarity, Support Vector Machines (SVM), or $k$-Nearest Neighbors.
3. **Fine-Tuning on Indian Scripts / Crowdsourced Data**:
   The weights serve as an initial feature representation that can be fine-tuned on `BHSig260-Hindi` or `BHSig260-Bengali`.
4. **Compatibility Constraint**:
   These weights **cannot** be loaded directly into a standard single-image CNN (e.g., Cells 9 or 133) because the tensor shapes, layer names (`sequential_10`), and multi-input topology are specifically bound to the Siamese structure.

---

## 4. Current State & Critical Findings in the Project

While the conceptual foundation (comparing CNNs vs Siamese networks with multiple distance metrics) is solid, an audit of the notebook reveals several bugs and inconsistencies that explain suboptimal results:

### Key Inconsistencies & Bugs in the Notebook:
1. **Synthetic Noise in Distance Comparisons (Cells 36–86)**:
   In the Siamese distance comparison section (Euclidean, Manhattan, Hamming, Minkowski), the code trained on `np.random.rand(100, 224, 224, 3)` (random noise) instead of real signatures, and then evaluated on previous CNN progress variables.
2. **Label Alignment in Contrastive Pair Generation (Cells 114–118)**:
   In Cell 118, `Y_train` was generated with 2,760 ones and 2,760 zeros via `np.insert(Y_train, 0, ...)`. In Cell 106, contrastive loss treats $y=1$ as similar ($d \to 0$) and $y=0$ as dissimilar ($d \ge m$). Mismatched label ordering can invert the optimization objective.
3. **Single-Sample Testing (Cells 122–128)**:
   Evaluation of the trained Siamese network was conducted on only a single pair of images (`original_20_10.png` vs `forgeries_20_10.png`) rather than a held-out test split of unseen writers (writers 11–55).
4. **BatchNorm Axis Anomaly (Cell 108)**:
   `BatchNormalization(axis=1)` was used on image tensors with channels-last format `(batch, height, width, channels)`. The channel axis is `axis=-1` (or 3). Setting `axis=1` normalized along image height ($145$), causing atypical normalization dynamics.
5. **ImageDataGenerator Misconfiguration on Crowdsourced Data (Cell 131)**:
   `real_dir` (272 subfolders) was passed to `train_generator` and `forged_dir` (252 subfolders) to `val_generator`. `ImageDataGenerator` interpreted subfolders as individual classes in binary mode, causing catastrophic loss values (`loss: -2037`).

---

## 5. Recommended Next Steps & Roadmap

To bring this project to completion, improve verification accuracy, and build a strong portfolio piece or academic submission, the following steps are recommended:

### Step 1: Unify the Siamese Pipeline into a Standalone Python Script
Move beyond notebook cells by creating a clean, modular Python codebase:
- `dataset.py`: A clean pairwise data loader that supports writer-independent splits (e.g., Writers 1–40 for training, 41–55 for testing).
- `model.py`: Siamese network architecture with an option for `GlobalAveragePooling2D` to reduce the model size from 455 MB to under 25 MB while improving generalization.
- `losses.py`: Contrastive loss and Triplet loss implementations.
- `evaluate.py`: ROC curve, Equal Error Rate (EER), Precision-Recall curve, and Area Under the ROC Curve (AUC).

### Step 2: Implement Writer-Independent Cross-Validation
In real-world verification systems, the model must verify signatures of **people it has never seen during training**:
- Train on CEDAR subjects 1–40.
- Test on CEDAR subjects 41–55.
- Measure False Acceptance Rate (FAR) and False Rejection Rate (FRR) at different distance thresholds.

### Step 3: Leverage the Multilingual Datasets (`BHSig260`)
The repository contains 14,040 Indian script signature images in `BHSig260-Bengali` and `BHSig260-Hindi` that are currently unused in the notebook. Benchmarking the Siamese network across Latin vs Indic scripts provides substantial academic depth.

### Step 4: Build a Web Demonstration Interface
Build an interactive Streamlit or Gradio UI that allows a user to:
1. Upload an anchor (genuine) signature.
2. Upload a query (test) signature.
3. Load the verification model to display the similarity score, Euclidean distance, and an authentic/forged verification verdict.

---

## 6. Production-Grade Modular Implementation & Benchmarks

In response to the identified architectural bloat and training inconsistencies, a clean, modular, end-to-end implementation was built from scratch:

```
AIDS II Miniproject/
├── dataset.py                # Writer-independent CEDAR dataset parser & balanced pair generator
├── model.py                  # Streamlined Siamese CNN with Global Average Pooling (< 5 MB checkpoint)
├── train.py                  # Contrastive Loss training loop with ReduceLROnPlateau & Early Stopping
├── evaluate.py               # Biometric evaluation module (FAR, FRR, EER, AUC, ROC & Distribution plots)
├── verify.py                 # CLI inference tool for pair authenticity verification
├── best_siamese_model.pth    # Trained Siamese weights checkpoint (4.85 MB)
├── training_history.json     # Epoch-by-epoch loss & accuracy log
└── evaluation_output/        # Empirical evaluation artifacts
    ├── distance_distribution.png  # Histogram & density plot across unseen writers
    ├── roc_curve.png              # Receiver Operating Characteristic curve
    └── evaluation_results.json    # Quantitative biometric performance metrics
```

### 6.1 Architectural Upgrades
| Dimension | Original Notebook (`model_weights33.h5`) | Modular Implementation (`best_siamese_model.pth`) |
| :--- | :--- | :--- |
| **Model Size** | **455.88 MB** (111.4M parameters) | **4.85 MB** (418K parameters) — **99% reduction** |
| **Spatial Transition** | Flattening 108,800 units into Dense(1024) | `AdaptiveAvgPool2d((1, 1))` (Global Average Pooling) |
| **Normalization** | Unconventional BatchNorm on height axis | Channels-first `BatchNorm2d` on feature map channels |
| **Output Space** | Unnormalized dense vector | Unit-hypersphere $L_2$-normalized 128-d embedding |
| **Data Partitioning** | 10 writers trained, single pair tested | **Writer-Independent**: Train (1–35), Val (36–40), Test (41–55) |
| **Pair Balancing** | Arbitrary array insertions | Deterministic $1:1$ balanced positive & negative pairs |

### 6.2 Empirical Evaluation on Unseen Test Writers (Writers 41 to 55)
The model was evaluated on **1,200 test signature pairs** exclusively from unseen individuals:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Area Under ROC Curve (AUC)** | **0.9349** (93.49%) | Outstanding discriminative separation between genuine & forged signatures |
| **Verification Accuracy** | **89.83%** | Evaluated at optimal decision threshold $\tau = 0.5332$ |
| **Recall (Forged Detection)** | **98.50%** | Detects $98.5\%$ of skilled forgery attempts |
| **Precision (Forged)** | **83.95%** | High certainty when flagging a signature as counterfeit |
| **False Acceptance Rate (FAR)**| **1.50%** | Only $1.5\%$ of forged signatures are mistakenly accepted |
| **False Rejection Rate (FRR)**| **18.83%** | Genuine signatures rejected due to high natural intra-writer variation |
| **Equal Error Rate (EER)** | **14.33%** | Operating point where $\text{FAR} \approx \text{FRR}$ at threshold $\tau = 0.6495$ |

### 6.3 Quickstart: Running Inference
To verify authenticity between any two signature images:
```bash
python verify.py --img1 data/signatures/full_org/original_45_1.png --img2 data/signatures/full_org/original_45_2.png
# Output: [+] GENUINE MATCH (Similarity: 84.33%, Distance: 0.3135)

python verify.py --img1 data/signatures/full_org/original_45_1.png --img2 data/signatures/full_forg/forgeries_45_1.png
# Output: [-] FORGERY DETECTED (Similarity: 31.45%, Distance: 1.3710)
```

