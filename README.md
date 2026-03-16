#  Car Damage Detection — Transfer Learning with Hyperparameter Optimization

> **Computer Vision · Multi-Class Classification · Deep Learning**  
> Automated vehicle damage classification to simulate real-world insurance claim triaging.

---

##  Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Data Preparation](#-data-preparation--preprocessing)
- [Modeling Strategy](#-modeling-strategy)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Skills Demonstrated](#-skills-demonstrated)

---

##  Overview

This project develops a deep learning model to **classify vehicle damage from images** across six categories, simulating the automated triage layer of an insurance claim pipeline. Working with a limited labeled dataset (~2,300 images), the system leverages **transfer learning with a pretrained ResNet backbone** to achieve stable generalization — without training from scratch.

Hyperparameter optimization via **Optuna** was used to fine-tune the learning rate and dropout, resulting in improved validation accuracy and training stability. The trained model is deployed with a **FastAPI backend** and an interactive **Streamlit frontend**.

---

##  Demo

The model is deployed via two components — a **FastAPI backend** that serves the model, and a **Streamlit frontend** for interactive image upload and prediction.

> 📌 The model expects **third-quarter front or rear view** images of the vehicle for best performance.

![img_2.png](img_2.png)
---
![img_3.png](img_3.png)
##  Dataset

| Property | Details |
|----------|---------|
| Total Images | ~2,300 labeled vehicle images |
| Task Type | Multi-class classification |
| Backbone Input | Fixed CNN input dimensions (resized) |
| Risk | High overfitting risk due to limited data size |

### Target Classes

| Label | Description |
|-------|-------------|
| `F_Normal` | Front view — no damage |
| `F_Crushed` | Front view — crushed/dented damage |
| `F_Breakage` | Front view — broken parts (lights, bumper, etc.) |
| `R_Normal` | Rear view — no damage |
| `R_Crushed` | Rear view — crushed/dented damage |
| `R_Breakage` | Rear view — broken parts |

---

##  Data Preparation & Preprocessing

- Resized all images to fixed CNN input dimensions compatible with ResNet
- Applied normalization and tensor transformation for PyTorch compatibility
- Implemented **data augmentation** to combat overfitting:
  - Random rotation
  - Horizontal flipping
  - Brightness & contrast adjustments
- Performed stratified train-validation split
- Built a custom **PyTorch `Dataset` and `DataLoader`** pipeline for efficient batch loading

---

##  Modeling Strategy

### Base Architecture

Transfer learning was applied using a **pretrained ResNet backbone** (pretrained on ImageNet).

```
ResNet Backbone (frozen weights)
        ↓
Custom Classification Head
        ↓
Dropout Layer  →  FC Layer  →  Softmax (6 classes)
```

### Fine-Tuning Strategy

- **Frozen feature extractor** — backbone weights were not updated during training
- Only the **classification head** was trained on the damage-labeled dataset
- This approach compensates for the small dataset size while leveraging rich pretrained feature representations

### Hyperparameter Optimization with Optuna

[Optuna](https://optuna.org/) was used to run an automated search over the hyperparameter space:

| Hyperparameter | Search Space | Best Value Found |
|---|---|---|
| Learning Rate | Log-uniform | **0.005** |
| Dropout Rate | Continuous | **0.20** |

Optuna's pruning and trial history were used to efficiently converge on the best configuration.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **78%** |
| Architecture | ResNet + Custom Head |
| Optimizer | Adam |
| Best Learning Rate | 0.005 |
| Best Dropout | 0.20 |

- ✅ Stable training convergence via transfer learning
- ✅ Reduced overfitting with dropout regularization + augmentation
- ✅ Confusion matrix analysis used to evaluate per-class performance
- ⚠️ Visually similar damage classes (e.g., Crushed vs. Breakage) showed higher misclassification rates — expected given dataset scale

> Given only ~2,300 labeled images, transfer learning enabled the model to achieve **meaningful generalization** that training from scratch would not have reached.

---

## 💡 Key Insights

1. **Transfer learning vs. training from scratch** — Transfer learning significantly outperformed a randomly initialized model, confirming that pretrained feature extractors provide a strong inductive bias even for domain-specific tasks.
2. **Optuna-driven tuning** — Automated hyperparameter search with Optuna improved both validation accuracy and convergence stability versus manual tuning.
3. **Class confusion patterns** — Visually similar damage types (e.g., Crushed vs. Breakage) were more likely to be confused, suggesting that additional data or finer-grained augmentation strategies could further improve performance.
4. **Data constraints as ceiling** — The upper bound on model performance is constrained by dataset size (~2,300 images across 6 classes). Scaling data collection would likely yield the largest accuracy gains.

---

##  Project Structure

```
car-damage-detection/
│
├── fastapi-server/
│   ├── model/                        # Model weights for API serving
│   ├── model_helper.py               # Model loading & inference logic
│   └── server.py                     # FastAPI server for model serving
│
├── streamlit app/
│   ├── model/
│   │   └── saved_model.pth           # Trained model weights
│   ├── app.py                        # Streamlit web app (inference interface)
│   ├── model_helper.py               # Helper functions for prediction
│   ├── requirements.txt              # Python dependencies
│   ├── img.png                       # Sample images
│   └── img_1.png
│
├── training/
│   ├── dataset/                      # Training image data
│   ├── damage_prediction.ipynb       # Main training notebook
│   └── hyperparameter_tunning.ipynb  # Optuna HPO notebook
│
└── README.md
```

---

##  Setup & Installation

**Prerequisites:** Python 3.8+

1. **Clone the repository**
   ```bash
   git clone https://github.com/bharath2903/car-damage-detection.git
   cd car-damage-detection
   ```

2. **Install dependencies**
   ```bash
   cd "streamlit app"
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Start the FastAPI Server

```bash
cd fastapi-server
uvicorn server:app --reload
```

The API will be available at `http://localhost:8000`. It exposes an endpoint that accepts an image and returns the predicted damage class.

### 2. Run the Streamlit App

```bash
cd "streamlit app"
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, upload a vehicle image, and view the predicted damage category.

### 3. Explore Training Notebooks

Open the notebooks inside the `training/` folder:

| Notebook | Description |
|----------|-------------|
| `damage_prediction.ipynb` | End-to-end model training with ResNet |
| `hyperparameter_tunning.ipynb` | Optuna HPO search for learning rate & dropout |

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-5b86e5?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

| Category | Tools |
|----------|-------|
| Framework | PyTorch, Torchvision |
| Architecture | ResNet (Transfer Learning) |
| HPO | Optuna |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Evaluation | Confusion Matrix, Validation Accuracy |
| Augmentation | Torchvision Transforms |

---

##  Skills Learnt

- **Deep Learning** — End-to-end CNN pipeline design and training in PyTorch
- **Transfer Learning** — Leveraging pretrained ImageNet weights for limited-data settings
- **Hyperparameter Optimization** — Automated search using Optuna (learning rate, dropout)
- **Computer Vision** — Image preprocessing, augmentation, and multi-class classification
- **Model Evaluation** — Validation accuracy tracking and confusion matrix analysis
- **ML Deployment** — Model served via FastAPI backend + Streamlit frontend

---

##  License

This project was completed as part of a  data science bootcamp at CodeBasics. Feel free to use it as a reference or build upon it.

---

<p align="center">Made with ❤️ and PyTorch</p>
