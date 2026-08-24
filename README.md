# Privacy-Aware Continual Knowledge Distillation for Cross-Domain Medical Image Classification

> **Advanced Programming for AI** — MSc Artificial Intelligence, University of Verona (2024–2025)
> Course Instructor: Prof. Cigdem Beyan

---

## Overview

This project investigates **continual learning** for sequential medical image classification under a **privacy-aware** framework. A ResNet-50 teacher model is trained on chest X-ray pneumonia detection (Task A) and distilled into a lightweight MobileNetV3 student via knowledge distillation. The student is then incrementally adapted to brain tumor MRI classification (Task B) without revisiting Task A data — a setting that mirrors real-world clinical deployment where hospitals continuously receive new imaging modalities.

**Novel contribution:** We audit whether the continual learning premise ("no stored data = private") holds in practice by running a Membership Inference Attack (MIA) on the trained model, then apply Differential Privacy (DP-SGD) to mitigate the identified leakage, charting the resulting privacy–utility trade-off.

---

## Key Results

### Continual Learning (4 methods × 4 teacher–student pairs)

| Method | Task A Retention | Task B Accuracy |
|--------|-----------------|-----------------|
| Naive fine-tuning | 74.3% | 94.5% |
| EWC | 93.9% | 93.9% |
| LwF | 75.0% | 95.1% |
| **BN-Freeze (ours)** | **100.0%** | **90.0%** |

BN-Freeze achieves **perfect Task A retention** across all 4 teacher–student combinations (ResNet-50/ViT-B16 × MobileNetV3/EfficientNet-B0), confirming that architectural stabilization outperforms algorithmic regularization alone.

### Privacy Study (MIA + DP-SGD)

| ε (privacy budget) | Task A Accuracy | MIA AUC |
|--------------------|----------------|---------|
| ∞ (no DP) | 91.9% | 0.526 |
| 8.0 | 85.4% | 0.512 |
| 3.0 | 77.4% | 0.506 |
| 1.0 | 86.1% | 0.503 |

MIA AUC monotonically decreases as ε decreases — DP-SGD successfully reduces membership leakage toward the random-guess baseline (0.50).

### EWC λ Ablation

| λ | Task A Retention | Task B Accuracy |
|---|-----------------|-----------------|
| 0 | 86.3% | 93.3% |
| 500 | 95.0% | 95.9% |
| **2000** | **95.4%** | **96.6%** |
| 8000 | 90.6% | 93.1% |

Optimal λ = 2000 — over-regularization at λ = 8000 hurts both retention and plasticity.

---

## Pipeline

```
Stage 1 — Teacher Pre-training
    ResNet-50 / ViT-B16 trained on Task A (Chest X-ray, 2 classes)
    Teacher accuracy: 96.59% (ResNet-50)

Stage 2 — Knowledge Distillation
    Teacher → MobileNetV3 / EfficientNet-B0 student
    Logit-based KD (temperature T=3, α=0.5)
    Student accuracy: 98.18% (beats teacher via KD)

Stage 3 — Continual Learning
    Student adapted to Task B (Brain Tumor MRI, 4 classes)
    4 methods: Naive / EWC / LwF / BN-Freeze
    Evaluated: Task A retention + Task B accuracy

Stage 4 — Privacy Audit
    Membership Inference Attack on trained continual model
    DP-SGD mitigation with ε ∈ {8.0, 3.0, 1.0}
    Privacy–utility trade-off measured via MIA AUC vs accuracy
```

---

## Datasets

| Task | Dataset | Classes | Size |
|------|---------|---------|------|
| Task A | [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | NORMAL, PNEUMONIA | ~5,863 |
| Task B | [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | glioma, meningioma, notumor, pituitary | ~7,023 |

Both datasets are automatically downloaded via `kagglehub` — no manual download required.

---

## Project Structure

```
APAI_Project/
├── config.py              # All hyperparameters (one place)
├── main.ipynb             # Main notebook — full pipeline
├── src/
│   ├── data.py            # Dataset download, preprocessing, loaders
│   ├── models.py          # Teacher (ResNet-50, ViT-B16) + Student (MobileNetV3, EfficientNet-B0)
│   ├── engine.py          # Train loop, evaluation, checkpointing
│   ├── distillation.py    # Logit-based knowledge distillation
│   ├── continual.py       # Naive / EWC / LwF / BN-Freeze
│   ├── privacy.py         # MIA (loss-based attack) + manual DP-SGD
│   ├── experiments.py     # 2-teacher × 2-student pair runner + λ sweep
│   └── plots.py           # All paper figures
├── checkpoints/           # Saved model weights (.pth)
└── results/               # CSVs + figures/
```

---

## Requirements

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install kagglehub pandas matplotlib seaborn scikit-learn timm jupyter
```

> **Note:** This project uses a manual DP-SGD implementation (no Opacus dependency) to avoid torch/opacus version conflicts. The clip-then-noise update rule is implemented directly in `src/privacy.py`.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Iam-Taki/Privacy-Aware-Continual-Knowledge-Distillation-APAI-.git
cd Privacy-Aware-Continual-Knowledge-Distillation-APAI-

# 2. Install dependencies
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install kagglehub pandas matplotlib seaborn scikit-learn timm jupyter

# 3. Run the notebook
jupyter notebook main.ipynb
```

Run cells in order — checkpoints are saved automatically, so interrupted runs can resume from where they left off.

---

## Implementation Notes

- **Config-driven:** all hyperparameters live in `config.py`, nothing is hardcoded
- **Checkpoint-based:** every stage saves `.pth` files; re-running loads from disk
- **Reproducible:** global seed (`SEED=42`) set everywhere
- **No Opacus:** DP-SGD implemented manually in `src/privacy.py` using plain PyTorch — avoids the torch/opacus version conflict that repeatedly broke CUDA support on this machine
- **BN-Freeze fix:** the original implementation only froze BN affine parameters, leaving conv weights trainable (which caused Task A statistics mismatch and worse-than-naive retention). The fix freezes the **entire backbone** during Task B adaptation

---

## Authors

| Name | Student ID | University |
|------|-----------|------------|
| Abdullah Al Noman Taki | VR528988 | University of Verona |
| [Teammate 2] | [ID] | University of Verona |
| [Teammate 3] | [ID] | University of Verona |

---

## Course

**Advanced Programming for AI (APAI)**
MSc Artificial Intelligence — University of Verona
Academic Year 2024–2025
Instructor: Prof. Cigdem Beyan
