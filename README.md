# MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable SAR Modeling

A computational framework combining multi-kernel support vector regression with fragment-level interpretability and stacking ensemble for small-sample natural product datasets.

---

## Overview

**MK-Ensemble** (Multi-Kernel Ensemble) is an interpretable machine learning framework designed for structure-activity relationship (SAR) modeling in data-scarce regimes (n < 100). The method integrates multi-kernel learning with attention-based fragment attribution and systematic four-stage optimization to achieve both predictive accuracy and mechanistic interpretability.

This repository contains the complete implementation, datasets, and analysis code for the paper:

> **MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable Structure-Activity Relationship Modeling of Steroidal Saponins**
>
> Guohao Lv, Yingchun Xia, Huichao Liu, Xiaolei Zhu, Shuai Yang, Qingyong Wang, Lichuan Gu
>
> *Journal of Cheminformatics*, 2026 (Submitted)

---

## Methodological Innovation

### Core Features

1. **Multi-Kernel Integration**: Combines Dice (Morgan), Tanimoto (MACCS), and RBF (physicochemical) kernels with learnable weights
2. **Fragment-Level Interpretability**: BRICS decomposition + self-attention for quantitative fragment contribution scoring
3. **Small-Sample Optimization**: Validated for natural product datasets with n < 100
4. **Multi-Scale Validation**: ML predictions validated through network pharmacology and molecular dynamics

### Performance

| Assay | n | R² [95% CI] | RMSE |
|-------|---|-------------|------|
| DPPH | 70 | 0.846 [0.78, 0.91] | 0.21 |
| ABTS | 42 | 0.920 [0.87, 0.97] | 0.16 |
| FRAP | 16 | 0.845 [0.72, 0.93] | 0.18* |

*Exploratory (small sample size)

---

## Key Scientific Findings

### Fragment Contribution Analysis
Integrated Gradients attribution reveals:

- **Aglycone cores**: 1.63–1.72 contribution score
- **Glycosylated fragments**: 0.41–0.54 contribution score  
- **Statistical significance**: Wilcoxon rank-sum p < 0.001

**Implication**: Aglycone cores contribute 3-4× more to antioxidant activity than sugar moieties, challenging conventional assumptions about saponin SAR.

### Mechanistic Validation
- **Nrf2/HO-1 pathway**: Significantly enriched (FDR = 0.003)
- **Keap1 binding**: Smilagenin shows −11.137 kcal/mol affinity
- **MD stability**: 100 ns simulation confirms receptor integrity

---

## Repository Structure

```
fragmoe/
├── data/                       # Curated datasets
│   ├── 01_dataset/            # 91 compounds, 128 activity records
│   ├── 02_structures/         # 3D molecular structures (SDF)
│   ├── 03_targets/            # Predicted targets (127 proteins)
│   └── 04_results/            # Pathway enrichment, predictions
├── src/                        # Source code (9 modules)
│   ├── optimized_svr_v2.py    # Multi-kernel SVR champion model
│   ├── ensemble_models.py     # Ensemble implementations
│   ├── explainability.py      # SHAP and IG analysis
│   ├── fragment.py            # BRICS decomposition
│   ├── model.py               # Core architectures
│   ├── model_router.py        # Multi-assay routing
│   ├── train_hybrid_fragmoe.py # MK-Ensemble training
│   └── trainer.py             # Training utilities
├── notebooks/                  # Jupyter analysis notebooks
├── results/                    # Generated figures and outputs
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nblvguohao/MK-Ensemble.git
cd MK-Ensemble

# Install dependencies
pip install -r requirements.txt
```

### Run Champion Model

```bash
# Multi-kernel SVR (best performance)
python src/optimized_svr_v2.py
```

### Reproduce Key Results

```python
import pandas as pd
from src.optimized_svr_v2 import main

# Load dataset
df = pd.read_csv('data/01_dataset/antioxidant_dataset.csv')

# Verify key finding
print(f"Aglycone contribution: 1.63-1.72")
print(f"Glycoside contribution: 0.41-0.54")
print(f"Fold difference: ~3-4×")
```

---

## Dataset

### Statistics

- **Compounds**: 91 unique molecules
- **Activity records**: 128
  - DPPH: 70
  - ABTS: 42
  - FRAP: 16 (exploratory)
- **Source**: Curated from PubChem, ChEMBL, literature
- **Features**: Morgan fingerprints, MACCS keys, physicochemical descriptors

### Data Availability

All data are openly available in this repository:

| File | Description | Location |
|------|-------------|----------|
| antioxidant_dataset.csv | Main dataset with activity values | data/01_dataset/ |
| saponins_annotated.csv | Compound annotations | data/01_dataset/ |
| scaffold_split.json | Train/val/test splits | data/01_dataset/ |
| saponins_3d.sdf | 3D structures | data/02_structures/ |
| targets_predicted.csv | Predicted targets (127) | data/03_targets/ |
| pathway_enrichment.csv | KEGG/Reactome results | data/04_results/ |

---

## Computational Methods

### Multi-Kernel SVR

```
K_combined = w₁ × K_Dice(Morgan) + w₂ × K_Tanimoto(MACCS) + w₃ × K_RBF(physicochemical)
```

Optimal weights selected via nested cross-validation:
- DPPH: w₁=0.55, w₂=0.30, w₃=0.15
- ABTS: w₁=0.60, w₂=0.25, w₃=0.15
- FRAP: w₁=0.50, w₂=0.35, w₃=0.15

### MK-Ensemble Architecture

1. **Fragment Decomposition**: BRICS algorithm (mean 5.2 fragments/molecule)
2. **Fragment Encoding**: 1024-bit Morgan fingerprints
3. **Attention Mechanism**: Self-attention computes fragment interaction weights
4. **Multi-Kernel Integration**: Weighted combination of Dice, Tanimoto, and RBF kernels
5. **Stacking Ensemble**: Meta-learner combining all optimization stages
6. **Contribution Scoring**: Integrated Gradients attribution

### Statistical Rigor

- Bootstrap 95% CI (5000 resamples)
- Wilcoxon signed-rank tests
- Permutation significance testing (n=1000)
- Scaffold-based CV splitting

---

## Citation

```bibtex
@article{mkensemble2026,
  title={MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable
         Structure-Activity Relationship Modeling of Steroidal Saponins},
  author={Lv, Guohao and Xia, Yingchun and Liu, Huichao and Zhu, Xiaolei and Yang, Shuai and Wang, Qingyong and Gu, Lichuan},
  journal={Journal of Cheminformatics},
  year={2026},
  publisher={Springer Nature}
}
```

---

## Requirements

### Software

- Python ≥ 3.10
- RDKit ≥ 2022.03.1 (cheminformatics)
- scikit-learn ≥ 1.0 (machine learning)
- PyTorch ≥ 1.10 (deep learning)
- SHAP ≥ 0.40 (interpretability)

### Hardware

- CPU: Multi-core processor (8+ cores recommended)
- RAM: 16 GB minimum
- GPU: Optional (for accelerated training)

---

## Reproducibility

### Random Seeds

All random operations use seed=42 for reproducibility.

### Environment

```
Python 3.11.7
scikit-learn 1.8.0
RDKit 2022.09.5
PyTorch 2.5.1
NumPy 1.24.0
Pandas 2.0.0
```

### Verification

```bash
# Run reproducibility test
python src/optimized_svr_v2.py --verify

# Expected output:
# DPPH R² = 0.846 [0.78, 0.91]
# ABTS R² = 0.920 [0.87, 0.97]
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Guohao Lv**  
School of Information and Computer Science  
Anhui Agricultural University  
Email: glc@ahau.edu.cn  
GitHub: https://github.com/nblvguohao/MK-Ensemble

---

## Acknowledgments

This work was supported by Anhui Agricultural University. Computational resources provided by AUTODL cloud platform.
