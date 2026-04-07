# FragMoE: Interpretable Antioxidant Activity Prediction of Steroidal Saponins

Insights from Machine Learning and Network Pharmacology

## Overview

This repository contains the code, data, and trained models for the paper:

> **Aglycone Cores Drive Antioxidant Activity of Steroidal Saponins from *Polygonatum cyrtonema*: Insights from Interpretable Machine Learning and Network Pharmacology**
>
> Guohao Lv, Lichuan Gu
>
> *Journal of Agricultural and Food Chemistry*, 2026 (Submitted)

This study investigates the structural determinants of antioxidant activity in steroidal saponins from *Polygonatum cyrtonema*, a traditional medicinal food plant. Using interpretable machine learning (FragMoE with Integrated Gradients), we demonstrate that **aglycone cores contribute 3-4× more than glycosylated fragments** to antioxidant activity, validated through network pharmacology, molecular docking (−11.137 kcal/mol for Smilagenin-Keap1), and 100 ns molecular dynamics simulation.

## Key Results

| Assay | n | Champion Model | R² [95% CI] |
|-------|---|---------------|-------------|
| DPPH | 70 | Multi-kernel SVR | 0.655 [0.481, 0.766] |
| ABTS | 42 | Multi-kernel SVR | 0.887 [0.785, 0.936] |
| FRAP | 16 | BayesianRidge | 0.853 [0.745, 0.908] (exploratory) |

## Repository Structure

```
fragmoe/
├── data/                    # Curated antioxidant dataset (91 compounds, 128 records)
├── src/                     # Core source code (27 modules)
│   ├── optimized_svr_v2.py  # Multi-kernel SVR champion model
│   ├── fragmoe_model.py     # FragMoE architecture (GIN + MoE + SAC)
│   └── ...
├── baselines/               # Baseline model implementations
│   ├── 01_traditional_ml.py # RF, XGBoost
│   ├── 04_fair_baselines.py # GPR-Tanimoto, SVR-Tanimoto(std), SVR-RBF(std)
│   └── ...
├── scripts/                 # Experiment scripts
│   ├── ablation_study.py    # 3 ablation experiments
│   ├── statistical_tests.py # Bootstrap CI, Wilcoxon, permutation tests
│   ├── shap_vs_sac_comparison.py  # Cross-method XAI validation
│   └── ...
├── manuscript/              # LaTeX manuscript files
│   ├── main.tex             # Main manuscript (BIB format)
│   ├── supplementary.tex    # Supplementary information
│   ├── cover_letter.tex     # Cover letter
│   └── 07_references.bib    # BibTeX references
├── results/                 # Experiment results and figures
├── configs/                 # Configuration files
├── environment.yml          # Conda environment (pinned versions)
├── REPRODUCE.md             # Step-by-step reproduction guide
└── LICENSE                  # MIT License
```

## Quick Start

### Environment Setup

```bash
conda env create -f environment.yml
conda activate fragmoe
```

### Reproduce All Results

```bash
# Run complete benchmark pipeline
python scripts/run_reproducible_benchmark.py

# Verify result integrity
python scripts/check_result_integrity.py
```

### Run Individual Components

```bash
# Champion model (multi-kernel SVR)
python src/optimized_svr_v2.py

# Fair baselines (GPR, standard SVR)
python baselines/04_fair_baselines.py

# Ablation study
python scripts/ablation_study.py

# Statistical tests (bootstrap CI, Wilcoxon, permutation)
python scripts/statistical_tests.py

# Cross-method XAI validation (SHAP vs permutation importance)
python scripts/shap_vs_sac_comparison.py
```

## Methods Summary

1. **Per-Assay Specialization**: Independent champion model selection per assay via nested LOOCV
2. **Multi-Kernel SVR**: K = w1*Dice(Morgan) + w2*Tanimoto(MACCS) + w3*RBF(saponin)
3. **FragMoE Module**: BRICS fragmentation -> GIN encoding -> MoE (4 experts) -> SAC attention
4. **Statistical Rigor**: Bootstrap 95% CI (5000 resamples), Wilcoxon signed-rank tests, permutation tests
5. **Multi-Scale Validation**: Network pharmacology -> Molecular docking (Keap1) -> 10 ns MD simulation

## Data Availability

All datasets supporting this study are openly available in this GitHub repository under the `data/` directory.

### Current Structure (Pre-acceptance)

Raw data files are currently organized in:
- `data/processed/` - Processed datasets
- `data/raw/` - Raw data files
- `data/splits/` - Train/validation/test splits

### Final Structure (Post-acceptance)

After paper acceptance, data will be reorganized into the following structure (see `data/README.md`):

```
data/
├── 01_dataset/           # Main dataset (91 compounds, 128 records)
├── 02_structures/        # 3D molecular structures
├── 03_targets/           # Predicted targets (127 targets)
├── 04_results/           # Pathway enrichment and predictions
└── 05_md/                # MD simulation data
```

### Dataset Summary

| Statistic | Value |
|-----------|-------|
| **Total compounds** | 91 unique molecules |
| **Total activity records** | 128 |
| - DPPH assay | 70 records |
| - ABTS assay | 42 records |
| - FRAP assay | 16 records (exploratory) |
| *P. cyrtonema* saponins | 16 compounds (17.6%) |
| **Molecular weight range** | 138-1065 Da |
| **Predicted targets** | 127 proteins |

No additional registration or access request is required. All data can be directly downloaded from this repository.

## Citation

```bibtex
@article{fragmoe2026,
  title={Aglycone Cores Drive Antioxidant Activity of Steroidal Saponins from
         Polygonatum cyrtonema: Insights from Interpretable Machine Learning and
         Network Pharmacology},
  author={Lv, Guohao and Gu, Lichuan},
  journal={Journal of Agricultural and Food Chemistry},
  year={2026},
  publisher={American Chemical Society}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Reproducibility

All random seeds are fixed at 42. See [REPRODUCE.md](REPRODUCE.md) for detailed reproduction instructions including expected outputs and verification checksums.

**Environment**: Python 3.11.7, scikit-learn 1.8.0, RDKit 2022.09.5, PyTorch 2.5.1, XGBoost 3.2.0
