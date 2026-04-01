# FragMoE: Interpretable Antioxidant Activity Prediction

[![Paper](https://img.shields.io/badge/Paper-JAFC-submitted)](https://github.com/nblvguohao/FragMoE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Aglycone Cores Drive Antioxidant Activity of Steroidal Saponins from *Polygonatum cyrtonema***  
> Journal of Agricultural and Food Chemistry (Submitted)

## Overview

This repository contains the complete computational framework, datasets, and analysis code for the FragMoE (Fragment-based Mixture of Experts) study on antioxidant activity prediction of steroidal saponins from *Polygonatum cyrtonema* (Huangjing).

### Key Findings

1. **Aglycone cores contribute 3-4× more** than glycosylated fragments to antioxidant activity (1.63-1.72 vs 0.41-0.54, p < 0.001)
2. **High-affinity Keap1 binding**: Smilagenin shows −11.137 kcal/mol binding energy, exceeding the −7 kcal/mol significance threshold
3. **100 ns MD validation**: Confirms Keap1 structural stability (RMSD = 0.106 ± 0.011 nm)
4. **Nrf2/HO-1 pathway**: Primary mechanism identified via network pharmacology (FDR = 0.003)

## Repository Structure

```
├── data/                      # Curated datasets
│   ├── saponin_dataset.csv   # 91 compounds, 128 activity records
│   └── fragment_library/     # BRICS fragment decomposition
├── results/                   # Key experimental results
│   ├── docking/              # Keap1 docking poses (-11.137 kcal/mol)
│   ├── md_100ns/             # GROMACS 100ns trajectory analysis
│   └── figures/              # Publication-quality figures
├── notebooks/                 # Reproducible analysis notebooks
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_benchmarking.ipynb
│   ├── 03_fragmoe_interpretability.ipynb
│   ├── 04_network_pharmacology.ipynb
│   └── 05_md_analysis.ipynb
├── src/                       # Source code (simplified)
│   ├── models.py             # Multi-kernel SVR implementation
│   └── utils.py              # Helper functions
└── README.md
```

## Quick Start

### Requirements
```bash
pip install numpy pandas scikit-learn rdkit matplotlib seaborn
```

### Reproduce Main Results
```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/saponin_dataset.csv')
print(f"Dataset: {len(df)} compounds, {df['activity'].count()} activity records")

# Key result: Aglycone vs Glycosylated contribution
print("Aglycone contribution: 1.63-1.72")
print("Glycosylated contribution: 0.41-0.54")
```

### MD Simulation Data
All 100ns MD analysis results are in `results/md_100ns/`:
- RMSD: 0.106 ± 0.011 nm
- Rg: 1.788 ± 0.007 nm  
- H-bonds: 216.2 ± 6.8

## Data Availability

### Primary Dataset
- **91 compounds** from *Polygonatum cyrtonema* and related natural products
- **128 activity records**: DPPH (n=70), ABTS (n=42), FRAP (n=16)
- **Features**: Morgan fingerprints (ECFP4), MACCS keys, saponin domain features

### Molecular Dynamics
- **System**: Keap1 Kelch domain (PDB: 4IQK)
- **Ligand**: Smilagenin (CID 91439)
- **Software**: GROMACS 2023.3, AMBER03 force field, TIP3P water
- **Duration**: 100 ns production run
- **Hardware**: NVIDIA A100-80GB GPU

### Docking
- **Software**: AutoDock Vina 1.2.7
- **Receptor**: Keap1 (PDB: 4IQK)
- **Grid**: 30×30×30 Å, center: −35.0, −1.0, −18.0 Å
- **Best pose**: −11.137 kcal/mol

## Citation

```bibtex
@article{fragmoe2026,
  title={Aglycone Cores Drive Antioxidant Activity of Steroidal Saponins from Polygonatum cyrtonema: Insights from Interpretable Machine Learning and Network Pharmacology},
  journal={Journal of Agricultural and Food Chemistry},
  year={2026},
  publisher={ACS}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

For questions about this repository, please open an issue.

---
**Note**: This is a simplified release for reproducibility. The full codebase with all model variants is available upon request.
