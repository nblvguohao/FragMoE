# FragMoE Dataset

## Overview

This directory contains the curated datasets for the JAFC paper:

> **Aglycone Cores Drive Antioxidant Activity of Steroidal Saponins from *Polygonatum cyrtonema*: Insights from Interpretable Machine Learning and Network Pharmacology**

## Directory Structure

```
data/
├── 01_dataset/           # Main datasets
├── 02_structures/        # 3D molecular structures
├── 03_targets/           # Predicted targets and pathway genes
├── 04_results/           # Analysis results
├── 05_md/                # MD simulation data (optional)
├── processed/            # Legacy processed data (to be reorganized)
├── raw/                  # Legacy raw data (to be reorganized)
└── splits/               # Legacy splits (to be reorganized)
```

## Data Organization (Post-Acceptance)

After paper acceptance, the following files will be added to this directory:

### 01_dataset/
- `antioxidant_dataset.csv` - Main dataset (91 compounds, 128 records)
- `saponins_annotated.csv` - Annotated compound information
- `scaffold_split.json` - Train/validation/test splits

### 02_structures/
- `saponins_3d.sdf` - 3D molecular structures (SDF format)

### 03_targets/
- `targets_predicted.csv` - Predicted protein targets (127 targets)
- `antioxidant_pathway_genes.csv` - Antioxidant pathway gene sets

### 04_results/
- `pathway_enrichment.csv` - KEGG/Reactome pathway enrichment results
- `model_predictions.csv` - Model predictions for all compounds

### 05_md/md_trajectories/
- MD simulation trajectories (optional, large files)

## Current Status

**Pre-acceptance**: The raw data files are currently in `processed/`, `raw/`, and `splits/` directories.

**Post-acceptance**: All data will be reorganized into the numbered directories above for clarity.

## Dataset Statistics

- **Total unique compounds**: 91
- **Total activity records**: 128
  - DPPH assay: 70 records
  - ABTS assay: 42 records
  - FRAP assay: 16 records (exploratory)
- *P. cyrtonema* steroidal saponins: 16 compounds (17.6%)
- **Molecular weight range**: 138-1065 Da
- **Predicted protein targets**: 127

## License

CC-BY-4.0 (See LICENSE.txt in repository root)

## Citation

```bibtex
@article{fragmoe2024,
  title={Aglycone Cores Drive Antioxidant Activity of Steroidal Saponins from
         Polygonatum cyrtonema: Insights from Interpretable Machine Learning and
         Network Pharmacology},
  author={Lv, Guohao and Gu, Lichuan},
  journal={Journal of Agricultural and Food Chemistry},
  year={2024}
}
```
