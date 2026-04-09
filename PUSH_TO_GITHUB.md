# Push to GitHub Instructions

## Repository
**URL**: https://github.com/nblvguohao/MK-Ensemble.git

## Steps

### 1. Initialize Repository (if not already done)
```bash
cd /path/to/github_release
git init
git remote add origin https://github.com/nblvguohao/MK-Ensemble.git
```

### 2. Add Files
```bash
git add .
git commit -m "Initial release: MK-Ensemble JAFC submission data

- Dataset: 91 compounds, 128 activity records
- Model performance: DPPH R²=0.655, ABTS R²=0.887
- Docking: Smilagenin -11.137 kcal/mol (Keap1)
- MD: 100ns GROMACS simulation results
- Analysis notebooks for reproducibility"
```

### 3. Push to GitHub
```bash
git push -u origin main
```

### 4. Create Release (Optional)
Go to GitHub → Releases → Create New Release
- Tag: v1.0.0
- Title: JAFC Submission Release
- Description: See README.md

## Files Included
- README.md
- LICENSE (MIT)
- data/ - Dataset tables
- results/ - MD analysis plots, docking pose
- notebooks/ - Reproducible analysis
