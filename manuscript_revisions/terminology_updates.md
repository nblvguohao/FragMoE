# 术语修改清单

## 1. "Mixture-of-Experts" → "Multi-Kernel Ensemble"

### 必须修改的位置

#### Abstract
**原文**:
> We developed FragMoE (Fragment-based Mixture-of-Experts), a machine learning framework...

**修改为**:
> We developed FragMoE (Fragment-based Multi-Kernel Ensemble), a machine learning framework...

#### Introduction
**原文**:
> This paper presents FragMoE (Fragment-based Mixture-of-Experts)...

**修改为**:
> This paper presents FragMoE (Fragment-based Multi-Kernel Ensemble)...

#### Methods (2.2节标题)
**原文**:
> 2.2.2 Fragment Decomposition and Attention

**修改为**:
> 2.2.2 Fragment Decomposition and Multi-Kernel Ensemble

**段落修改**:
**原文**:
> The attention-weighted fragment representation is concatenated with molecular-level features for final prediction.

**修改为**:
> The kernel-weighted fragment representation is concatenated with molecular-level features for final prediction.

#### Discussion
**原文**:
> The multi-kernel approach combines molecular representations without requiring large training sets, and fragment-based interpretability yields chemical insights...

**修改为**:
> FragMoE employs a multi-kernel ensemble that combines molecular representations without requiring large training sets, and fragment-based interpretability yields chemical insights...

---

## 2. 性能声称修改

### Abstract - Results部分
**原文**:
> FragMoE achieved stronger predictive performance (DPPH: R² = 0.655 [0.481, 0.766], ABTS: R² = 0.887 [0.785, 0.936]) than baseline methods.

**修改为**:
> FragMoE achieved comparable or superior predictive performance compared to baseline methods (DPPH: R² = 0.588 vs Random Forest R² = 0.623, p = 0.596; ABTS: R² = 0.867 vs Random Forest R² = 0.795, p = 0.629), with particularly strong results on ABTS assay.

### Results (3.1节)
**原文**:
> FragMoE outperformed baseline methods across all assays (Table 1).

**修改为**:
> FragMoE demonstrated competitive or superior performance compared to baseline methods (Table 1). On ABTS, FragMoE achieved the highest R² (0.867), while on DPPH, performance was comparable to Random Forest (R² = 0.588 vs 0.623, p = 0.596).

---

## 3. FRAP数据处理

### Abstract - Results部分
**原文**:
> FRAP (n = 16)

**修改为**:
> FRAP exploratory analysis (n = 16, see Supplementary)

### Methods (2.1节)
**添加说明**:
> **Note on FRAP assay**: Due to limited sample size (n = 16), FRAP results are presented as exploratory analysis in Supplementary Table S2 and are not included in the main performance comparison.

### Results (3.1节)
**修改**:
> For the primary DPPH assay (n = 70) and ABTS assay (n = 42), the model achieved R² = 0.588 [0.500, 0.676] and R² = 0.867 [0.812, 0.922], respectively. FRAP results (n = 16) are provided as exploratory analysis (Supplementary Table S2).

---

## 4. 统计显著性说明

### Results (3.1节后添加)
**新增段落**:
> Statistical comparisons using Wilcoxon signed-rank tests showed that FragMoE performed significantly better than XGBoost on both DPPH (p = 0.019) and ABTS (p = 0.014), and significantly better than SVR-RBF on ABTS (p < 0.001). Performance differences between FragMoE and Random Forest were not statistically significant on either assay (DPPH: p = 0.596; ABTS: p = 0.629; Table S1).

---

## 5. Discussion修改

### 添加局限性讨论
**在Discussion中添加**:
> **Performance characteristics**: While FragMoE demonstrated strong performance on ABTS, its performance on DPPH was comparable to but not significantly better than Random Forest. This pattern suggests that the advantages of fragment-based modeling may vary depending on the specific antioxidant mechanism. The ABTS assay, involving both hydrogen atom transfer and electron transfer, may benefit more from the multi-kernel approach than DPPH, which primarily measures hydrogen atom transfer.

> **Sample size limitations**: The dataset size, particularly for FRAP (n = 16), limits the statistical power of our conclusions. Future work will expand the dataset through systematic literature mining to validate the fragment contribution findings across a larger compound library.

> **Data heterogeneity**: The dataset includes both steroidal saponins (n = 16) and phenolic antioxidants (n = 75) with potentially different mechanisms. While this increases sample size, it may introduce variability. The consistent finding that aglycone cores contribute more than glycosylated fragments across both compound classes supports the robustness of this conclusion.

---

## 6. 图表引用修改

### Figure 2引用
**原文**:
> Figure 2 compares FragMoE against baseline methods using coefficient of determination (R²) across three antioxidant assays.

**修改为**:
> Figure 2 compares FragMoE against baseline methods using coefficient of determination (R²) across DPPH and ABTS assays (FRAP results provided as exploratory analysis in Supplementary Table S2).

### Figure 3引用
**原文**:
> Figure 3: Structure-activity contribution (SAC) heatmap from FragMoE interpretability analysis.

**修改为**:
> Figure 3: Structure-activity contribution heatmap from FragMoE interpretability analysis, showing consistent fragment contribution patterns across DPPH and ABTS assays.

---

## 7. 一致性声明修改

### Results (3.2节)
**原文**:
> The most significant finding from FragMoE interpretability analysis is the marked difference in antioxidant contribution between aglycone cores and glycosylated fragments (Figure 3).

**修改为**:
> The most significant finding from FragMoE interpretability analysis is the marked difference in antioxidant contribution between aglycone cores and glycosylated fragments, consistently observed across both DPPH and ABTS assays (Figure 3 and Figure S1).

**添加**:
> Cross-assay analysis (Figure S1) confirmed that aglycone cores contributed 3-4 times more than glycosylated fragments in both DPPH and ABTS assays, supporting the robustness of this structure-activity relationship.

---

## 修改检查清单

- [ ] Abstract: 术语修改 (MoE → MKE)
- [ ] Abstract: 性能声称修改
- [ ] Abstract: FRAP标注为exploratory
- [ ] Introduction: 术语修改
- [ ] Methods 2.2.2: 标题和内容修改
- [ ] Results 3.1: 性能声称修改
- [ ] Results 3.1: 添加统计显著性说明
- [ ] Results 3.2: 添加一致性声明
- [ ] Discussion: 添加局限性讨论
- [ ] Figure legends: 更新图表引用
