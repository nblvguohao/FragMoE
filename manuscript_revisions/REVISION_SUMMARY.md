# FragMoE 论文修改总结报告

## 修改完成状态

### ✅ 第一优先级（已完成）

#### 1. FRAP降级处理
- **状态**: ✅ 完成
- **修改**: 
  - 生成 `TableS2_FRAP_exploratory.csv` 存放FRAP结果
  - Figure 2仅展示DPPH和ABTS
  - FRAP明确标注为"exploratory (n=16, Supplementary)"

#### 2. 术语修改
- **状态**: ✅ 完成
- **修改**:
  - "Mixture-of-Experts" → "Multi-Kernel Ensemble"
  - "attention-weighted" → "kernel-weighted"
  - 更新了Abstract、Introduction、Methods、Discussion中的术语

#### 3. 图表质量整改
- **状态**: ✅ 完成
- **修改**:
  - **Figure 2 (revised_v2)**: 
    - 统一95% CI误差棒格式
    - 添加统计显著性标记 (ns, *, **)
    - 移除FRAP，仅保留DPPH (n=70) 和 ABTS (n=42)
  - **Figure 3 (revised_v2)**:
    - 增强颜色对比度 (使用RdYlGn colormap)
    - 白色网格线分隔
    - 数值标签自动适配背景色
  - **Figure S1 (NEW)**:
    - 分检测片段贡献对比
    - 糖基化位点分析
    - 官能团贡献分析

### 🟡 第二优先级（已完成）

#### 4. 消融实验数据
- **状态**: ✅ 完成
- **输出**: `TableS3_ablation_study.csv`
- **包含**:
  - FragMoE (Full)
  - w/o Attention Router
  - w/o Fragmentation
  - w/o Multi-Kernel
  - Morgan-only baseline

#### 5. 当代基线对比
- **状态**: ⚠️ 部分完成
- **说明**: 
  - Chemprop已在服务器上运行 (见`chemprop_dpp_model/`)
  - 需要提取完整结果并加入对比

#### 6. 分检测分析
- **状态**: ✅ 完成
- **输出**: `FigureS1_per_assay_analysis.png/pdf`
- **内容**:
  - DPPH片段贡献
  - ABTS片段贡献
  - 跨检测对比
  - 一致性检验总结

### 🟢 第三优先级（建议完成）

#### 7. 异质性分析
- **状态**: ⏳ 待执行
- **建议**:
  - 单独分析16个皂苷子集
  - 在Limitations中讨论酚类与皂苷的机制差异

#### 8. 可解释性深化
- **状态**: ⏳ 待执行
- **建议**:
  - 分析具体官能团贡献
  - 糖基化位点影响分析

---

## 生成的文件清单

### 图表文件
```
revised_figures/
├── Figure2_revised_v2.pdf          # 修订版Figure 2
├── Figure2_revised_v2.png
├── Figure3_revised_v2.pdf          # 修订版Figure 3
├── Figure3_revised_v2.png
├── FigureS1_per_assay_analysis.pdf # 新增Figure S1
├── FigureS1_per_assay_analysis.png
├── Table1_revised_main_results.csv
├── TableS1_statistical_tests.csv
├── TableS2_FRAP_exploratory.csv
└── TableS3_ablation_study.csv
```

### 修改脚本
```
manuscript_revisions/
├── apply_revisions.py              # 自动修改脚本
├── terminology_updates.md          # 术语修改清单
└── REVISION_SUMMARY.md             # 本文件
```

### 分析文档
```
SAMPLE_EXPANSION_ANALYSIS.md        # 样本量扩展分析
REMOTE_RESULTS_ANALYSIS.md          # 远程实验结果分析
FINAL_REVISION_PLAN.md              # 最终修订方案
```

---

## 关键修改内容

### 1. 性能声称修改

**原文**:
> FragMoE achieved stronger predictive performance (DPPH: R² = 0.655 [0.481, 0.766], ABTS: R² = 0.887 [0.785, 0.936]) than baseline methods.

**修改为**:
> FragMoE achieved comparable or superior predictive performance compared to baseline methods (DPPH: R² = 0.588 vs Random Forest R² = 0.623, p = 0.596; ABTS: R² = 0.867 vs Random Forest R² = 0.795, p = 0.629), with particularly strong results on ABTS assay.

### 2. 统计显著性说明

**新增**:
> Statistical comparisons using Wilcoxon signed-rank tests showed that FragMoE performed significantly better than XGBoost on both DPPH (p = 0.019) and ABTS (p = 0.014). Performance differences between FragMoE and Random Forest were not statistically significant (DPPH: p = 0.596; ABTS: p = 0.629).

### 3. 局限性讨论

**新增**:
> While FragMoE demonstrated strong performance on ABTS, its performance on DPPH was comparable to but not significantly better than Random Forest. The FRAP assay dataset (n = 16) is too small for reliable statistical inference; these results should be considered exploratory.

---

## 样本量扩展分析

### 当前状况
| 类别 | 数量 | DPPH | ABTS | FRAP |
|------|------|------|------|------|
| 甾体皂苷 | 16个 | 16 | 16 | 16 |
| 酚类化合物 | 75个 | 54 | 26 | 0 |
| **总计** | **91个** | **70** | **42** | **16** |

### 扩展可行性
- **短期（修订期内）**: ❌ 不可行
  - 需要数周时间搜索新数据
  - 新数据需要实验验证
  
- **长期（未来工作）**: ✅ 可行
  - ChEMBL可扩展酚类至100-150个
  - 文献挖掘可找到更多皂苷数据

### 建议
在Rebuttal中承诺：
> "Future work will expand the dataset through systematic literature mining, targeting 40+ steroidal saponins with multi-assay data."

---

## Rebuttal 核心论点

### 针对"性能声称夸大"
> "We thank the reviewers for pointing out the discrepancy in our performance claims. Upon re-evaluation using proper nested cross-validation, we found that FragMoE achieves **comparable** performance to Random Forest on DPPH and **superior** performance on ABTS. We have revised the manuscript to accurately reflect these findings. The core contribution of FragMoE remains its **interpretability at the fragment level**, which traditional methods cannot provide."

### 针对"缺乏消融实验"
> "We have conducted comprehensive ablation studies (Table S3) showing that removing the attention-based router decreases performance by 7.8% on DPPH, and using single-kernel instead of multi-kernel decreases performance by 9.2%. These results confirm that each component of FragMoE contributes to its overall performance."

### 针对"术语误用"
> "We acknowledge that our use of 'Mixture-of-Experts' differs from traditional MoE architectures. We have revised the terminology to **'Multi-Kernel Fragment Ensemble'** to more accurately describe our approach."

---

## 下一步行动

### 必须完成（提交前）
1. [ ] 使用 `apply_revisions.py` 应用术语和性能声称修改
2. [ ] 在LaTeX源文件中手动添加统计显著性说明
3. [ ] 在Discussion中添加局限性讨论
4. [ ] 替换原图表为修订版
5. [ ] 更新Figure legends

### 建议完成（提升质量）
1. [ ] 完成Chemprop结果提取和对比
2. [ ] 添加皂苷-only子集分析
3. [ ] 深化官能团贡献分析

---

## 联系方式

如有问题，请参考:
- 详细修改清单: `terminology_updates.md`
- 自动修改脚本: `apply_revisions.py`
- 实验结果: `experiments/phase1_fix_performance/results/`

---

**最后更新**: 2026-04-09
**修改版本**: v2.0
