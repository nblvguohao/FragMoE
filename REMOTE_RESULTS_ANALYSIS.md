# 远程服务器实验结果分析

## 数据来源
- **服务器**: user@100.112.165.109
- **目录**: ~/FragMoE/
- **同步时间**: 2026-04-08

---

## 核心发现

### 1. 模型性能对比

#### DPPH Assay (n=70)
| Model | R² | RMSE | MAE | vs FragMoE |
|-------|-----|------|-----|-----------|
| **FragMoE v5** | **0.588** | 0.230 | 0.176 | - |
| Random Forest | 0.623 | 0.220 | 0.174 | p=0.596 (ns) |
| XGBoost | 0.490 | 0.256 | 0.201 | p=0.019 (*) |
| SVR-RBF | 0.318 | 0.296 | 0.239 | p=0.095 (ns) |
| PLS | 0.512 | 0.250 | 0.196 | p=0.018 (*) |

**结论**: FragMoE与Random Forest性能相当（无显著差异），但显著优于XGBoost和PLS

#### ABTS Assay (n=42)
| Model | R² | RMSE | MAE | vs FragMoE |
|-------|-----|------|-----|-----------|
| **FragMoE v5** | **0.867** | 0.195 | 0.147 | - |
| Random Forest | 0.795 | 0.242 | 0.195 | p=0.629 (ns) |
| XGBoost | 0.715 | 0.286 | 0.220 | p=0.014 (*) |
| SVR-RBF | 0.253 | 0.463 | 0.326 | p<0.001 (***) |
| PLS | 0.725 | 0.281 | 0.224 | p=0.072 (ns) |

**结论**: FragMoE数值上优于所有基线，但与RF无统计显著差异

#### FRAP Assay (n=16)
| Model | R² | RMSE | MAE | vs FragMoE |
|-------|-----|------|-----|-----------|
| **FragMoE v5** | **0.789** | 0.137 | 0.116 | - |
| Random Forest | 0.727 | 0.155 | 0.122 | p=0.003 (**) |
| XGBoost | 0.075 | 0.286 | 0.204 | p=0.642 (ns) |
| SVR-RBF | 0.687 | 0.166 | 0.136 | p=0.255 (ns) |
| PLS | 0.690 | 0.165 | 0.125 | p=0.063 (ns) |

**结论**: FragMoE显著优于Random Forest，但注意样本量仅n=16

---

## 关键洞察

### 1. 性能一致性
- **ABTS**: FragMoE表现最佳（R²=0.867）
- **FRAP**: FragMoE显著优于RF（R²=0.789 vs 0.727）
- **DPPH**: FragMoE与RF相当（R²=0.588 vs 0.623），差异不显著

### 2. 统计显著性总结
- FragMoE在FRAP上显著优于RF (p=0.003)
- FragMoE在DPPH上显著优于XGBoost (p=0.019) 和 PLS (p=0.018)
- FragMoE在ABTS上显著优于XGBoost (p=0.014) 和 SVR (p<0.001)

### 3. 消融实验结果 (benchmark_v5_ablation.csv)
多核融合消融分析显示：
- 移除Random Forest内核：DPPH R² = 0.563 (下降 0.013)
- 移除XGBoost内核：DPPH R² = 0.575 (下降 -0.065，即提升)
- 移除PLS内核：DPPH R² = 0.578 (下降 -0.222，即大幅提升)

这表明某些基线模型可能对融合产生负面影响。

---

## 与论文声称的对比

### 论文声称
> "FragMoE achieved stronger predictive performance (DPPH: R² = 0.655 [0.481, 0.766], ABTS: R² = 0.887 [0.785, 0.936]) than baseline methods"

### 实际结果
- DPPH: R² = 0.588 (低于声称的0.655，且低于RF的0.623)
- ABTS: R² = 0.867 (接近声称的0.887)

### 问题分析
1. **DPPH性能被夸大**: 实际R²=0.588 vs 声称0.655，差距约0.067
2. **与RF比较**: DPPH上FragMoE并未显著优于RF (p=0.596)
3. **声称过于绝对**: 应修改为"FragMoE achieves comparable or superior performance"

---

## 论文修改建议

### 1. 修改性能声称
**原句**:
> FragMoE achieved stronger predictive performance than baseline methods

**修改建议**:
> FragMoE achieved comparable or superior predictive performance compared to baseline methods, with particularly strong results on ABTS (R² = 0.867) and FRAP (R² = 0.789) assays. On DPPH, FragMoE performed comparably to Random Forest (R² = 0.588 vs 0.623, p = 0.596).

### 2. 添加统计显著性说明
在Results部分添加:
> Statistical comparisons using Wilcoxon signed-rank tests showed that FragMoE performed significantly better than XGBoost and PLS on DPPH (p < 0.05), and significantly better than Random Forest on FRAP (p = 0.003). Performance differences between FragMoE and Random Forest on DPPH and ABTS were not statistically significant.

### 3. 坦诚讨论DPPH结果
在Discussion中添加:
> While FragMoE demonstrated strong performance on ABTS and FRAP assays, its performance on DPPH was comparable to but not significantly better than Random Forest. This suggests that the advantages of fragment-based interpretability may be more pronounced for certain types of antioxidant mechanisms (e.g., ABTS radical scavenging) than others (DPPH radical scavenging).

---

## 文件清单

已同步到本地的结果文件:
- `experiments/phase1_fix_performance/results/benchmark_v4_results.csv` - v4基线对比
- `experiments/phase1_fix_performance/results/benchmark_v5_ensemble.csv` - FragMoE v5性能
- `experiments/phase1_fix_performance/results/benchmark_v5_statistical_tests.csv` - 统计检验
- `experiments/phase1_fix_performance/results/benchmark_v5_ablation.csv` - 消融实验
- `experiments/phase1_fix_performance/results/benchmark_v5_mordred.csv` - Mordred特征对比

---

## 下一步行动

1. **更新Table 1**: 使用benchmark_v5_ensemble.csv和benchmark_v4_results.csv中的数据
2. **添加统计检验**: 将benchmark_v5_statistical_tests.csv的结果加入正文
3. **修改声称**: 将"superior performance"改为"comparable or superior"
4. **FRAP处理**: 考虑将FRAP降级为exploratory分析（n=16）

---

## 补充说明

### Chemprop状态
远程服务器上已运行Chemprop (D-MPNN) 基准测试:
- `chemprop_DPPH_data.csv` - DPPH数据
- `chemprop_ABTS_data.csv` - ABTS数据
- `chemprop_FRAP_data.csv` - FRAP数据
- `chemprop_dpp_model/` - 训练好的模型

需要检查Chemprop的实际性能结果是否已保存到某个文件中。
