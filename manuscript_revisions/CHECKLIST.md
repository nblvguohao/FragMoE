# 论文修改执行检查清单

## 样本量扩展分析结论

### 当前数据集组成
| 类别 | 数量 | DPPH | ABTS | FRAP |
|------|------|------|------|------|
| 甾体皂苷 | 16个化合物 | 16条 | 16条 | 16条 |
| 酚类化合物 | 75个化合物 | 54条 | 26条 | 0条 |
| **总计** | **91个化合物** | **70条** | **42条** | **16条** |

### 扩展可行性
- **短期内（修订期）**: ❌ 不可行 - 需要数周时间搜索和验证新数据
- **长期（未来工作）**: ✅ 可行 - 可从ChEMBL/PubChem扩展

**建议**: 在Rebuttal中承诺未来扩展，而非在修订期内强行增加

---

## 修改完成状态

### ✅ 已完成（第一优先级）

- [x] **FRAP降级处理**
  - [x] TableS2_FRAP_exploratory.csv 已生成
  - [x] Figure 2 仅展示DPPH和ABTS
  - [x] FRAP标注为"exploratory"

- [x] **术语修改**
  - [x] MoE → Multi-Kernel Ensemble
  - [x] attention-weighted → kernel-weighted
  - [x] apply_revisions.py 脚本已创建

- [x] **图表质量整改**
  - [x] Figure2_revised_v2 (95% CI误差棒，统计显著性)
  - [x] Figure3_revised_v2 (RdYlGn颜色映射)
  - [x] FigureS1_per_assay_analysis (分检测分析)

### ✅ 已完成（第二优先级）

- [x] **消融实验数据** - TableS3_ablation_study.csv
- [x] **分检测分析** - FigureS1
- [~] **当代基线** - Chemprop已运行，需提取结果

### ⏳ 待执行（第三优先级）

- [ ] 皂苷-only子集分析
- [ ] 官能团贡献深化分析

---

## 提交前必须完成的步骤

### 1. 应用术语修改
```bash
cd fragmoe_repo/manuscript_revisions
python apply_revisions.py --manuscript path/to/your/manuscript.tex --output revised_manuscript.tex
```

### 2. 手动添加关键段落
- [ ] Abstract: 性能声称修改
- [ ] Results 3.1后: 统计显著性说明
- [ ] Discussion: 局限性讨论

### 3. 替换图表
- [ ] Figure 2 → Figure2_revised_v2.pdf
- [ ] Figure 3 → Figure3_revised_v2.pdf
- [ ] 添加 Figure S1

### 4. 更新表格
- [ ] Table 1 → Table1_revised_main_results.csv
- [ ] 添加 Table S1 (统计检验)
- [ ] 添加 Table S2 (FRAP exploratory)
- [ ] 添加 Table S3 (消融实验)

### 5. 准备Rebuttal
- [ ] 针对"性能声称"的回复
- [ ] 针对"缺乏消融实验"的回复
- [ ] 针对"术语误用"的回复

---

## 文件位置速查

| 文件 | 路径 |
|------|------|
| 修订图表 | `revised_figures/` |
| 修改脚本 | `manuscript_revisions/apply_revisions.py` |
| 术语清单 | `manuscript_revisions/terminology_updates.md` |
| 实验结果 | `experiments/phase1_fix_performance/results/` |
| 修改总结 | `manuscript_revisions/REVISION_SUMMARY.md` |

---

## 预期审稿人关注点及回应

### 1. "性能声称夸大"
**回应**: 
> "We have revised the performance claims to accurately reflect the nested CV results. FragMoE achieves comparable performance to Random Forest on DPPH and superior performance on ABTS. The core contribution is interpretability, not marginal R² improvements."

### 2. "FRAP样本量问题"
**回应**:
> "We acknowledge the limitation of FRAP (n=16) and have moved these results to Supplementary as exploratory analysis. The main conclusions are based on DPPH (n=70) and ABTS (n=42)."

### 3. "MoE术语误用"
**回应**:
> "We have revised the terminology to 'Multi-Kernel Fragment Ensemble' to accurately describe our approach."

---

## 最终检查

- [ ] 所有图表分辨率 ≥ 300 DPI
- [ ] 所有表格格式正确
- [ ] 统计检验结果准确
- [ ] 术语一致性检查
- [ ] 拼写和语法检查

**预计完成时间**: 4-6小时
