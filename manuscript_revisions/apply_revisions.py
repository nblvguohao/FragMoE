"""
论文修改应用脚本

这个脚本帮助用户系统地应用所有必要的修改：
1. 术语修改 (MoE -> Multi-Kernel Ensemble)
2. 性能声称修改
3. FRAP降级处理
4. 统计显著性说明

使用方法:
    python apply_revisions.py --manuscript path/to/manuscript.tex --output path/to/revised.tex

或者直接查看修改建议:
    python apply_revisions.py --check-only
"""

import argparse
import re
from pathlib import Path
import sys

# 修改规则定义
REVISIONS = {
    "terminology": {
        "description": "术语修改 (MoE -> Multi-Kernel Ensemble)",
        "replacements": [
            (
                r"Fragment-based Mixture-of-Experts",
                "Fragment-based Multi-Kernel Ensemble",
            ),
            (r"Mixture-of-Experts framework", "multi-kernel ensemble framework"),
            (r"the MoE layer", "the multi-kernel ensemble layer"),
            (r"attention-weighted fragment", "kernel-weighted fragment"),
            (
                r"self-attention mechanism.*fragment-fragment interaction",
                "kernel weighting mechanism that learns optimal combination of molecular representations",
            ),
        ],
    },
    "performance_claims": {
        "description": "性能声称修改",
        "replacements": [
            (
                r"FragMoE achieved stronger predictive performance.*?than baseline methods",
                "FragMoE achieved comparable or superior predictive performance compared to baseline methods",
            ),
            (
                r"FragMoE outperformed baseline methods across all assays",
                "FragMoE demonstrated competitive or superior performance compared to baseline methods",
            ),
            (
                r"exceeding the performance of",
                "comparable to",
            ),
        ],
    },
    "frap_handling": {
        "description": "FRAP降级处理",
        "replacements": [
            (
                r"across three antioxidant assays",
                "across DPPH and ABTS assays (FRAP results provided as exploratory analysis)",
            ),
            (
                r"across three complementary assays: DPPH.*?ABTS.*?and FRAP",
                "across two primary assays (DPPH and ABTS); FRAP results (n = 16) are provided as exploratory analysis",
            ),
            (
                r"FRAP \(n = 16\)",
                "FRAP exploratory analysis (n = 16, Supplementary)",
            ),
        ],
    },
}

# 需要在特定位置添加的新段落
ADDITIONS = {
    "statistical_tests": {
        "after_pattern": r"Table 1[.]?",
        "content": """Statistical comparisons using Wilcoxon signed-rank tests showed that FragMoE performed significantly better than XGBoost on both DPPH (p = 0.019) and ABTS (p = 0.014), and significantly better than SVR-RBF on ABTS (p < 0.001). Performance differences between FragMoE and Random Forest were not statistically significant on either assay (DPPH: p = 0.596; ABTS: p = 0.629; Table S1).""",
    },
    "limitations": {
        "after_pattern": r"Limitations and Future Work",
        "content": """Several limitations should be acknowledged. First, while FragMoE demonstrated strong performance on ABTS, its performance on DPPH was comparable to but not significantly better than Random Forest, suggesting that the advantages of fragment-based modeling may vary by assay type. Second, the FRAP assay dataset (n = 16) is too small for reliable statistical inference; these results should be considered exploratory. Third, the dataset includes both steroidal saponins and phenolic antioxidants with potentially different mechanisms; future work will validate findings on larger, compound-class-specific datasets.""",
    },
}


def apply_revisions(text, revision_type):
    """应用特定类型的修改"""
    revisions = REVISIONS[revision_type]
    count = 0
    for pattern, replacement in revisions["replacements"]:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        if matches > 0:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            count += matches
    return text, count


def check_revisions(text):
    """检查需要修改的内容"""
    results = {}
    for rev_type, rev_data in REVISIONS.items():
        matches = 0
        for pattern, _ in rev_data["replacements"]:
            matches += len(re.findall(pattern, text, re.IGNORECASE))
        results[rev_type] = {
            "description": rev_data["description"],
            "matches": matches,
        }
    return results


def generate_report(original_text, revised_text):
    """生成修改报告"""
    report = ["=" * 70, "论文修改报告", "=" * 70, ""]

    # 统计修改
    for rev_type, rev_data in REVISIONS.items():
        original_count = 0
        for pattern, _ in rev_data["replacements"]:
            original_count += len(re.findall(pattern, original_text, re.IGNORECASE))

        report.append(f"\n【{rev_data['description']}】")
        report.append(f"  发现匹配: {original_count} 处")

    report.extend(
        [
            "",
            "=" * 70,
            "修改建议",
            "=" * 70,
            "",
            "1. 必须修改（影响审稿结果）:",
            "   - 术语: MoE -> Multi-Kernel Ensemble",
            "   - 性能声称: 'superior' -> 'comparable or superior'",
            "   - FRAP: 降级为exploratory analysis",
            "",
            "2. 强烈建议修改:",
            "   - 添加统计显著性说明",
            "   - 添加局限性讨论",
            "   - 更新Figure引用",
            "",
            "3. 图表修改:",
            "   - Figure 2: 仅保留DPPH和ABTS",
            "   - Figure 3: 增强颜色对比度",
            "   - 添加Figure S1: 分检测分析",
            "",
            "=" * 70,
        ]
    )

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="论文修改应用脚本")
    parser.add_argument(
        "--manuscript", type=str, help="原稿文件路径 (.tex or .docx)"
    )
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument(
        "--check-only", action="store_true", help="仅检查，不应用修改"
    )
    parser.add_argument(
        "--revision-type",
        choices=["all", "terminology", "performance_claims", "frap_handling"],
        default="all",
        help="修改类型",
    )

    args = parser.parse_args()

    # 如果没有提供文件，显示修改指南
    if not args.manuscript:
        print("=" * 70)
        print("FragMoE 论文修改指南")
        print("=" * 70)
        print()
        print("使用说明:")
        print("1. 查看修改建议: python apply_revisions.py --check-only")
        print(
            "2. 应用所有修改: python apply_revisions.py --manuscript manuscript.tex --output revised.tex"
        )
        print()
        print("=" * 70)
        print("修改清单")
        print("=" * 70)
        print()
        for rev_type, rev_data in REVISIONS.items():
            print(f"【{rev_data['description']}】")
            for pattern, replacement in rev_data["replacements"][:3]:  # 显示前3个
                print(f"  {pattern[:50]}... -> {replacement[:40]}...")
            print()
        return

    # 读取文件
    manuscript_path = Path(args.manuscript)
    if not manuscript_path.exists():
        print(f"错误: 文件不存在 {manuscript_path}")
        sys.exit(1)

    with open(manuscript_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    # 仅检查模式
    if args.check_only:
        results = check_revisions(original_text)
        print("=" * 70)
        print("修改检查结果")
        print("=" * 70)
        for rev_type, data in results.items():
            print(f"\n【{data['description']}】")
            print(f"  需要修改: {data['matches']} 处")
        return

    # 应用修改
    revised_text = original_text
    total_changes = 0

    if args.revision_type in ["all", "terminology"]:
        revised_text, count = apply_revisions(revised_text, "terminology")
        total_changes += count
        print(f"术语修改: {count} 处")

    if args.revision_type in ["all", "performance_claims"]:
        revised_text, count = apply_revisions(revised_text, "performance_claims")
        total_changes += count
        print(f"性能声称修改: {count} 处")

    if args.revision_type in ["all", "frap_handling"]:
        revised_text, count = apply_revisions(revised_text, "frap_handling")
        total_changes += count
        print(f"FRAP处理修改: {count} 处")

    # 保存修改后的文件
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(revised_text)
        print(f"\n修改后的文件已保存: {output_path}")

    # 生成报告
    report = generate_report(original_text, revised_text)
    report_path = (
        Path(args.output).parent / "revision_report.txt"
        if args.output
        else Path("revision_report.txt")
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"修改报告已保存: {report_path}")

    print(f"\n总计修改: {total_changes} 处")


if __name__ == "__main__":
    main()
