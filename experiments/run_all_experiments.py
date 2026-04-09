#!/usr/bin/env python3
"""
运行所有修订实验

执行顺序：
1. Phase 1: 修复性能评估（嵌套CV）
2. Phase 2: 消融实验
3. Phase 3: 预训练模型基线
4. 生成修订图表

使用方法：
    python run_all_experiments.py --assays DPPH ABTS --skip-pretrained

注意：预训练模型实验需要下载大型模型，可能需要较长时间
"""

import sys
from pathlib import Path
import subprocess
import argparse
import time

# 添加路径
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "phase1_fix_performance"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "phase2_ablation"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "phase3_pretrained"))

def run_phase1(assays, n_folds=5):
    """运行Phase 1: 嵌套CV评估"""
    print("\n" + "="*70)
    print("PHASE 1: 嵌套交叉验证评估")
    print("="*70)

    from phase1_fix_performance.nested_cv_evaluation import nested_cv_evaluation

    results = {}
    for assay in assays:
        try:
            print(f"\n>>> 评估 {assay} ...")
            start = time.time()
            result = nested_cv_evaluation(assay, n_folds)
            results[assay] = result
            elapsed = time.time() - start
            print(f"    完成 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    失败: {e}")
            results[assay] = None

    return results


def run_phase2(assays, n_folds=5):
    """运行Phase 2: 消融实验"""
    print("\n" + "="*70)
    print("PHASE 2: 消融实验")
    print("="*70)

    from phase2_ablation.ablation_study import run_ablation_study

    results = {}
    for assay in assays:
        try:
            print(f"\n>>> 消融实验 {assay} ...")
            start = time.time()
            result = run_ablation_study(assay, n_folds)
            results[assay] = result
            elapsed = time.time() - start
            print(f"    完成 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    失败: {e}")
            results[assay] = None

    return results


def run_phase3(assays, n_folds=5):
    """运行Phase 3: 预训练模型基线"""
    print("\n" + "="*70)
    print("PHASE 3: 预训练模型基线对比")
    print("="*70)
    print("注意：此阶段需要下载预训练模型，可能需要较长时间")

    from phase3_pretrained.pretrained_baselines import evaluate_pretrained_baselines

    results = {}
    for assay in assays:
        try:
            print(f"\n>>> 预训练基线 {assay} ...")
            start = time.time()
            result = evaluate_pretrained_baselines(assay, n_folds)
            results[assay] = result
            elapsed = time.time() - start
            print(f"    完成 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    失败: {e}")
            results[assay] = None

    return results


def generate_figures():
    """生成修订图表"""
    print("\n" + "="*70)
    print("生成修订图表")
    print("="*70)

    try:
        import revised_figures.generate_revised_figures as grf
        print(">>> 生成图表...")
        grf.generate_figure2_revised()
        grf.generate_figure_s1_ablation()
        grf.generate_figure_s2_pretrained()
        grf.generate_figure_s3_fragment_contribution()
        grf.generate_table_summary()
        print("    完成")
    except Exception as e:
        print(f"    失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='运行FragMoE修订实验')
    parser.add_argument('--assays', nargs='+', default=['DPPH', 'ABTS'],
                       choices=['DPPH', 'ABTS', 'FRAP'],
                       help='要评估的检测类型')
    parser.add_argument('--folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='跳过Phase 1')
    parser.add_argument('--skip-phase2', action='store_true',
                       help='跳过Phase 2')
    parser.add_argument('--skip-phase3', action='store_true',
                       help='跳过Phase 3（预训练模型）')
    parser.add_argument('--skip-figures', action='store_true',
                       help='跳过图表生成')
    parser.add_argument('--figures-only', action='store_true',
                       help='仅生成图表（使用已有结果）')

    args = parser.parse_args()

    start_time = time.time()

    if args.figures_only:
        generate_figures()
    else:
        if not args.skip_phase1:
            run_phase1(args.assays, args.folds)

        if not args.skip_phase2:
            run_phase2(args.assays, args.folds)

        if not args.skip_phase3:
            run_phase3(args.assays, args.folds)

        if not args.skip_figures:
            generate_figures()

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"所有实验完成！总耗时: {elapsed/60:.1f} 分钟")
    print("="*70)


if __name__ == '__main__':
    main()
