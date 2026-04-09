"""
消融实验 (Ablation Study)

验证各组件的必要性：
1. 移除自注意力/Router机制 → 平均池化
2. 单核SVR vs 多核融合
3. 移除片段分解 → 直接使用分子级GNN
4. 不同片段化方法对比

输出：Table S3 - 消融实验结果
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, BRICS, Recap
from rdkit.Chem.Scaffolds import MurckoScaffold

# 导入评估框架
sys.path.append(str(Path(__file__).parent.parent / "phase1_fix_performance"))
from nested_cv_evaluation import (
    compute_morgan_fingerprints, compute_mordred_descriptors,
    mol_to_pyg_data, decompose_fragments, atom_features,
    train_random_forest, train_xgboost, train_svr
)

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data/01_dataset/antioxidant_dataset.csv"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)


# ==========================================
# 消融实验模型变体
# ==========================================

class GINEncoder(nn.Module):
    """GIN编码器"""
    def __init__(self, in_dim=9, hidden_dim=128, out_dim=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Linear(dims[i+1], dims[i+1]),
            )
            from torch_geometric.nn import GINConv
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class FragMoE_NoRouter(nn.Module):
    """消融1: 移除Router机制，使用平均池化"""
    def __init__(self, d_frag=128, dropout=0.3):
        super().__init__()
        self.gin = GINEncoder(in_dim=9, hidden_dim=d_frag, out_dim=d_frag, dropout=dropout)

        # 无Router，直接平均池化
        self.predictor = nn.Sequential(
            nn.Linear(d_frag, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, frag_batch, mol_idx):
        frag_emb = self.gin(frag_batch.x, frag_batch.edge_index, frag_batch.batch)

        # 平均池化聚合
        n_mols = mol_idx.max().item() + 1
        mol_emb = torch.zeros(n_mols, frag_emb.shape[1], device=frag_emb.device)

        for i in range(n_mols):
            mask = (mol_idx == i)
            if mask.sum() > 0:
                mol_emb[i] = frag_emb[mask].mean(dim=0)

        return self.predictor(mol_emb).squeeze()


class FragMoE_NoFragmentation(nn.Module):
    """消融2: 移除片段分解，直接使用分子级GNN"""
    def __init__(self, d_mol=128, dropout=0.3):
        super().__init__()
        self.gin = GINEncoder(in_dim=9, hidden_dim=d_mol, out_dim=d_mol, dropout=dropout)

        self.predictor = nn.Sequential(
            nn.Linear(d_mol, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, mol_batch):
        mol_emb = self.gin(mol_batch.x, mol_batch.edge_index, mol_batch.batch)
        return self.predictor(mol_emb).squeeze()


class FragMoE_SingleExpert(nn.Module):
    """消融3: 单Expert（相当于标准MLP）"""
    def __init__(self, d_frag=128, dropout=0.3):
        super().__init__()
        self.gin = GINEncoder(in_dim=9, hidden_dim=d_frag, out_dim=d_frag, dropout=dropout)

        self.expert = nn.Sequential(
            nn.Linear(d_frag, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, frag_batch, mol_idx):
        frag_emb = self.gin(frag_batch.x, frag_batch.edge_index, frag_batch.batch)

        n_mols = mol_idx.max().item() + 1
        mol_emb = torch.zeros(n_mols, frag_emb.shape[1], device=frag_emb.device)

        for i in range(n_mols):
            mask = (mol_idx == i)
            if mask.sum() > 0:
                mol_emb[i] = frag_emb[mask].mean(dim=0)

        return self.expert(mol_emb).squeeze()


class FragMoE_Full(nn.Module):
    """完整版FragMoE（带Router和多个Experts）"""
    def __init__(self, d_frag=128, n_experts=4, dropout=0.3):
        super().__init__()
        self.gin = GINEncoder(in_dim=9, hidden_dim=d_frag, out_dim=d_frag, dropout=dropout)

        # Router
        self.router = nn.Sequential(
            nn.Linear(d_frag, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
        )

        # Multiple Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_frag, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ) for _ in range(n_experts)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(n_experts, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, frag_batch, mol_idx):
        frag_emb = self.gin(frag_batch.x, frag_batch.edge_index, frag_batch.batch)

        # Router权重
        router_logits = self.router(frag_emb)
        weights = F.softmax(router_logits, dim=1).unsqueeze(-1)

        # Expert预测
        expert_preds = torch.stack([expert(frag_emb) for expert in self.experts], dim=1)
        weighted_preds = (weights * expert_preds).sum(dim=1)

        # 分子级聚合
        n_mols = mol_idx.max().item() + 1
        mol_preds = torch.zeros(n_mols, 1, device=frag_emb.device)

        for i in range(n_mols):
            mask = (mol_idx == i)
            if mask.sum() > 0:
                mol_preds[i] = weighted_preds[mask].mean()

        return mol_preds.squeeze()


def decompose_murcko_only(smiles):
    """仅使用Murcko骨架"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    fragments = []

    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf and scaf.GetNumAtoms() > 0:
            frag_data = mol_to_pyg_data(Chem.MolToSmiles(scaf))
            if frag_data is not None:
                fragments.append(frag_data)
    except Exception:
        pass

    # Fallback到完整分子
    if not fragments:
        whole = mol_to_pyg_data(smiles)
        if whole is not None:
            fragments.append(whole)

    return fragments


def decompose_brics_only(smiles):
    """仅使用BRICS分解"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    fragments = []

    try:
        brics_frags = list(BRICS.BRICSDecompose(mol))
        for frag_smi in brics_frags:
            clean_mol = Chem.MolFromSmiles(frag_smi)
            if clean_mol and clean_mol.GetNumAtoms() >= 3:
                frag_data = mol_to_pyg_data(frag_smi)
                if frag_data is not None:
                    fragments.append(frag_data)
    except Exception:
        pass

    # Fallback
    if not fragments:
        whole = mol_to_pyg_data(smiles)
        if whole is not None:
            fragments.append(whole)

    return fragments


# ==========================================
# 训练函数
# ==========================================

def train_gnn_model(model_class, train_data, test_data, n_epochs=150, device='cpu', **model_kwargs):
    """通用GNN训练函数"""
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 准备训练数据
    train_fragments = []
    train_mol_idx = []
    train_labels = []

    for i, (smiles, label) in enumerate(zip(train_data['SMILES'], train_data['pIC50'])):
        frags = decompose_fragments(smiles)
        if frags:
            train_fragments.extend(frags)
            train_mol_idx.extend([i] * len(frags))
            train_labels.append(label)

    if not train_fragments:
        return None

    train_batch = Batch.from_data_list(train_fragments)
    train_mol_idx = torch.tensor(train_mol_idx, dtype=torch.long)
    train_y = torch.tensor(train_labels, dtype=torch.float)

    # 训练
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if model_class == FragMoE_NoFragmentation:
            # 直接使用分子图
            mol_data_list = [mol_to_pyg_data(s) for s in train_data['SMILES']]
            mol_data_list = [d for d in mol_data_list if d is not None]
            if mol_data_list:
                mol_batch = Batch.from_data_list(mol_data_list).to(device)
                pred = model(mol_batch)
                loss = criterion(pred, train_y[:len(pred)].to(device))
                loss.backward()
                optimizer.step()
        else:
            pred = model(train_batch.to(device), train_mol_idx.to(device))
            loss = criterion(pred, train_y.to(device))
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()

    test_fragments = []
    test_mol_idx = []
    test_labels = []

    for i, (smiles, label) in enumerate(zip(test_data['SMILES'], test_data['pIC50'])):
        frags = decompose_fragments(smiles)
        if frags:
            test_fragments.extend(frags)
            test_mol_idx.extend([i] * len(frags))
            test_labels.append(label)

    if not test_fragments:
        return None

    test_batch = Batch.from_data_list(test_fragments)
    test_mol_idx = torch.tensor(test_mol_idx, dtype=torch.long)
    test_y = torch.tensor(test_labels, dtype=torch.float)

    with torch.no_grad():
        if model_class == FragMoE_NoFragmentation:
            mol_data_list = [mol_to_pyg_data(s) for s in test_data['SMILES']]
            mol_data_list = [d for d in mol_data_list if d is not None]
            if mol_data_list:
                mol_batch = Batch.from_data_list(mol_data_list).to(device)
                pred = model(mol_batch)
                pred_np = pred.cpu().numpy()
                y_np = test_y[:len(pred_np)].numpy()
            else:
                return None
        else:
            pred = model(test_batch.to(device), test_mol_idx.to(device))
            pred_np = pred.cpu().numpy()
            y_np = test_y.numpy()

    return {
        'R2': r2_score(y_np, pred_np),
        'RMSE': np.sqrt(mean_squared_error(y_np, pred_np)),
        'MAE': mean_absolute_error(y_np, pred_np),
    }


# ==========================================
# 消融实验主流程
# ==========================================

def run_ablation_study(assay_name='DPPH', n_splits=5):
    """运行完整的消融实验"""
    print(f"\n{'='*60}")
    print(f"消融实验 - {assay_name}")
    print(f"{'='*60}")

    # 加载数据
    df = pd.read_csv(DATA_PATH)
    assay_data = df[df['assay_type'] == assay_name].copy()

    print(f"总样本数: {len(assay_data)}")

    # 分层
    assay_data['molwt'] = assay_data['SMILES'].apply(
        lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else 0
    )
    assay_data['mw_bin'] = pd.qcut(assay_data['molwt'], q=n_splits, labels=False, duplicates='drop')

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储所有结果
    all_results = {
        'FragMoE_Full': [],
        'FragMoE_NoRouter': [],
        'FragMoE_NoFragmentation': [],
        'FragMoE_SingleExpert': [],
        'RandomForest': [],
        'XGBoost': [],
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(assay_data, assay_data['mw_bin'])):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        train_data = assay_data.iloc[train_idx]
        test_data = assay_data.iloc[test_idx]

        train_dict = {
            'SMILES': train_data['SMILES'].tolist(),
            'pIC50': train_data['pIC50'].values
        }
        test_dict = {
            'SMILES': test_data['SMILES'].tolist(),
            'pIC50': test_data['pIC50'].values
        }

        # 1. 完整FragMoE
        print("  Testing FragMoE (Full)...")
        result = train_gnn_model(FragMoE_Full, train_dict, test_dict, n_epochs=150, device=device,
                                 d_frag=128, n_experts=4)
        if result:
            all_results['FragMoE_Full'].append(result)
            print(f"    R² = {result['R2']:.3f}")

        # 2. 无Router
        print("  Testing FragMoE (No Router)...")
        result = train_gnn_model(FragMoE_NoRouter, train_dict, test_dict, n_epochs=150, device=device,
                                 d_frag=128)
        if result:
            all_results['FragMoE_NoRouter'].append(result)
            print(f"    R² = {result['R2']:.3f}")

        # 3. 无片段分解
        print("  Testing FragMoE (No Fragmentation)...")
        result = train_gnn_model(FragMoE_NoFragmentation, train_dict, test_dict, n_epochs=150, device=device,
                                 d_mol=128)
        if result:
            all_results['FragMoE_NoFragmentation'].append(result)
            print(f"    R² = {result['R2']:.3f}")

        # 4. 单Expert
        print("  Testing FragMoE (Single Expert)...")
        result = train_gnn_model(FragMoE_SingleExpert, train_dict, test_dict, n_epochs=150, device=device,
                                 d_frag=128)
        if result:
            all_results['FragMoE_SingleExpert'].append(result)
            print(f"    R² = {result['R2']:.3f}")

        # 5. Random Forest基线
        print("  Testing Random Forest...")
        X_train_morgan = np.array([compute_morgan_fingerprints(s) for s in train_data['SMILES']])
        X_test_morgan = np.array([compute_morgan_fingerprints(s) for s in test_data['SMILES']])
        X_train_mordred = np.array([compute_mordred_descriptors(s) for s in train_data['SMILES']])
        X_test_mordred = np.array([compute_mordred_descriptors(s) for s in test_data['SMILES']])
        X_train = np.hstack([X_train_morgan, X_train_mordred])
        X_test = np.hstack([X_test_morgan, X_test_mordred])
        y_train = train_data['pIC50'].values
        y_test = test_data['pIC50'].values

        result = train_random_forest(X_train, y_train, X_test, y_test)
        all_results['RandomForest'].append(result)
        print(f"    R² = {result['R2']:.3f}")

        # 6. XGBoost基线
        print("  Testing XGBoost...")
        result = train_xgboost(X_train, y_train, X_test, y_test)
        all_results['XGBoost'].append(result)
        print(f"    R² = {result['R2']:.3f}")

    # 汇总结果
    print(f"\n{'='*60}")
    print("消融实验汇总结果")
    print(f"{'='*60}")

    summary = []
    for model_name, model_results in all_results.items():
        if model_results:
            r2_scores = [r['R2'] for r in model_results]
            rmse_scores = [r['RMSE'] for r in model_results]
            mae_scores = [r['MAE'] for r in model_results]

            summary.append({
                'Model': model_name,
                'R2_mean': np.mean(r2_scores),
                'R2_std': np.std(r2_scores),
                'RMSE_mean': np.mean(rmse_scores),
                'MAE_mean': np.mean(mae_scores),
            })

            print(f"\n{model_name}:")
            print(f"  R² = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
            print(f"  RMSE = {np.mean(rmse_scores):.3f}")

    # 保存结果
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_PATH / f"ablation_results_{assay_name}.csv", index=False)
    print(f"\n结果已保存至: {RESULTS_PATH / f'ablation_results_{assay_name}.csv'}")

    return summary_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--assay', type=str, default='DPPH', choices=['DPPH', 'ABTS', 'FRAP'])
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    results = run_ablation_study(args.assay, args.folds)
