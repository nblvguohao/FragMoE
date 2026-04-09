"""
嵌套交叉验证评估框架
解决原评估可能存在的问题：
1. 数据泄露：使用train_test_split可能导致信息泄露
2. 超参数调优：需要在验证集上选择，而非测试集
3. 性能评估：使用nested CV获得无偏估计
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import BRICS

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch

# Constants
ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data/01_dataset/antioxidant_dataset.csv"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)


def compute_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    """计算Morgan指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def compute_mordred_descriptors(smiles):
    """计算简化版分子描述符（类似Mordred的关键特征）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20)

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.MaxAbsPartialCharge(mol) if hasattr(Descriptors, 'MaxAbsPartialCharge') else 0,
        Descriptors.MinAbsPartialCharge(mol) if hasattr(Descriptors, 'MinAbsPartialCharge') else 0,
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
    ]
    return np.array(features)


def create_maccs_like_features(smiles):
    """创建类似MACCS的简化特征"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20)

    features = [
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]'))),  # 碳
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#8]'))),  # 氧
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#7]'))),  # 氮
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#16]'))), # 硫
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#9]'))),  # 氟
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#17]'))), # 氯
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#35]'))), # 溴
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#53]'))), # 碘
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('a'))),     # 芳香原子
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[OH]'))),  # 羟基
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[NH]'))),  # 氨基
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('C=O'))),   # 羰基
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('C=C'))),   # 双键
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('C#C'))),   # 三键
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[Ring1]'))),  # 单环
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[Ring2]'))),  # 双环
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[Ring3]'))),  # 三环
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]~[#7]~[#8]'))),  # C-N-O模式
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6](=[#8])-[#7]'))), # 酰胺
        int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]#[#7]'))),        # 腈
    ]
    return np.array(features)


def atom_features(atom):
    """原子特征提取"""
    from rdkit import Chem
    hyb = atom.GetHybridization()
    hyb_vec = [0, 0, 0]
    if hyb == Chem.rdchem.HybridizationType.SP:
        hyb_vec[0] = 1
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        hyb_vec[1] = 1
    elif hyb == Chem.rdchem.HybridizationType.SP3:
        hyb_vec[2] = 1

    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        float(atom.GetFormalCharge()),
        atom.GetTotalNumHs() / 4.0,
        *hyb_vec,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def mol_to_pyg_data(smiles):
    """将SMILES转换为PyG Data对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 原子特征
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    # 边索引
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=mol.GetNumAtoms())


def decompose_fragments(smiles):
    """BRICS片段分解"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    fragments = []

    # 添加整个分子
    whole = mol_to_pyg_data(smiles)
    if whole is not None:
        fragments.append(whole)

    # BRICS分解
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

    return fragments if fragments else [whole] if whole else []


# ==========================================
# 基线模型
# ==========================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """训练Random Forest（带超参数调优）"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }

    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    return {
        'model': 'RandomForest',
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'best_params': grid.best_params_
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """训练XGBoost（带超参数调优）"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1],
    }

    model = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    return {
        'model': 'XGBoost',
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'best_params': grid.best_params_
    }


def train_svr(X_train, y_train, X_test, y_test):
    """训练SVR（带超参数调优）"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'epsilon': [0.01, 0.1, 0.5],
    }

    model = SVR(kernel='rbf')
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    return {
        'model': 'SVR',
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'best_params': grid.best_params_
    }


# ==========================================
# FragMoE模型（改进版）
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
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class SimpleFragMoE(nn.Module):
    """简化但有效的FragMoE实现"""
    def __init__(self, d_frag=128, n_experts=4, dropout=0.3):
        super().__init__()

        # GIN编码器
        self.gin = GINEncoder(in_dim=9, hidden_dim=d_frag, out_dim=d_frag, dropout=dropout)

        # Router（简化版）
        self.router = nn.Sequential(
            nn.Linear(d_frag, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
        )

        # Experts
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

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(n_experts, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, frag_batch, mol_idx):
        # 编码片段
        frag_emb = self.gin(frag_batch.x, frag_batch.edge_index, frag_batch.batch)

        # Router输出权重
        router_logits = self.router(frag_emb)

        # 每个Expert的预测
        expert_preds = torch.stack([expert(frag_emb) for expert in self.experts], dim=1)

        # Softmax权重
        weights = F.softmax(router_logits, dim=1).unsqueeze(-1)

        # 加权Expert输出
        weighted_preds = (weights * expert_preds).sum(dim=1)

        # 分子级聚合
        n_mols = mol_idx.max().item() + 1
        mol_preds = torch.zeros(n_mols, 1, device=frag_emb.device)

        for i in range(n_mols):
            mask = (mol_idx == i)
            if mask.sum() > 0:
                mol_preds[i] = weighted_preds[mask].mean()

        return mol_preds.squeeze()


def train_fragmoe(train_data, test_data, n_epochs=200, device='cpu'):
    """训练FragMoE"""
    model = SimpleFragMoE(d_frag=128, n_experts=4).to(device)
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
        pred = model(train_batch.to(device), train_mol_idx.to(device))
        loss = criterion(pred, train_y.to(device))
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()

    # 准备测试数据
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
        pred = model(test_batch.to(device), test_mol_idx.to(device))
        pred_np = pred.cpu().numpy()
        y_np = test_y.numpy()

    return {
        'model': 'FragMoE',
        'R2': r2_score(y_np, pred_np),
        'RMSE': np.sqrt(mean_squared_error(y_np, pred_np)),
        'MAE': mean_absolute_error(y_np, pred_np),
    }


# ==========================================
# 主评估流程
# ==========================================

def nested_cv_evaluation(assay_name='DPPH', n_splits=5):
    """
    嵌套交叉验证评估

    外层CV：评估模型泛化性能
    内层CV：超参数调优
    """
    print(f"\n{'='*60}")
    print(f"嵌套CV评估 - {assay_name}")
    print(f"{'='*60}")

    # 加载数据
    df = pd.read_csv(DATA_PATH)
    assay_data = df[df['assay_type'] == assay_name].copy()

    print(f"总样本数: {len(assay_data)}")

    # 按scaffold分组进行分层（简化版：按分子量分层）
    assay_data['molwt'] = assay_data['SMILES'].apply(
        lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else 0
    )
    assay_data['mw_bin'] = pd.qcut(assay_data['molwt'], q=n_splits, labels=False)

    # 外层CV
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {
        'RandomForest': [],
        'XGBoost': [],
        'SVR': [],
        'FragMoE': [],
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(assay_data, assay_data['mw_bin'])):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        train_data = assay_data.iloc[train_idx]
        test_data = assay_data.iloc[test_idx]

        print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")

        # 准备特征
        X_train_morgan = np.array([compute_morgan_fingerprints(s) for s in train_data['SMILES']])
        X_test_morgan = np.array([compute_morgan_fingerprints(s) for s in test_data['SMILES']])

        X_train_mordred = np.array([compute_mordred_descriptors(s) for s in train_data['SMILES']])
        X_test_mordred = np.array([compute_mordred_descriptors(s) for s in test_data['SMILES']])

        # 合并特征
        X_train = np.hstack([X_train_morgan, X_train_mordred])
        X_test = np.hstack([X_test_morgan, X_test_mordred])

        y_train = train_data['pIC50'].values
        y_test = test_data['pIC50'].values

        # 训练基线模型
        print("  Training Random Forest...")
        rf_result = train_random_forest(X_train, y_train, X_test, y_test)
        results['RandomForest'].append(rf_result)
        print(f"    R² = {rf_result['R2']:.3f}")

        print("  Training XGBoost...")
        xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
        results['XGBoost'].append(xgb_result)
        print(f"    R² = {xgb_result['R2']:.3f}")

        print("  Training SVR...")
        svr_result = train_svr(X_train, y_train, X_test, y_test)
        results['SVR'].append(svr_result)
        print(f"    R² = {svr_result['R2']:.3f}")

        # 训练FragMoE
        print("  Training FragMoE...")
        train_dict = {'SMILES': train_data['SMILES'].tolist(), 'pIC50': y_train}
        test_dict = {'SMILES': test_data['SMILES'].tolist(), 'pIC50': y_test}

        fragmoe_result = train_fragmoe(train_dict, test_dict, n_epochs=150, device=device)
        if fragmoe_result:
            results['FragMoE'].append(fragmoe_result)
            print(f"    R² = {fragmoe_result['R2']:.3f}")
        else:
            print("    FragMoE training failed")

    # 汇总结果
    print(f"\n{'='*60}")
    print("汇总结果（5-Fold Nested CV）")
    print(f"{'='*60}")

    summary = []
    for model_name, model_results in results.items():
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
            print(f"  MAE = {np.mean(mae_scores):.3f}")

    # 保存结果
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_PATH / f"nested_cv_results_{assay_name}.csv", index=False)
    print(f"\n结果已保存至: {RESULTS_PATH / f'nested_cv_results_{assay_name}.csv'}")

    return summary_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--assay', type=str, default='DPPH', choices=['DPPH', 'ABTS', 'FRAP'])
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    results = nested_cv_evaluation(args.assay, args.folds)
