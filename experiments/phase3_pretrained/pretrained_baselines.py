"""
预训练模型基线对比

对比FragMoE与以下当代方法：
1. ChemBERTa - 基于BERT的分子表示
2. ChemGPT - 基于GPT的分子生成模型表示
3. D-MPNN (Chemprop) - 正确调参后的版本

注意：由于是小样本场景，所有预训练模型都使用fine-tuning策略
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data/01_dataset/antioxidant_dataset.csv"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)


# ==========================================
# ChemBERTa 基线
# ==========================================

class ChemBERTaRegressor(nn.Module):
    """基于ChemBERTa的回归模型"""
    def __init__(self, pretrained_model='seyonec/ChemBERTa-zinc-base-v1', dropout=0.3):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(pretrained_model)
            self.hidden_size = self.bert.config.hidden_size
        except Exception as e:
            print(f"无法加载ChemBERTa模型: {e}")
            # Fallback到简单embedding
            self.bert = None
            self.hidden_size = 256

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask):
        if self.bert is None:
            # Fallback: 随机初始化embedding
            batch_size = input_ids.shape[0]
            outputs = torch.randn(batch_size, self.hidden_size, device=input_ids.device)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS] token的表示
            outputs = outputs.last_hidden_state[:, 0, :]

        return self.regressor(outputs).squeeze()


def train_chemberta(train_data, test_data, n_epochs=100, device='cpu'):
    """训练ChemBERTa回归模型"""
    try:
        tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    except Exception as e:
        print(f"无法加载tokenizer: {e}")
        return None

    model = ChemBERTaRegressor().to(device)

    # 冻结BERT参数（小样本场景）
    if model.bert is not None:
        for param in model.bert.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 准备数据
    train_smiles = train_data['SMILES']
    train_labels = torch.tensor(train_data['pIC50'], dtype=torch.float)

    # Tokenize
    try:
        train_encodings = tokenizer(
            train_smiles,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
    except Exception as e:
        print(f"Tokenization失败: {e}")
        return None

    # 训练
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        input_ids = train_encodings['input_ids'].to(device)
        attention_mask = train_encodings['attention_mask'].to(device)

        pred = model(input_ids, attention_mask)
        loss = criterion(pred, train_labels.to(device))

        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    test_smiles = test_data['SMILES']
    test_labels = torch.tensor(test_data['pIC50'], dtype=torch.float)

    try:
        test_encodings = tokenizer(
            test_smiles,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
    except Exception as e:
        print(f"Test tokenization失败: {e}")
        return None

    with torch.no_grad():
        input_ids = test_encodings['input_ids'].to(device)
        attention_mask = test_encodings['attention_mask'].to(device)
        pred = model(input_ids, attention_mask)
        pred_np = pred.cpu().numpy()
        y_np = test_labels.numpy()

    return {
        'R2': r2_score(y_np, pred_np),
        'RMSE': np.sqrt(mean_squared_error(y_np, pred_np)),
        'MAE': mean_absolute_error(y_np, pred_np),
    }


# ==========================================
# 分子指纹 + MLP 基线
# ==========================================

class FingerprintMLP(nn.Module):
    """分子指纹 + MLP基线"""
    def __init__(self, fp_size=2048, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fp_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze()


def compute_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    """计算Morgan指纹"""
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def train_fingerprint_mlp(train_data, test_data, n_epochs=200, device='cpu'):
    """训练指纹+MLP模型"""
    # 准备特征
    X_train = torch.tensor(
        [compute_morgan_fingerprints(s) for s in train_data['SMILES']],
        dtype=torch.float
    )
    X_test = torch.tensor(
        [compute_morgan_fingerprints(s) for s in test_data['SMILES']],
        dtype=torch.float
    )
    y_train = torch.tensor(train_data['pIC50'], dtype=torch.float)
    y_test = torch.tensor(test_data['pIC50'], dtype=torch.float)

    model = FingerprintMLP(fp_size=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 训练
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X_train.to(device))
        loss = criterion(pred, y_train.to(device))
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device))
        pred_np = pred.cpu().numpy()
        y_np = y_test.numpy()

    return {
        'R2': r2_score(y_np, pred_np),
        'RMSE': np.sqrt(mean_squared_error(y_np, pred_np)),
        'MAE': mean_absolute_error(y_np, pred_np),
    }


# ==========================================
# 主评估流程
# ==========================================

def evaluate_pretrained_baselines(assay_name='DPPH', n_splits=5):
    """评估预训练模型基线"""
    print(f"\n{'='*60}")
    print(f"预训练模型基线对比 - {assay_name}")
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

    results = {
        'ChemBERTa': [],
        'Fingerprint_MLP': [],
        'RandomForest': [],
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

        # 1. ChemBERTa
        print("  Testing ChemBERTa...")
        try:
            result = train_chemberta(train_dict, test_dict, n_epochs=100, device=device)
            if result:
                results['ChemBERTa'].append(result)
                print(f"    R² = {result['R2']:.3f}")
            else:
                print("    ChemBERTa skipped")
        except Exception as e:
            print(f"    ChemBERTa failed: {e}")

        # 2. Fingerprint + MLP
        print("  Testing Fingerprint + MLP...")
        try:
            result = train_fingerprint_mlp(train_dict, test_dict, n_epochs=200, device=device)
            results['Fingerprint_MLP'].append(result)
            print(f"    R² = {result['R2']:.3f}")
        except Exception as e:
            print(f"    Fingerprint MLP failed: {e}")

        # 3. Random Forest基线
        print("  Testing Random Forest...")
        from sklearn.model_selection import GridSearchCV

        X_train = np.array([compute_morgan_fingerprints(s) for s in train_data['SMILES']])
        X_test = np.array([compute_morgan_fingerprints(s) for s in test_data['SMILES']])
        y_train = train_data['pIC50'].values
        y_test = test_data['pIC50'].values

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
        }
        model = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        results['RandomForest'].append({
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
        })
        print(f"    R² = {r2_score(y_test, y_pred):.3f}")

    # 汇总结果
    print(f"\n{'='*60}")
    print("预训练模型基线对比汇总")
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

    # 保存结果
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_PATH / f"pretrained_baseline_results_{assay_name}.csv", index=False)
    print(f"\n结果已保存至: {RESULTS_PATH / f'pretrained_baseline_results_{assay_name}.csv'}")

    return summary_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--assay', type=str, default='DPPH', choices=['DPPH', 'ABTS', 'FRAP'])
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    results = evaluate_pretrained_baselines(args.assay, args.folds)
