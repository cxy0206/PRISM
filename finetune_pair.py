#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise solvation regression fine-tuning (solute + solvent + temperature).

Task
----
Given:
    - solute SMILES
    - solvent SMILES
    - temperature (K)
predict:
    - logS (solubility, e.g., in log10 scale)

Model
-----
- Multiple pre-trained ChemSpace encoders (e.g., task_0 / task_1 / task_2 / task_none)
  are loaded from checkpoint files.
- For both solute and solvent, we:
    * encode with each encoder,
    * fuse encoder outputs with a WeightedFusion module.
- A small MLP (TempPairMLP) combines:
    * fused solute embedding
    * fused solvent embedding
    * embedded temperature
  to produce a scalar prediction.

Data format
-----------
Training CSV (and test CSVs) are expected to contain at least:
    - solute_smiles
    - solvent_smiles
    - temperature  (or 't')
    - logS         (or 'logs')

Column names are resolved in a case-insensitive way with simple fallbacks.

Metrics
-------
For each run (seed) and dataset:
    - MSE  (Mean Squared Error)
    - pct(|pred - logS| <= 1.0)   # coverage within ±1 log unit

Across seeds, the script reports mean ± std for:
    - validation MSE
    - per-test-dataset MSE
    - per-test-dataset pct(|pred-logS| <= 1)

Typical usage
-------------
Example:

    python finetune_pair_solvation.py \\
        --train_csv datasets/bigsoldb_chemprop_nonaq.csv \\
        --test_csvs datasets/leeds_all_chemprop.csv,\\
                    datasets/solprop_chemprop_nonaq.csv \\
        --encoder_ckpts "ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,\\
                         ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,\\
                         ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,\\
                         ckpt/tau_0.1__scale_0.25/task_none/best_model.pth" \\
        --space_indices 0,1,2 \\
        --seeds 42,43,44 \\
        --batch_size 64 --val_ratio 0.05 --epochs 200 --weight_decay 1e-5 \\
        --solute_dropout 0.3 --solvent_dropout 0.3 --mlp_hidden 1024,128,64 \\
        --mlp_dropout 0.2 --t_embed_dim 32 \\
        --learning_rate 1e-4 --emb_dim 512 \\
        --patience_es 50 --patience_lr 30 --lr_factor 0.5 \\
        --out_dir results/solvation_pair_finetune
"""

import os
import json
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from rdkit import Chem  # kept for future extensions
from sklearn.model_selection import train_test_split

# ---------- external project modules ----------
from model.getdata import smiles2graph
from model.mpn_vas_st import GNNModelWithNewLoss
from model.fusion import WeightedFusion

# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[info] Using device:", device)

# -------------------------------------------------------------------------
# Global hyperparameters (default values; can be overridden via CLI)
# -------------------------------------------------------------------------
LEARNING_RATE = 1e-4
EMB_DIM = 512
PATIENCE_ES = 50
PATIENCE_LR = 30
LR_FACTOR = 0.5


# ============================ basic utils ============================

def set_global_seed(seed: int):
    """
    Set Python / NumPy / PyTorch random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    """
    Create directory `p` if it does not exist.
    """
    os.makedirs(p, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    """
    Save a Python dict as a JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ================== pct(|pred − y| ≤ 1) ==================

def pct_within_1_np(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute percentage of samples satisfying |pred - label| <= 1.

    Parameters
    ----------
    preds : np.ndarray
        Model predictions.
    labels : np.ndarray
        Ground-truth labels.

    Returns
    -------
    float
        Fraction of samples with absolute error <= 1.
    """
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    err = np.abs(preds - labels)
    return float((err <= 1.0).mean())


# ================== CSV → graphs ==================

def _resolve_col(df: pd.DataFrame, name: str, fallback: Optional[List[str]] = None) -> str:
    """
    Resolve a column name in a DataFrame in a case-insensitive way.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    name : str
        Desired column name (case-insensitive).
    fallback : list of str, optional
        Additional candidate names to try if `name` is not found.

    Returns
    -------
    str
        The actual column name in `df`.

    Raises
    ------
    KeyError
        If no matching column is found.
    """
    norm = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key in norm:
        return norm[key]
    if fallback:
        for cand in fallback:
            cand_key = cand.strip().lower()
            if cand_key in norm:
                return norm[cand_key]
    raise KeyError(f"Column {name} not found in {df.columns}")


def _smiles_graph_inputs_from_csv(csv_path: str):
    """
    Read a solvation CSV and convert solute / solvent SMILES to graph objects.

    Expected columns (case-insensitive):
        - solute_smiles
        - solvent_smiles
        - temperature (or 't')
        - logS (or 'logs')

    Returns
    -------
    d1, d2 : list of Data
        Graphs for solute and solvent.
    y : np.ndarray
        logS labels.
    T : np.ndarray
        Temperature values.
    """
    df = pd.read_csv(csv_path)

    solute_col = _resolve_col(df, "solute_smiles", ["solute_smiles"])
    solvent_col = _resolve_col(df, "solvent_smiles", ["solvent_smiles"])
    temp_col = _resolve_col(df, "temperature", ["temperature", "t"])
    label_col = _resolve_col(df, "logS", ["logs", "logS"])

    y = df[label_col].astype(float).to_numpy()
    T = df[temp_col].astype(float).to_numpy()

    solute = df[solute_col].astype(str).tolist()
    solvent = df[solvent_col].astype(str).tolist()

    # We pass both y and T to smiles2graph; y typically becomes Data.y
    # and T may be stored in global features (depending on implementation).
    d1 = smiles2graph(solute, y=y, properties=T)
    d2 = smiles2graph(solvent, y=y, properties=T)
    return d1, d2, y, T


# ================== dataset classes & loaders ==================

class PairDataset(Dataset):
    """
    Dataset over solute / solvent pairs, temperature, and label.

    Each item is:
        (solute_graph, solvent_graph, temperature, label)
    """
    def __init__(self, s, v, temps, labels):
        self.s = s
        self.v = v
        self.temps = temps
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.s[idx], self.v[idx], float(self.temps[idx]), float(self.labels[idx])


def collate_pair(batch):
    """
    Collate function that:
        - batches solute graphs,
        - batches solvent graphs,
        - converts temperatures and labels into tensors.
    """
    sol, solv, temps, labels = zip(*batch)
    b1 = Batch.from_data_list(list(sol))
    b2 = Batch.from_data_list(list(solv))
    temps = torch.tensor(temps, dtype=torch.float32).view(-1, 1)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return b1, b2, temps, labels


def data_builder(
    d1,
    d2,
    y,
    T,
    batch_size: int = 64,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 1e-6,
):
    """
    Split the dataset (on the training CSV) into train / val / test
    using random splits, then build DataLoaders.

    In this script we only use the train and val loaders; the small test
    split is kept for compatibility with the original ablation code.

    Parameters
    ----------
    d1, d2 : list of Data
        Solute and solvent graphs.
    y : np.ndarray
        Labels.
    T : np.ndarray
        Temperatures.
    batch_size : int
    seed : int
    val_ratio : float
    test_ratio : float

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    idx = list(range(len(y)))
    train_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_ratio / max(1e-8, (1 - test_ratio)),
        random_state=seed,
    )

    def sub(lst, ids):
        return [lst[i] for i in ids]

    ds_tr = PairDataset(sub(d1, train_idx), sub(d2, train_idx),
                        sub(T, train_idx), sub(y, train_idx))
    ds_va = PairDataset(sub(d1, val_idx), sub(d2, val_idx),
                        sub(T, val_idx), sub(y, val_idx))
    ds_te = PairDataset(sub(d1, test_idx), sub(d2, test_idx),
                        sub(T, test_idx), sub(y, test_idx))

    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_pair)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate_pair)
    te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_pair)
    return tr, va, te


def make_full_test_loader(csv_path: str, batch_size: int = 64) -> DataLoader:
    """
    Build a test DataLoader from a CSV, without splitting.

    Parameters
    ----------
    csv_path : str
        Path to test CSV.
    batch_size : int

    Returns
    -------
    DataLoader
        Test loader over the full dataset.
    """
    d1, d2, y, T = _smiles_graph_inputs_from_csv(csv_path)
    ds = PairDataset(d1, d2, T, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pair)


# ================== encoder loading ==================

def load_pretrained_encoders(sample, ckpt_paths: List[str]):
    """
    Load a list of pre-trained ChemSpace encoders from checkpoint paths.

    Parameters
    ----------
    sample : Data
        A sample graph used to infer node / edge feature dimensions.
    ckpt_paths : list of str
        Paths to encoder checkpoints.

    Returns
    -------
    encs : list of GNNModelWithNewLoss
        Encoder modules.
    """
    encs = []
    for path in ckpt_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")

        model = GNNModelWithNewLoss(
            num_node_features=sample.x.shape[1],
            num_edge_features=sample.edge_attr.shape[1],
            num_global_features=0,
            cov_num=3,
            hidden_dim=EMB_DIM,
        ).to(device)

        ck = torch.load(path, map_location=device)
        state = ck.get("encoder_state_dict", ck)
        model.load_state_dict(state, strict=True)
        encs.append(model)
    return encs


# ================== model definition ==================

def _stack_encode(encs: List[nn.Module], batch: Batch) -> torch.Tensor:
    """
    Encode a batch with each encoder and stack the embeddings along a new dimension.

    Parameters
    ----------
    encs : list of nn.Module
    batch : Batch

    Returns
    -------
    embs : torch.Tensor
        Shape: [batch_size, num_encoders, emb_dim]
    """
    return torch.stack([enc(batch) for enc in encs], dim=1)


class TempPairMLP(nn.Module):
    """
    MLP that combines:
        - fused solute embedding
        - fused solvent embedding
        - embedded temperature

    to predict a scalar logS value.
    """
    def __init__(
        self,
        z_dim: int,
        t_embed_dim: int = 32,
        hidden: Tuple[int, ...] = (1024, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.t_enc = nn.Sequential(nn.Linear(1, t_embed_dim), nn.ReLU())
        in_dim = z_dim * 2 + t_embed_dim

        layers: List[nn.Module] = []
        dims = [in_dim, *hidden, 1]
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.nn = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if T.dim() == 1:
            T = T.unsqueeze(1)
        t = self.t_enc(T.to(z1.device))
        x = torch.cat([z1, z2, t], dim=-1)
        return self.nn(x).squeeze(-1)


class SolvationPredictor(nn.Module):
    """
    Full solvation regression model:

        solute encoders  -> WeightedFusion -> fused solute embedding
        solvent encoders -> WeightedFusion -> fused solvent embedding
        + temperature     -> TempPairMLP   -> scalar logS prediction
    """
    def __init__(
        self,
        sol_encs: List[nn.Module],
        solv_encs: List[nn.Module],
        sol_fusion: nn.Module,
        solv_fusion: nn.Module,
        combiner: TempPairMLP,
    ):
        super().__init__()
        self.sol_encs = nn.ModuleList(sol_encs)
        self.solv_encs = nn.ModuleList(solv_encs)
        self.sol_f = sol_fusion
        self.solv_f = solv_fusion
        self.combine = combiner

    def _fuse(self, fusion: nn.Module, embs: torch.Tensor, g: Optional[torch.Tensor]):
        out = fusion(embs, global_features=g)
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        sol_batch: Batch,
        solv_batch: Batch,
        temps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = _stack_encode(self.sol_encs, sol_batch)
        z2 = _stack_encode(self.solv_encs, solv_batch)

        g1 = getattr(sol_batch, "global_features", None)
        g2 = getattr(solv_batch, "global_features", None)

        z1 = self._fuse(self.sol_f, z1, g1)
        z2 = self._fuse(self.solv_f, z2, g2)

        pred = self.combine(z1, z2, temps)
        return pred, z1, z2


def build_model_subset(
    sample_s,
    sample_v,
    ckpt_paths: List[str],
    solute_dropout: float,
    solvent_dropout: float,
    mlp_hidden: Tuple[int, ...],
    mlp_dropout: float,
    t_embed_dim: int,
) -> SolvationPredictor:
    """
    Build a SolvationPredictor using a subset of pre-trained encoders.

    Parameters
    ----------
    sample_s, sample_v : Data
        Example solute and solvent graphs used to infer feature sizes.
    ckpt_paths : list of str
        Paths to encoder checkpoints (one per ChemSpace task).
    solute_dropout, solvent_dropout : float
        Dropout rates for solute / solvent fusion modules.
    mlp_hidden : tuple of int
        Hidden layer sizes for TempPairMLP.
    mlp_dropout : float
        Dropout rate in TempPairMLP.
    t_embed_dim : int
        Temperature embedding dimension.

    Returns
    -------
    SolvationPredictor
    """
    sol_encs = load_pretrained_encoders(sample_s, ckpt_paths)
    solv_encs = load_pretrained_encoders(sample_v, ckpt_paths)

    sol_f = WeightedFusion(len(ckpt_paths), EMB_DIM, dropout=solute_dropout, pair=True).to(device)
    solv_f = WeightedFusion(len(ckpt_paths), EMB_DIM, dropout=solvent_dropout, pair=True).to(device)

    # Probe the fused embedding dimension
    with torch.no_grad():
        b1 = Batch.from_data_list([sample_s]).to(device)
        b2 = Batch.from_data_list([sample_v]).to(device)
        z1 = _stack_encode(sol_encs, b1)
        z2 = _stack_encode(solv_encs, b2)
        g1 = getattr(b1, "global_features", None)
        g2 = getattr(b2, "global_features", None)
        z1 = sol_f(z1, g1)
        z2 = solv_f(z2, g2)
        if isinstance(z1, tuple):
            z1 = z1[0]
        if isinstance(z2, tuple):
            z2 = z2[0]
        z_dim = z1.shape[-1]

    comb = TempPairMLP(
        z_dim,
        t_embed_dim=t_embed_dim,
        hidden=mlp_hidden,
        dropout=mlp_dropout,
    ).to(device)

    return SolvationPredictor(sol_encs, solv_encs, sol_f, solv_f, comb).to(device)


# ================== training & testing ==================

def rmse_tensor(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSE between prediction and target tensors.
    """
    pred = pred.view(-1)
    tgt = tgt.view(-1)
    return torch.sqrt(nn.MSELoss()(pred, tgt))


def train_one(
    model: SolvationPredictor,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    epochs: int,
    weight_decay: float,
    prefix: str = "",
) -> float:
    """
    Train the model for a single seed and return the best validation RMSE.

    Parameters
    ----------
    model : SolvationPredictor
    tr_loader, va_loader : DataLoader
    epochs : int
    weight_decay : float
    prefix : str
        String prefix printed before log messages (useful for multi-seed runs).

    Returns
    -------
    float
        Best validation RMSE observed during training.
    """
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=LR_FACTOR,
        patience=PATIENCE_LR,
    )

    best = float("inf")
    patience = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        total = 0.0
        for s, v, t, y in tr_loader:
            s = s.to(device)
            v = v.to(device)
            t = t.to(device)
            y = y.view(-1).to(device)

            p, _, _ = model(s, v, t)
            loss = rmse_tensor(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        tr_rmse = total / max(1, len(tr_loader))

        # validation
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for s, v, t, y in va_loader:
                s = s.to(device)
                v = v.to(device)
                t = t.to(device)
                y = y.view(-1).to(device)

                p, _, _ = model(s, v, t)
                preds.append(p.cpu())
                labs.append(y.cpu())

        if preds:
            preds_cat = torch.cat(preds)
            labs_cat = torch.cat(labs)
            val_rmse = rmse_tensor(preds_cat, labs_cat).item()
        else:
            val_rmse = float("inf")

        sched.step(val_rmse)

        if val_rmse + 1e-8 < best:
            best = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_ES:
                print(f"{prefix}[EarlyStop] best_val_rmse={best:.4f}")
                break

        print(f"{prefix}[Ep {ep:03d}] train_RMSE={tr_rmse:.4f} val_RMSE={val_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best


def test_loader_metrics(
    model: SolvationPredictor,
    loader: DataLoader,
) -> Tuple[float, float]:
    """
    Evaluate the model on a given loader and compute:

        - MSE
        - pct(|pred - label| <= 1)

    Returns
    -------
    mse_val : float
    pct1 : float
    """
    model.eval()
    mse = nn.MSELoss()

    preds = []
    labs = []

    with torch.no_grad():
        for s, v, t, y in loader:
            s = s.to(device)
            v = v.to(device)
            t = t.to(device)
            y = y.view(-1).to(device)

            p, _, _ = model(s, v, t)
            preds.append(p.cpu().numpy())
            labs.append(y.cpu().numpy())

    if not preds:
        return float("inf"), 0.0

    preds_np = np.concatenate(preds)
    labs_np = np.concatenate(labs)

    mse_val = float(mse(torch.tensor(preds_np), torch.tensor(labs_np)).item())
    pct1 = pct_within_1_np(preds_np, labs_np)
    return mse_val, pct1


# ================== argument parsing & main ==================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairwise solvation regression fine-tuning (solute + solvent + temperature)."
    )

    # Data
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training CSV with solute_smiles, solvent_smiles, temperature, logS.",
    )
    parser.add_argument(
        "--test_csvs",
        type=str,
        required=True,
        help=(
            "Comma-separated list of test CSV paths. "
            "Each must have the same columns as the training CSV."
        ),
    )

    # Encoders & subset
    parser.add_argument(
        "--encoder_ckpts",
        type=str,
        required=True,
        help=(
            "Comma-separated list of encoder checkpoint paths. "
            "Example: '.../task_0/best_model.pth,.../task_1/best_model.pth,...'"
        ),
    )
    parser.add_argument(
        "--space_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated indices specifying which encoders to use "
            "(subset of [0..N-1]; default: all)."
        ),
    )

    # Seeds / runs
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated list of seeds for independent runs (default: '42,43,44').",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation ratio on the training CSV (random split).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs per seed.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--solute_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for solute fusion module.",
    )
    parser.add_argument(
        "--solvent_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for solvent fusion module.",
    )
    parser.add_argument(
        "--mlp_hidden",
        type=str,
        default="1024,128,64",
        help="Comma-separated hidden layer sizes for TempPairMLP (default: '1024,128,64').",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.2,
        help="Dropout rate in TempPairMLP.",
    )
    parser.add_argument(
        "--t_embed_dim",
        type=int,
        default=32,
        help="Temperature embedding dimension in TempPairMLP.",
    )

    # Global hyperparameters to override defaults
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=512,
        help="Embedding dimension for each encoder (default: 512).",
    )
    parser.add_argument(
        "--patience_es",
        type=int,
        default=50,
        help="Early-stopping patience in epochs (default: 50).",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=30,
        help="LR scheduler patience in epochs (default: 30).",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="LR decay factor for ReduceLROnPlateau (default: 0.5).",
    )

    # Output
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/solvation_pair_finetune",
        help="Output directory for CSV summaries.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional experiment name used as a tag in output files.",
    )

    return parser.parse_args()


def main():
    global LEARNING_RATE, EMB_DIM, PATIENCE_ES, PATIENCE_LR, LR_FACTOR

    args = parse_args()

    # Override global hyperparameters
    LEARNING_RATE = args.learning_rate
    EMB_DIM = args.emb_dim
    PATIENCE_ES = args.patience_es
    PATIENCE_LR = args.patience_lr
    LR_FACTOR = args.lr_factor

    print("[info] Hyperparameters:")
    print(f"  learning_rate = {LEARNING_RATE}")
    print(f"  emb_dim       = {EMB_DIM}")
    print(f"  patience_es   = {PATIENCE_ES}")
    print(f"  patience_lr   = {PATIENCE_LR}")
    print(f"  lr_factor     = {LR_FACTOR}")

    ensure_dir(args.out_dir)

    # Parse encoder checkpoint list
    all_ckpts = [p.strip() for p in args.encoder_ckpts.split(",") if p.strip()]
    if not all_ckpts:
        raise ValueError("No encoder checkpoints provided via --encoder_ckpts.")

    # Parse subset indices
    if args.space_indices is None:
        subset_indices = list(range(len(all_ckpts)))
    else:
        subset_indices = [int(s.strip()) for s in args.space_indices.split(",") if s.strip()]
    if not subset_indices:
        raise ValueError("Empty subset specified via --space_indices.")

    ckpt_paths_subset = [all_ckpts[i] for i in subset_indices]
    subset_name = "_".join(str(i) for i in subset_indices)

    print(f"[info] Using encoder indices: {subset_indices}")
    print("[info] Encoder checkpoints used:")
    for idx, path in zip(subset_indices, ckpt_paths_subset):
        print(f"  - index {idx}: {path}")

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds provided via --seeds.")

    # Parse test CSVs
    test_csvs = [p.strip() for p in args.test_csvs.split(",") if p.strip()]
    if not test_csvs:
        raise ValueError("No test CSVs provided via --test_csvs.")

    test_aliases = [os.path.splitext(os.path.basename(p))[0] for p in test_csvs]

    # For each test dataset, store list of metrics over seeds
    test_mse: Dict[str, List[float]] = {name: [] for name in test_aliases}
    test_pct1: Dict[str, List[float]] = {name: [] for name in test_aliases}
    val_rmse_list: List[float] = []

    # One row per seed per dataset for detailed CSV
    detailed_records: List[Dict[str, Any]] = []

    base_dataset_name = os.path.splitext(os.path.basename(args.train_csv))[0]
    exp_tag = args.exp_name if args.exp_name else base_dataset_name

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        set_global_seed(seed)

        # Build train / val from training CSV
        d1, d2, y, T = _smiles_graph_inputs_from_csv(args.train_csv)
        tr_loader, va_loader, _ = data_builder(
            d1,
            d2,
            y,
            T,
            batch_size=args.batch_size,
            seed=seed,
            val_ratio=args.val_ratio,
            test_ratio=1e-6,  # effectively no test split on the training CSV
        )

        if len(tr_loader.dataset) == 0:
            raise RuntimeError("Empty training dataset after splitting; check your CSV and ratios.")

        sample_s = tr_loader.dataset.s[0]
        sample_v = tr_loader.dataset.v[0]

        # Build model for this subset
        model = build_model_subset(
            sample_s=sample_s,
            sample_v=sample_v,
            ckpt_paths=ckpt_paths_subset,
            solute_dropout=args.solute_dropout,
            solvent_dropout=args.solvent_dropout,
            mlp_hidden=tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip()),
            mlp_dropout=args.mlp_dropout,
            t_embed_dim=args.t_embed_dim,
        )

        # Train
        best_val_rmse = train_one(
            model,
            tr_loader,
            va_loader,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            prefix=f"[subset={subset_name}|seed={seed}] ",
        )
        val_rmse_list.append(best_val_rmse)
        print(f"[Seed {seed}] best_val_RMSE={best_val_rmse:.4f}")

        # Test on each external test CSV
        for csv_path, alias in zip(test_csvs, test_aliases):
            te_loader = make_full_test_loader(csv_path, batch_size=args.batch_size)
            mse_val, pct1 = test_loader_metrics(model, te_loader)
            test_mse[alias].append(mse_val)
            test_pct1[alias].append(pct1)

            print(
                f"[Seed {seed}] Test '{alias}': "
                f"MSE={mse_val:.4f}, pct(|err|<=1)={pct1:.3f}"
            )

            detailed_records.append(
                {
                    "seed": seed,
                    "subset": subset_name,
                    "num_encoders": len(subset_indices),
                    "train_dataset": base_dataset_name,
                    "test_dataset": alias,
                    "val_rmse": best_val_rmse,
                    "test_mse": mse_val,
                    "test_pct1": pct1,
                }
            )

    # Aggregate across seeds
    val_mean = float(np.mean(val_rmse_list))
    val_std = float(np.std(val_rmse_list))

    print("\n===== Summary across seeds =====")
    print(f"Val RMSE = {val_mean:.4f} ± {val_std:.4f}")

    for alias in test_aliases:
        mse_arr = np.array(test_mse[alias], dtype=float)
        pct_arr = np.array(test_pct1[alias], dtype=float)
        print(
            f"Test '{alias}': "
            f"MSE = {mse_arr.mean():.4f} ± {mse_arr.std():.4f} | "
            f"pct(|err|<=1) = {pct_arr.mean():.3f} ± {pct_arr.std():.3f}"
        )

    # Save detailed per-seed records
    df_detail = pd.DataFrame(detailed_records)
    detail_path = os.path.join(
        args.out_dir,
        f"{exp_tag}_{subset_name}_pair_detail.csv",
    )
    df_detail.to_csv(detail_path, index=False)

    # Save aggregated summary (long format: one row per dataset)
    summary_rows = []

    summary_rows.append(
        {
            "train_dataset": base_dataset_name,
            "test_dataset": "val",
            "subset": subset_name,
            "num_encoders": len(subset_indices),
            "mse_mean": val_mean**2,  # Note: val_mean is RMSE; store MSE here for completeness
            "mse_std": None,
            "rmse_mean": val_mean,
            "rmse_std": val_std,
            "pct1_mean": None,
            "pct1_std": None,
        }
    )

    for alias in test_aliases:
        mse_arr = np.array(test_mse[alias], dtype=float)
        pct_arr = np.array(test_pct1[alias], dtype=float)
        summary_rows.append(
            {
                "train_dataset": base_dataset_name,
                "test_dataset": alias,
                "subset": subset_name,
                "num_encoders": len(subset_indices),
                "mse_mean": float(mse_arr.mean()),
                "mse_std": float(mse_arr.std()),
                "rmse_mean": float(np.sqrt(mse_arr.mean())),
                "rmse_std": None,  # can be derived if needed
                "pct1_mean": float(pct_arr.mean()),
                "pct1_std": float(pct_arr.std()),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        args.out_dir,
        f"{exp_tag}_{subset_name}_pair_summary.csv",
    )
    df_summary.to_csv(summary_path, index=False)

    print(f"\n[done] Detailed results saved to: {detail_path}")
    print(f"[done] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
