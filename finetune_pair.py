#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRISM pure-solvent solubility fine-tuning.

Split the supplied BigSolDB training CSV by canonical solute into training and
validation sets. Evaluate Leeds and SolProp through separate external test CSVs.
Save each seed's split, model configuration, and best-validation checkpoint.
The checkpoint can initialize the fusion modules in mixed-solvent training.
Report MSE, RMSE, and the fraction of predictions within one log unit.
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
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

# ---------- external project modules ----------
from model.getdata import smiles2graph
from model.fusion import make_fusion
from model.checkpoints import load_encoders, encoder_output_dim

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


def data_builder(d1, d2, y, T, batch_size=64, seed=42, val_ratio=0.05, split_path=None):
    """Split BigSolDB into solute-disjoint train/validation sets. Tests are external."""
    if not (len(d1) == len(d2) == len(y) == len(T)):
        raise ValueError("Graph, target, and temperature rows are not aligned.")
    groups = []
    for graph in d1:
        mol = Chem.MolFromSmiles(graph.smiles)
        if mol is None:
            raise ValueError(f"Invalid solute SMILES: {graph.smiles!r}")
        groups.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    if len(set(groups)) < 2:
        raise ValueError("At least two distinct solutes are needed for train/validation splitting.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(y)), groups=groups))
    if split_path:
        ensure_dir(os.path.dirname(split_path) or ".")
        split = np.full(len(y), "train", dtype=object)
        split[val_idx] = "val"
        pd.DataFrame({"row_index": np.arange(len(y)), "canonical_solute": groups,
                      "split": split}).to_csv(split_path, index=False)
    def loader(ids, shuffle):
        subset = lambda values: [values[i] for i in ids]
        dataset = PairDataset(subset(d1), subset(d2), subset(T), subset(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pair)
    return loader(train_idx, True), loader(val_idx, False)


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

def load_pretrained_encoders(sample, ckpt_paths, use_global_features=False, target_device=None):
    return load_encoders(sample, ckpt_paths, target_device or device,
                         allow_global_features=use_global_features)


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


def build_model_subset(sample_s, sample_v, ckpt_paths, solute_dropout, solvent_dropout,
                       mlp_hidden, mlp_dropout, t_embed_dim, fusion_type="adaptive",
                       use_global_features=False):
    sol_encs = load_pretrained_encoders(sample_s, ckpt_paths, use_global_features)
    solv_encs = load_pretrained_encoders(sample_v, ckpt_paths, use_global_features)
    z_dim = encoder_output_dim(sol_encs)
    if z_dim != encoder_output_dim(solv_encs):
        raise ValueError("Solute and solvent encoder dimensions differ.")
    sol_f = make_fusion(len(ckpt_paths), z_dim, dropout=solute_dropout, pair=True,
                        fusion_type=fusion_type, use_global_features=use_global_features).to(device)
    solv_f = make_fusion(len(ckpt_paths), z_dim, dropout=solvent_dropout, pair=True,
                         fusion_type=fusion_type, use_global_features=use_global_features).to(device)
    comb = TempPairMLP(z_dim, t_embed_dim=t_embed_dim, hidden=mlp_hidden, dropout=mlp_dropout).to(device)
    model = SolvationPredictor(sol_encs, solv_encs, sol_f, solv_f, comb).to(device)
    model.fusion_type = fusion_type
    model.use_global_features = use_global_features
    return model


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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        help="Fraction of canonical solute groups reserved for validation; test datasets are external.",
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
        help="Legacy argument; actual width is read from encoder checkpoints.",
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

    parser.add_argument("--use_global_features", action="store_true")
    parser.add_argument("--fusion_type", choices=["static", "adaptive"], default="adaptive")
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
        run_dir = os.path.join(args.out_dir, f"seed{seed}")
        ensure_dir(run_dir)
        tr_loader, va_loader = data_builder(
            d1,
            d2,
            y,
            T,
            batch_size=args.batch_size,
            seed=seed,
            val_ratio=args.val_ratio,
            split_path=os.path.join(run_dir, "split.csv"),
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
            fusion_type=args.fusion_type,
            use_global_features=args.use_global_features,
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
        torch.save({
            "format_version": 2,
            "model_state_dict": model.state_dict(),
            "fusion_type": model.fusion_type,
            "use_global_features": model.use_global_features,
            "encoder_configs": [enc.encoder_config() for enc in model.sol_encs],
            "encoder_ckpts": ckpt_paths_subset,
            "config": vars(args),
            "seed": seed,
            "best_val_rmse": best_val_rmse,
        }, os.path.join(run_dir, "model_best.pth"))
        save_json(vars(args), os.path.join(run_dir, "config.json"))
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
                    "test_rmse": float(np.sqrt(mse_val)),
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
            "mse_mean": float(np.square(val_rmse_list).mean()),
            "mse_std": float(np.square(val_rmse_list).std(ddof=0)),
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
                "rmse_mean": float(np.sqrt(mse_arr).mean()),
                "rmse_std": float(np.sqrt(mse_arr).std(ddof=0)),
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
