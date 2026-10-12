#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRISM single-molecule fine-tuning.

Train the encoders and fusion module on scaffold-split classification or
regression tasks. Default fusion is static for classification and adaptive for
regression. Encoder dimensions are restored from checkpoint configurations;
global molecular features require explicit opt-in.
"""

import os
import argparse
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader as GeoDataLoader
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[info] Device:", device)

# -------------------------------------------------------------------------
# Global hyperparameters (can be overridden by CLI arguments)
# -------------------------------------------------------------------------
LEARNING_RATE = 1e-4  # default learning rate
EMB_DIM = 512         # default embedding dimension
PATIENCE_ES = 20      # default early-stopping patience (epochs)
PATIENCE_LR = 10      # default LR scheduler patience (epochs)
LR_FACTOR = 0.5       # default LR decay factor


# -------------------------------------------------------------------------
# Project-specific imports
# -------------------------------------------------------------------------
from model.getdata import smiles2graph
from model.fusion import make_fusion, FusionFineTuneModel
from model.checkpoints import load_encoders, encoder_output_dim


# -------------------------------------------------------------------------
# Murcko-scaffold split utilities
# -------------------------------------------------------------------------
def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Generate the Bemis–Murcko scaffold SMILES string for a given molecule.
    This is used to construct scaffold-based train/val/test splits.
    """
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles,
        includeChirality=include_chirality,
    )


def scaffold_split(
    smiles_list: List[str],
    val_split: float,
    test_split: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a scaffold-based split of the dataset into train, validation, and test.

    The split is performed at scaffold level:
    molecules sharing the same scaffold are assigned to the same subset.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings, one per molecule.
    val_split : float
        Fraction of the data to use for validation.
    test_split : float
        Fraction of the data to use for test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Arrays of indices for each split.
    """
    rng = np.random.RandomState(seed)
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scf = generate_scaffold(smi, include_chirality=True)
        scaffolds[scf].append(i)
    groups = list(scaffolds.values())
    rng.shuffle(groups)

    n = len(smiles_list)
    n_val, n_test = int(n * val_split), int(n * test_split)
    val_idx, test_idx, train_idx = [], [], []
    for g in groups:
        if len(val_idx) + len(g) <= n_val:
            val_idx += g
        elif len(test_idx) + len(g) <= n_test:
            test_idx += g
        else:
            train_idx += g

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# -------------------------------------------------------------------------
# Dataset wrapper
# -------------------------------------------------------------------------
class MoleculeListDataset(torch.utils.data.Dataset):
    """
    Simple Dataset wrapper that stores a list of PyG Data objects.

    It also supports indexing by a list / numpy array of indices, which can
    be convenient for creating scaffold-based subsets.
    """
    def __init__(self, data_list):
        self._data = list(data_list)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        # Allow slicing with tensors, lists, numpy arrays.
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(idx, (list, tuple, np.ndarray)):
            return MoleculeListDataset([self._data[i] for i in idx])
        return self._data[idx]


def load_data(
    csv_path: str,
    label_col: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
    task_type: str = "classification",
):
    """
    Load a CSV file and construct train/val/test loaders with scaffold split.

    The CSV file is expected to contain:
        - a 'smiles' column with molecular SMILES
        - a label column (specified by `label_col`)

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV.
    label_col : str
        Name of the label column in the CSV.
    batch_size : int
    val_split : float
    test_split : float
    seed : int
    task_type : {'classification', 'regression'}
        Determines how labels are processed. For classification, labels are
        binarized if more than 2 unique values are present.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        Torch Geometric DataLoaders for each split.
    smiles_list : list of str
        Original SMILES list (useful for debugging / logging).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    if label_col not in df.columns:
        raise ValueError(f"Input CSV must contain the label column '{label_col}'.")

    smiles_list = df["smiles"].tolist()
    y = df[label_col].astype(np.float32).values

    # For binary classification, ensure labels are in {0,1}.
    if task_type == "classification":
        unique_vals = np.unique(y)
        if len(unique_vals) > 2:
            # Simple threshold-based binarization.
            # Users can customize according to their dataset.
            y = (y > 0.5).astype(np.float32)

    data_list = smiles2graph(smiles_list, y=y)
    base_dataset = MoleculeListDataset(data_list)

    tr_idx, vl_idx, te_idx = scaffold_split(smiles_list, val_split, test_split, seed)

    def subset(ds, idx):
        if idx.size == 0:
            return MoleculeListDataset([])
        tensor_idx = torch.tensor(idx, dtype=torch.long)
        return ds[tensor_idx]

    train_ds = subset(base_dataset, tr_idx)
    val_ds = subset(base_dataset, vl_idx)
    test_ds = subset(base_dataset, te_idx)

    train_loader = GeoDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, smiles_list


# -------------------------------------------------------------------------
# Encoder & fusion model construction
# -------------------------------------------------------------------------
def load_pretrained_encoders(sample, ckpt_paths, use_global_features=False):
    return load_encoders(sample, ckpt_paths, device, allow_global_features=use_global_features)


def get_finetune_model(sample, dropout, ckpt_paths, fusion_type="static", use_global_features=False):
    """Build a trainable PRISM encoder ensemble and its prediction head."""
    encoders = load_pretrained_encoders(sample, ckpt_paths, use_global_features)
    fusion = make_fusion(len(encoders), encoder_output_dim(encoders), dropout=dropout,
                         fusion_type=fusion_type, use_global_features=use_global_features).to(device)
    return FusionFineTuneModel(encoders, fusion, fusion_type=fusion_type).to(device)


# -------------------------------------------------------------------------
# Helper functions for handling fusion outputs
# -------------------------------------------------------------------------
def _maybe_flatten_for_concat(model, embs: torch.Tensor) -> torch.Tensor:
    """
    If the fusion method is 'concat', flatten encoder dimension into feature dimension.
    Otherwise, return the embeddings as-is.

    Parameters
    ----------
    model : FusionFineTuneModel
        The fine-tuning model, which may have attribute 'fusion_method'.
    embs : torch.Tensor
        Tensor of shape [batch_size, num_encoders, emb_dim].

    Returns
    -------
    torch.Tensor
        Either the same shape (if not concat), or [batch_size, num_encoders * emb_dim].
    """
    if getattr(model, "fusion_method", None) == "concat":
        return embs.view(embs.size(0), -1)
    return embs


def _get_global_features(batch, embs_device):
    """
    Extract optional global features from a batch, if present.
    They are moved to the same device as the embeddings.

    Parameters
    ----------
    batch : Data
    embs_device : torch.device

    Returns
    -------
    torch.Tensor or None
    """
    gfeat = getattr(batch, "global_features", None)
    if gfeat is not None:
        gfeat = gfeat.to(embs_device)
        if gfeat.dim() == 1:
            gfeat = gfeat.unsqueeze(0)
    return gfeat


# -------------------------------------------------------------------------
# Training & evaluation (classification)
# -------------------------------------------------------------------------
def train_and_validate_classification(
    model,
    train_loader,
    val_loader,
    epochs: int,
    weight_decay: float = 0.0,
):
    """
    Train and validate the model for a classification task using BCEWithLogitsLoss.

    Returns
    -------
    train_losses : list of float
        Per-epoch training losses.
    val_aucs : list of float
        Per-epoch validation ROC-AUC scores.
    """
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=PATIENCE_LR
    )

    train_losses, val_aucs = [], []
    best_val_auc, best_state, patience_cnt = -1e9, None, 0

    for epoch in range(epochs):
        # --------- training ---------
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            embs = [encoder(batch) for encoder in model.encoders]
            embs = torch.stack(embs, dim=1)  # [B, num_encoders, emb_dim]
            embs = _maybe_flatten_for_concat(model, embs)
            gfeat = _get_global_features(batch, embs.device)

            out = model.fusion(embs, global_features=gfeat)
            logits = out[0] if isinstance(out, tuple) else out
            label = batch.y.view(-1).float().to(device)

            loss = bce(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_loss)

        # --------- validation ---------
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embs = [encoder(batch) for encoder in model.encoders]
                embs = torch.stack(embs, dim=1)
                embs = _maybe_flatten_for_concat(model, embs)
                gfeat = _get_global_features(batch, embs.device)

                out = model.fusion(embs, global_features=gfeat)
                logits = out[0] if isinstance(out, tuple) else out
                preds.append(logits.cpu())
                labs.append(batch.y.view(-1).cpu())

        preds = torch.cat(preds) if preds else torch.empty(0)
        labs = torch.cat(labs) if labs else torch.empty(0)

        if preds.numel() == 0:
            auc = 0.5
        else:
            probs = torch.sigmoid(preds).numpy()
            try:
                auc = roc_auc_score(labs.numpy(), probs)
            except Exception:
                auc = 0.5

        val_aucs.append(auc)
        scheduler.step(auc)

        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_ES:
                print(f"[Early Stop] epoch={epoch}, best_val_auc={best_val_auc:.4f}")
                break

        print(f"[Epoch {epoch:03d}] Train BCE={avg_loss:.4f}, Val AUC={auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return train_losses, val_aucs


def test_model_classification(model, test_loader) -> float:
    """
    Evaluate the model on the test set for a classification task.

    Returns
    -------
    auc : float
        ROC-AUC score on the test set.
    """
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            embs = [encoder(batch) for encoder in model.encoders]
            embs = torch.stack(embs, dim=1)
            embs = _maybe_flatten_for_concat(model, embs)
            gfeat = _get_global_features(batch, embs.device)

            out = model.fusion(embs, global_features=gfeat)
            logits = out[0] if isinstance(out, tuple) else out
            preds.append(logits.cpu())
            labs.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds) if preds else torch.empty(0)
    labs = torch.cat(labs) if labs else torch.empty(0)

    if preds.numel() == 0:
        return 0.5

    probs = torch.sigmoid(preds).numpy()
    try:
        auc = roc_auc_score(labs.numpy(), probs)
    except Exception:
        auc = 0.5
    return auc


# -------------------------------------------------------------------------
# Training & evaluation (regression)
# -------------------------------------------------------------------------
def train_and_validate_regression(
    model,
    train_loader,
    val_loader,
    epochs: int,
    weight_decay: float = 0.0,
):
    """
    Train and validate the model for a regression task using MSELoss.

    Returns
    -------
    train_losses : list of float
        Per-epoch training losses.
    val_mses : list of float
        Per-epoch validation MSE values.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    mse = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=PATIENCE_LR
    )

    train_losses, val_mses = [], []
    best_val_mse = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(epochs):
        # --------- training ---------
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            embs = [encoder(batch) for encoder in model.encoders]
            embs = torch.stack(embs, dim=1)
            embs = _maybe_flatten_for_concat(model, embs)
            gfeat = _get_global_features(batch, embs.device)

            out = model.fusion(embs, global_features=gfeat)
            pred = out[0] if isinstance(out, tuple) else out
            label = batch.y.view(-1).float().to(device)

            loss = mse(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_loss)

        # --------- validation ---------
        model.eval()
        with torch.no_grad():
            preds, labs = [], []
            for batch in val_loader:
                batch = batch.to(device)
                embs = [encoder(batch) for encoder in model.encoders]
                embs = torch.stack(embs, dim=1)
                embs = _maybe_flatten_for_concat(model, embs)
                gfeat = _get_global_features(batch, embs.device)

                out = model.fusion(embs, global_features=gfeat)
                pred = out[0] if isinstance(out, tuple) else out
                preds.append(pred.cpu())
                labs.append(batch.y.view(-1).cpu())

            preds = torch.cat(preds) if preds else torch.empty(0)
            labs = torch.cat(labs) if labs else torch.empty(0)
            mse_val = mse(preds, labs).item() if preds.numel() > 0 else float("inf")

        val_mses.append(mse_val)
        scheduler.step(mse_val)

        if mse_val + 1e-8 < best_val_mse:
            best_val_mse = mse_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_ES:
                print(f"[Early Stop] epoch={epoch}, best_val_mse={best_val_mse:.4f}")
                break

        print(f"[Epoch {epoch:03d}] Train MSE={avg_loss:.4f}, Val MSE={mse_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return train_losses, val_mses


def test_model_regression(model, test_loader) -> float:
    """
    Evaluate the model on the test set for a regression task.

    Returns
    -------
    rmse : float
        Root Mean Squared Error (RMSE) on the test set.
    """
    mse = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        preds, labs = [], []
        for batch in test_loader:
            batch = batch.to(device)
            embs = [encoder(batch) for encoder in model.encoders]
            embs = torch.stack(embs, dim=1)
            embs = _maybe_flatten_for_concat(model, embs)
            gfeat = _get_global_features(batch, embs.device)

            out = model.fusion(embs, global_features=gfeat)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu())
            labs.append(batch.y.view(-1).cpu())

        preds = torch.cat(preds) if preds else torch.empty(0)
        labs = torch.cat(labs) if labs else torch.empty(0)

        mse_val = mse(preds, labs).item() if preds.numel() > 0 else float("inf")
    rmse = float(np.sqrt(mse_val)) if mse_val < float("inf") else float("inf")
    return rmse


# -------------------------------------------------------------------------
# Plotting utility
# -------------------------------------------------------------------------
def plot_training_curve(
    train_values: List[float],
    val_values: List[float],
    metric_name: str,
    title: str,
    out_path: str,
):
    """
    Plot training vs validation curves and save them as a PNG.

    Parameters
    ----------
    train_values : list of float
        Training metric per epoch (e.g., loss).
    val_values : list of float
        Validation metric per epoch (e.g., AUC or MSE).
    metric_name : str
        Name of the validation metric (for y-axis label).
    title : str
        Figure title.
    out_path : str
        Path to save the figure.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-task fine-tuning for PRISM multi-encoder fusion (WeightedFusion)."
    )

    # Task / data configuration
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Type of downstream task.",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="Path to the dataset CSV file (must contain 'smiles' and label column).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        required=True,
        help="Name of the label column in the CSV file.",
    )

    # Pretrained encoders & space subset
    parser.add_argument(
        "--encoder_ckpts",
        type=str,
        required=True,
        help=(
            "Comma-separated list of encoder checkpoint paths. "
            "Example: 'path_to_task0.pth,path_to_task1.pth,path_to_task2.pth,path_to_task_none.pth'"
        ),
    )
    parser.add_argument(
        "--space_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated indices specifying which encoders to use (subset of [0..N-1]). "
            "If omitted, all encoders given by --encoder_ckpts are used."
        ),
    )

    # Training hyperparameters (per-run)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the fusion module.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for fine-tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data used for validation (scaffold split).",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data used for test (scaffold split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for splits and the first run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs (with different seeds) to estimate mean±std.",
    )

    # Global optimization hyperparameters (override module-level defaults)
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
        default=20,
        help="Early-stopping patience in epochs (default: 20).",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=10,
        help="LR scheduler patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="LR decay factor for ReduceLROnPlateau (default: 0.5).",
    )

    # Output configuration
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/finetune_single",
        help="Directory where results (plots and summary) will be saved.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional experiment name (used only for logging / filenames).",
    )

    parser.add_argument("--use_global_features", action="store_true")
    parser.add_argument("--fusion_type", choices=["auto", "static", "adaptive"], default="auto",
                        help="auto uses static fusion for classification and adaptive fusion for regression.")
    return parser.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    # Override global hyperparameters with CLI values
    global LEARNING_RATE, EMB_DIM, PATIENCE_ES, PATIENCE_LR, LR_FACTOR
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

    # Parse encoder checkpoint list
    all_ckpts = [p.strip() for p in args.encoder_ckpts.split(",") if p.strip()]
    if len(all_ckpts) == 0:
        raise ValueError("No encoder checkpoints provided via --encoder_ckpts.")

    # Parse subset indices
    if args.space_indices is None:
        subset_indices = list(range(len(all_ckpts)))
    else:
        subset_indices = [int(s.strip()) for s in args.space_indices.split(",") if s.strip()]
    if len(subset_indices) == 0:
        raise ValueError("Empty subset specified via --space_indices.")

    # Build the actual subset of checkpoints
    ckpt_paths_subset = [all_ckpts[i] for i in subset_indices]

    # Derive a human-readable name for this subset (e.g., "0_1_2_3")
    subset_name = "_".join(str(i) for i in subset_indices)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Using encoders with indices: {subset_indices}")
    print(f"[info] Checkpoints used in this run:")
    for i, p in zip(subset_indices, ckpt_paths_subset):
        print(f"  - index {i}: {p}")

    metric_runs = []  # AUC (classification) or RMSE (regression) per run
    summary_records = []

    base_dataset_name = os.path.splitext(os.path.basename(args.data_csv))[0]
    exp_tag = args.exp_name if args.exp_name else base_dataset_name

    for run_id in range(args.runs):
        run_seed = args.seed + run_id
        print(f"\n===== Run {run_id + 1}/{args.runs} (seed={run_seed}) =====")

        # Set seeds for reproducibility
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        # Load data with scaffold split using this run's seed
        train_loader, val_loader, test_loader, smiles_list = load_data(
            csv_path=args.data_csv,
            label_col=args.label_col,
            batch_size=args.batch_size,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=run_seed,
            task_type=args.task_type,
        )

        # Use the first training sample to infer feature dimensions for encoders
        if len(train_loader.dataset) == 0:
            raise RuntimeError("Empty training dataset: cannot proceed.")
        sample = train_loader.dataset[0]

        # Build the fine-tuning model (encoders + fusion head)
        fusion_type = args.fusion_type
        if fusion_type == "auto":
            fusion_type = "static" if args.task_type == "classification" else "adaptive"
        model = get_finetune_model(sample, dropout=args.dropout, ckpt_paths=ckpt_paths_subset,
                                   fusion_type=fusion_type, use_global_features=args.use_global_features)

        # Train & validate
        if args.task_type == "classification":
            train_vals, val_vals = train_and_validate_classification(
                model,
                train_loader,
                val_loader,
                epochs=args.epochs,
                weight_decay=args.weight_decay,
            )
            metric_name = "AUC"
            test_metric = test_model_classification(model, test_loader)
        else:
            train_vals, val_vals = train_and_validate_regression(
                model,
                train_loader,
                val_loader,
                epochs=args.epochs,
                weight_decay=args.weight_decay,
            )
            metric_name = "MSE"
            test_metric = test_model_regression(model, test_loader)

        metric_runs.append(test_metric)

        # Plot training curve only for the first run (to avoid clutter)
        if run_id == 0:
            curve_path = os.path.join(
                args.out_dir,
                f"{exp_tag}_{subset_name}_train_curve_run{run_id + 1}.png",
            )
            plot_title = f"{exp_tag} | subset={subset_name} | task={args.task_type}"
            plot_training_curve(
                train_values=train_vals,
                val_values=val_vals,
                metric_name=metric_name,
                title=plot_title,
                out_path=curve_path,
            )

        summary_records.append(
            {
                "run_id": run_id,
                "seed": run_seed,
                "task_type": args.task_type,
                "dataset": base_dataset_name,
                "subset": subset_name,
                "metric_name": "AUC" if args.task_type == "classification" else "RMSE",
                "test_metric": test_metric,
            }
        )
        print(
            f"[Run {run_id + 1}] Test {summary_records[-1]['metric_name']}"
            f" = {test_metric:.4f}"
        )

    # Aggregate over runs
    metric_mean = float(np.mean(metric_runs))
    metric_std = float(np.std(metric_runs))

    print("\n===== Final summary over runs =====")
    if args.task_type == "classification":
        print(f"AUC = {metric_mean:.4f} ± {metric_std:.4f}")
    else:
        print(f"RMSE = {metric_mean:.4f} ± {metric_std:.4f}")

    # Save detailed per-run results and overall summary
    df_runs = pd.DataFrame(summary_records)
    runs_path = os.path.join(
        args.out_dir, f"{exp_tag}_{subset_name}_runs_{args.task_type}.csv"
    )
    df_runs.to_csv(runs_path, index=False)

    summary_path = os.path.join(
        args.out_dir, f"{exp_tag}_{subset_name}_summary_{args.task_type}.csv"
    )
    pd.DataFrame(
        [
            {
                "task_type": args.task_type,
                "dataset": base_dataset_name,
                "subset": subset_name,
                "metric_name": "AUC" if args.task_type == "classification" else "RMSE",
                "metric_mean": metric_mean,
                "metric_std": metric_std,
                "runs": args.runs,
            }
        ]
    ).to_csv(summary_path, index=False)

    print(f"\n[done] Detailed runs saved to: {runs_path}")
    print(f"[done] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
