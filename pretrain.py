#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-run pretraining script for ChemSpace-style GNN encoder.

Main features:
- Read a CSV file with a `smiles` column.
- For each SMILES, compute three VSA descriptor vectors:
    * SMR_VSA1-10
    * SlogP_VSA1-10
    * PEOE_VSA1-14
- Convert (SMILES, VSA vectors) into graph data via `smiles2graph`.
- Filter out empty graphs (no nodes or edges).
- Train a single GNNModelWithNewLoss instance with a simple train/val split.
- Log tau_phys statistics at each epoch via a runtime monkey-patch:
    * Wrap `_soft_labels_from_distance` to record tau_phys.
    * Expose `begin_epoch()` / `end_epoch_stats()` hooks on the model.
- Save artifacts into `save_dir`:
    * config.json
    * history.csv
    * tau_phys_epoch.csv
    * best_model.pth
    * training_curve.png

This script is intended as a minimal, reproducible pretraining entry point
for research code (e.g., for publication and open-sourcing on GitHub).

python pretrain_single.py \
  --data_csv ./data/zinc_smiles.csv \
  --save_dir mpnn_pretrain/zinc_single \
  --num_epochs 200 \
  --property_index none \
  --tau 0.1 \
  --tau_phys_scale 0.5

"""

import os
import json
import csv
import time
from typing import Dict, Any, Iterable, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import MethodType
import argparse

from torch.utils.data import DataLoader as TorchDataLoader, random_split

# Project-specific imports (adjust paths according to your repository layout)
from rdkit import Chem
from rdkit.Chem import Descriptors

from model.getdata import smiles2graph
from model.mpn_vas_st import GNNModelWithNewLoss


# =============================================================================
# Dataset utilities
# =============================================================================

class FilterEmptyGraphs(torch.utils.data.Dataset):
    """
    Torch Dataset wrapper that filters out graphs with:
      - no node features (x is None or has 0 rows), or
      - no edges (edge_index is None or has 0 columns).

    This is useful when graph construction can occasionally fail or
    produce empty graphs that would break the training loop.
    """
    def __init__(self, dataset: Iterable):
        self.filtered_dataset: List[Any] = [
            d for d in dataset
            if getattr(d, "x", None) is not None and d.x.size(0) > 0
            and getattr(d, "edge_index", None) is not None and d.edge_index.size(1) > 0
        ]

    def __len__(self) -> int:
        return len(self.filtered_dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.filtered_dataset[idx]


def calculate_vsa_vectors(smiles: str):
    """
    Compute three types of VSA descriptor vectors for a given SMILES.

    Returns
    -------
    smr : List[float] or None
        SMR_VSA1-10 values.
    slogp : List[float] or None
        SlogP_VSA1-10 values.
    peoe : List[float] or None
        PEOE_VSA1-14 values.

    If the SMILES cannot be parsed or any descriptor evaluation fails,
    (None, None, None) is returned and the molecule should be skipped.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    try:
        smr = [float(getattr(Descriptors, f"SMR_VSA{i}")(mol)) for i in range(1, 11)]
        slogp = [float(getattr(Descriptors, f"SlogP_VSA{i}")(mol)) for i in range(1, 11)]
        peoe = [float(getattr(Descriptors, f"PEOE_VSA{i}")(mol)) for i in range(1, 15)]
        return smr, slogp, peoe
    except Exception:
        # Any failure in descriptor computation leads to dropping this molecule.
        return None, None, None


def load_graph_dataset_with_vsa(csv_path: str):
    """
    Load a CSV file with a 'smiles' column and build a list of graph objects
    with associated VSA-based properties.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file. The file must contain a column
        named 'smiles' or 'SMILES'.

    Returns
    -------
    data_list : List[torch_geometric.data.Data]
        List of graph objects created by `smiles2graph`.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Be robust to 'smiles' vs 'SMILES'
    smiles_col = None
    for cand in ["smiles", "SMILES"]:
        if cand in df.columns:
            smiles_col = cand
            break
    if smiles_col is None:
        raise ValueError("Input CSV must contain a 'smiles' or 'SMILES' column.")

    smiles_list: List[str] = []
    properties: List[Tuple[List[float], List[float], List[float]]] = []

    n_total = len(df)
    n_failed = 0

    print(f"[data] Reading SMILES and computing VSA descriptors from: {csv_path}")
    for smi in tqdm(df[smiles_col].tolist(), desc="Computing VSA descriptors"):
        smr, slogp, peoe = calculate_vsa_vectors(smi)
        if smr is None:
            n_failed += 1
            continue
        smiles_list.append(smi)
        properties.append((smr, slogp, peoe))

    print(f"[data] Total rows: {n_total}, valid molecules: {len(smiles_list)}, "
          f"failed VSA computations: {n_failed}")

    # Convert SMILES + properties to graph data using your project utility
    data_list = smiles2graph(smiles_list, properties=properties)
    print(f"[data] Constructed {len(data_list)} graph objects.")

    if not data_list:
        raise RuntimeError("No valid graph data was constructed. "
                           "Check input file and descriptor computation.")

    print("[data] Example Data object:", data_list[0])
    return data_list


# =============================================================================
# Monkey-patch: enable tau_phys logging (no changes to model file)
# =============================================================================

def enable_tau_logging(model):
    """
    Enable per-epoch tau_phys statistics logging on the given model instance.

    This function assumes the model class defines a method:
        _soft_labels_from_distance(self, D, *args, **kwargs)
    which returns a tuple (Y, tau_val), where tau_val is a scalar tensor
    (or a Python float) representing the physical temperature parameter.

    At runtime, we:
      1. Wrap `_soft_labels_from_distance` to record all tau_val values
         into `model._epoch_tau_phys_samples`.
      2. Attach two helper hooks to the instance:
           - model.begin_epoch()
           - model.end_epoch_stats()
         which respectively reset and summarize the collected tau_phys values.
    """
    if not hasattr(model.__class__, "_soft_labels_from_distance"):
        print("[warn] model class has no _soft_labels_from_distance; "
              "tau_phys logging disabled.")
        return

    model._epoch_tau_phys_samples = []
    model._tau_phys_epoch_stats = []

    # Unbound original method from the class, so we can re-bind it with extra logic
    orig_unbound = model.__class__._soft_labels_from_distance

    def _wrapped_soft_labels_from_distance(self, D, *args, **kwargs):
        """
        Wrapper that forwards to the original method and records tau_phys
        for each call during the current epoch.
        """
        Y, tau_val = orig_unbound(self, D, *args, **kwargs)
        try:
            # tau_val may be a tensor; we try our best to get a Python float
            v = float(getattr(tau_val, "detach", lambda: tau_val)().item())
        except Exception:
            v = float(tau_val)
        self._epoch_tau_phys_samples.append(v)
        return Y, tau_val

    # Bind wrapper to the instance
    model._soft_labels_from_distance = MethodType(_wrapped_soft_labels_from_distance, model)

    def _begin_epoch(self):
        """Reset tau_phys samples at the beginning of each epoch."""
        self._epoch_tau_phys_samples = []

    def _end_epoch_stats(self) -> Dict[str, float]:
        """
        Aggregate tau_phys samples into summary statistics and store them
        in `model._tau_phys_epoch_stats`.
        """
        arr = self._epoch_tau_phys_samples or []
        if not arr:
            stats = dict(
                tau_phys_mean=float("nan"),
                tau_phys_median=float("nan"),
                tau_phys_min=float("nan"),
                tau_phys_max=float("nan"),
                tau_phys_count=0.0,
            )
        else:
            t = torch.tensor(arr, dtype=torch.float32)
            stats = dict(
                tau_phys_mean=float(t.mean().item()),
                tau_phys_median=float(t.median().item()),
                tau_phys_min=float(t.min().item()),
                tau_phys_max=float(t.max().item()),
                tau_phys_count=float(t.numel()),
            )
        model._tau_phys_epoch_stats.append(stats)
        return stats

    model.begin_epoch = MethodType(_begin_epoch, model)
    model.end_epoch_stats = MethodType(_end_epoch_stats, model)


# =============================================================================
# Training loop
# =============================================================================

def train_with_list_batches(
    model,
    data_list,
    save_path: str,
    num_epochs: int = 1000,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    patience: int = 50,
    batch_size: int = 128,
    best_val_loss_all: float = float("inf"),
    exp_name: Optional[str] = None,
) -> float:
    """
    Train a single model instance on a (train, val) split of the given data list.

    Notes
    -----
    - This function assumes that `model.get_loss(batch)` is defined and returns
      a scalar loss tensor.
    - `batch` is a list of PyG Data objects (we keep the list structure and let
      the model handle batching internally).
    - The model itself is responsible for moving data to devices (CPU/GPU).

    Parameters
    ----------
    model : nn.Module
        A model instance with a `get_loss` method.
    data_list : List[torch_geometric.data.Data]
        Full dataset as a list of graph objects.
    save_path : str
        Directory where all artifacts of this run will be stored.
    num_epochs : int, optional
        Maximum number of training epochs.
    lr : float, optional
        Initial learning rate.
    weight_decay : float, optional
        Weight decay for the Adam optimizer.
    patience : int, optional
        Early stopping patience in epochs.
    batch_size : int, optional
        Batch size (in terms of number of Data objects).
    best_val_loss_all : float, optional
        Global threshold for saving the best model. If you only have one run,
        just leave it at +inf.
    exp_name : str, optional
        Optional tag for this experiment, stored in config.json.

    Returns
    -------
    best_val_loss : float
        Best validation loss achieved during training.
    """
    print(f"[train] Saving artifacts to: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Persist one-off config (includes tau & tau_phys_scale if present)
    config = {
        "exp_name": exp_name or "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "batch_size": batch_size,
        "tau": float(getattr(model, "tau", 0.1)),
        "tau_phys": None
        if getattr(model, "tau_phys", None) is None
        else float(model.tau_phys),
        "tau_phys_scale": float(getattr(model, "tau_phys_scale", 0.5)),
        "property_index": getattr(model, "property_index", None),
        "hidden_dim": getattr(model, "hidden_dim", None),
        "dropout_rate": getattr(model, "dropout_rate", None),
        "cov_num": getattr(model, "cov_num", None),
    }
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Dataset & loaders
    ds = FilterEmptyGraphs(data_list)
    n_train = int(0.8 * len(ds))
    train_set, val_set = random_split(ds, [n_train, len(ds) - n_train])

    # Keep the structure as list[Data]; let the model handle batching
    collate = lambda xs: xs
    train_loader = TorchDataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate
    )
    val_loader = TorchDataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(patience // 2, 1)
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    # CSV logs
    hist_csv = os.path.join(save_path, "history.csv")
    tau_csv = os.path.join(save_path, "tau_phys_epoch.csv")
    with open(hist_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])
    with open(tau_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "tau_phys_mean", "tau_phys_median",
             "tau_phys_min", "tau_phys_max", "tau_phys_count"]
        )

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        if hasattr(model, "begin_epoch"):
            model.begin_epoch()

        # --------- train phase ---------
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.get_loss(batch)
            if (not torch.isfinite(loss)) or (not loss.requires_grad):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1

        avg_train_loss = total_loss / max(steps, 1)
        train_losses.append(avg_train_loss)

        # --------- validation phase ---------
        model.eval()
        val_loss, vsteps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                l = model.get_loss(batch)
                if torch.isfinite(l):
                    val_loss += float(l.item())
                    vsteps += 1
        avg_val_loss = val_loss / max(vsteps, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        # --------- tau_phys stats ---------
        if hasattr(model, "end_epoch_stats"):
            stats = model.end_epoch_stats()
        else:
            stats = dict(
                tau_phys_mean=float("nan"),
                tau_phys_median=float("nan"),
                tau_phys_min=float("nan"),
                tau_phys_max=float("nan"),
                tau_phys_count=0.0,
            )

        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, lr_now])
        with open(tau_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    stats["tau_phys_mean"],
                    stats["tau_phys_median"],
                    stats["tau_phys_min"],
                    stats["tau_phys_max"],
                    stats["tau_phys_count"],
                ]
            )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"LR {lr_now:.2e} | tau_phys(mean) {stats['tau_phys_mean']:.4g}"
        )

        # --------- early stopping & best model saving ---------
        if avg_val_loss < best_val_loss and avg_val_loss < best_val_loss_all:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "encoder_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                os.path.join(save_path, "best_model.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[train] Early stopping at epoch {epoch}")
                break

    # --------- plot training curve with baseline log N ---------
    def avg_logN(dloader):
        Ns = [max(len(batch) - 1, 1) for batch in dloader]
        return float(np.log(np.mean(Ns))) if Ns else 0.0

    train_eval_loader = TorchDataLoader(
        train_set, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    val_eval_loader = TorchDataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    baseline_train = avg_logN(train_eval_loader)
    baseline_val = avg_logN(val_eval_loader)

    if val_losses:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.axhline(
            y=baseline_train,
            linestyle="--",
            linewidth=1.5,
            label=f"Train Baseline (log N)≈{baseline_train:.3f}",
        )
        plt.axhline(
            y=baseline_val,
            linestyle=":",
            linewidth=1.5,
            label=f"Val Baseline (log N)≈{baseline_val:.3f}",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Process (Best Val Loss: {best_val_loss:.4f})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "training_curve.png"))
        plt.close()

    return best_val_loss


# =============================================================================
# Main entry point
# =============================================================================

def build_model(
    atom_dim: int,
    bond_dim: int,
    num_global_features: int,
    property_index: Optional[int],
    hidden_dim: int,
    cov_num: int,
    dropout_rate: float,
    tau: float,
    tau_phys_scale: float,
    save_path: str,
):
    """
    Construct a fresh GNNModelWithNewLoss instance and attach tau-related
    attributes as well as tau_phys logging hooks.
    """
    model = GNNModelWithNewLoss(
        num_node_features=atom_dim,
        num_edge_features=bond_dim,
        num_global_features=num_global_features,
        hidden_dim=hidden_dim,
        cov_num=cov_num,
        dropout_rate=dropout_rate,
        property_index=property_index,  # 0, 1, 2, or None
        save_path=save_path,
    )

    # Optional support for multi-task setups in the original project.
    if hasattr(model, "add_task"):
        model.add_task(property_index=property_index, save_path=save_path)

    # Attach tau-related attributes expected by the loss function and config logger.
    model.tau = float(tau)
    model.tau_phys_scale = float(tau_phys_scale)
    # tau_phys itself is typically updated inside the model; we leave it as None
    if not hasattr(model, "tau_phys"):
        model.tau_phys = None

    # Enable runtime logging of tau_phys statistics
    enable_tau_logging(model)

    return model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pretraining script.
    """
    parser = argparse.ArgumentParser(
        description="Single-run pretraining script for ChemSpace-style GNN encoder."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="./data/vsa_zinc_25k.csv",
        help="Path to the input CSV file (must contain 'smiles' column).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="mpnn_pretrain",
        help="Directory to store all artifacts of this run.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (number of graphs per batch).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (in epochs).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of the GNN encoder.",
    )
    parser.add_argument(
        "--cov_num",
        type=int,
        default=3,
        help="Number of covariance channels / message-passing steps (project-specific).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in the GNN encoder.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Logits temperature hyperparameter used in the contrastive loss.",
    )
    parser.add_argument(
        "--tau_phys_scale",
        type=float,
        default=0.5,
        help="Scaling factor for the physical temperature (tau_phys).",
    )
    parser.add_argument(
        "--property_index",
        type=str,
        default="0",
        help=(
            "Which VSA channel to treat as supervised property index. "
            "Options: '0', '1', '2', or 'none'."
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional experiment name stored in config.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Map property_index argument to an integer or None
    if args.property_index.lower() in ["none", "-1"]:
        property_index = None
    else:
        property_index = int(args.property_index)

    # 1) Load graphs from the input CSV (with on-the-fly VSA computation)
    data_list = load_graph_dataset_with_vsa(args.data_csv)

    # 2) Infer feature dimensions from the first Data object
    example = data_list[0]
    atom_dim = example.x.shape[1]
    bond_dim = example.edge_attr.shape[1] if getattr(example, "edge_attr", None) is not None else 0
    num_global_features = getattr(example, "global_features", torch.zeros(1, 0)).shape[1]

    print(f"[info] atom_dim={atom_dim}, bond_dim={bond_dim}, "
          f"num_global_features={num_global_features}")

    # 3) Build a fresh model
    model = build_model(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        num_global_features=num_global_features,
        property_index=property_index,
        hidden_dim=args.hidden_dim,
        cov_num=args.cov_num,
        dropout_rate=args.dropout,
        tau=args.tau,
        tau_phys_scale=args.tau_phys_scale,
        save_path=args.save_dir,
    )

    # 4) Train the model
    train_kwargs = dict(
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        batch_size=args.batch_size,
        best_val_loss_all=float("inf"),  # single run
        exp_name=args.exp_name,
    )

    best_val_loss = train_with_list_batches(
        model=model,
        data_list=data_list,
        save_path=args.save_dir,
        **train_kwargs,
    )

    print(f"[done] Training finished. Best validation loss: {best_val_loss:.6f}")
    print(f"[done] All artifacts saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
