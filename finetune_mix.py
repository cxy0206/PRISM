#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture solvation training / evaluation (BMixSolvationPredictor).

This script consolidates the code you pasted into a single runnable .py file.

It supports two main modes:

1) mix_runs
   - Run multiple mixture experiments with two splits and (att / noatt) modes.
   - Initialize from a single-solvent checkpoint (e.g. ckpt/leeds.pth).

2) mix_ablation
   - Mixture ablation that loads your saved Leeds ablation checkpoints:
       ablation_solvation/subset_{name}/model_best.pth
   - Builds a BMix model with the same subset of ChemSpace encoders,
     loads the corresponding checkpoint, then trains/evals on mixture data.

Assumptions (match your project):
- model.getdata.smiles2graph exists
- finetune_pair.load_pretrained_encoders, finetune_pair._stack_encode, finetune_pair.TempPairMLP exist
- model.fusion.WeightedFusion_new exists
"""

import os
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
import matplotlib.pyplot as plt

from torch_geometric.data import Batch

# ---- your project imports ----
from model.getdata import smiles2graph
from finetune_pair import load_pretrained_encoders, _stack_encode, TempPairMLP
from model.fusion import WeightedFusion_new as WeightedFusion


# ============================================================
# utils
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pct_within_1_np(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return float((np.abs(preds - labels) <= 1.0).mean())


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_filename(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "")
        .replace("?", "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace('"', "")
        .replace("'", "")
        .strip()
    )


def unwrap_subset_dataset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


# ============================================================
# RatioGate + RSCrossAttention
# ============================================================
class RatioGate(nn.Module):
    """
    Very small ratio gate:
        ratio -> g in [0, ~0.25 * scale]
    using g = scale * x(1-x)  where x is clamped ratio in [0,1].
    """
    def __init__(self, ratio_dim: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, ratio: torch.Tensor) -> torch.Tensor:
        if ratio.dim() == 1:
            ratio = ratio.unsqueeze(-1)
        x = torch.clamp(ratio, 0.0, 1.0)
        base = x * (1.0 - x)
        g = self.scale * base
        return g


class RSCrossAttention(nn.Module):
    """
    Simplified Ratio-aware Symmetric Cross-Attention.
    - No Q/K/V projections; tokens are used directly.
    - Always uses softmax.
    """
    def __init__(
        self,
        d_model: int,
        use_entmax: bool = False,   # kept for compatibility
        ratio_dim: int = 1,
        ratio_hidden_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.ratio_gate = RatioGate(ratio_dim=ratio_dim, hidden_dim=ratio_hidden_dim)

    def _cross_once(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor, ratio: torch.Tensor):
        B, La, D = tokens_a.shape
        B2, Lb, D2 = tokens_b.shape
        assert B == B2, "batch size mismatch between A and B"
        assert D == D2 == self.d_model, "embedding dim mismatch"

        Qa, Ka, Va = tokens_a, tokens_a, tokens_a
        Qb, Kb, Vb = tokens_b, tokens_b, tokens_b

        cross_a, attn_a2b = self._cross_once(Qa, Kb, Vb)
        cross_b, attn_b2a = self._cross_once(Qb, Ka, Va)

        g = self.ratio_gate(ratio)  # (B,1)
        g_expanded = g.view(B, 1, 1)

        out_a = (1.0 - g_expanded) * tokens_a + g_expanded * cross_a
        out_b = (1.0 - g_expanded) * tokens_b + g_expanded * cross_b

        return out_a, out_b, attn_a2b, attn_b2a, g


# ============================================================
# BMixSolvationPredictor
# ============================================================
class BMixSolvationPredictor(nn.Module):
    def __init__(
        self,
        sol_encs: List[nn.Module],
        solv_encs: List[nn.Module],
        sol_fusion: nn.Module,
        solv_fusion: nn.Module,
        combiner: TempPairMLP,
        d_model: int,
        use_entmax: bool = False,
        ratio_dim: int = 1,
        ratio_hidden_dim: int = 16,
        use_attn_for_residual: bool = True,
    ):
        super().__init__()
        self.sol_encs = nn.ModuleList(sol_encs)
        self.solv_encs = nn.ModuleList(solv_encs)
        self.sol_f = sol_fusion
        self.solv_f = solv_fusion
        self.combine = combiner

        self.attends = RSCrossAttention(
            d_model=d_model,
            use_entmax=use_entmax,
            ratio_dim=ratio_dim,
            ratio_hidden_dim=ratio_hidden_dim,
        )

        self.cross_scale = nn.Parameter(torch.tensor(0.0))
        self.space_head_s1 = nn.Linear(d_model, 1)
        self.space_head_s2 = nn.Linear(d_model, 1)
        self.use_attn_for_residual = use_attn_for_residual

    @staticmethod
    def _fuse(fusion: nn.Module, embs: torch.Tensor, g: Optional[torch.Tensor]):
        out = fusion(embs, global_features=g)  # WeightedFusion_new signature in your project
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        sol_batch: Batch,
        solv1_batch: Batch,
        solv2_batch: Batch,
        ratio_batch: torch.Tensor,
        temps: torch.Tensor,
    ):
        # 1) Encode solute / solvents in ChemSpace (per-space embeddings: [B,K,D])
        z1_spaces = _stack_encode(self.sol_encs, sol_batch)
        z2_0 = _stack_encode(self.solv_encs, solv1_batch)
        z3_0 = _stack_encode(self.solv_encs, solv2_batch)

        g1 = getattr(sol_batch, "global_features", None)
        g2 = getattr(solv1_batch, "global_features", None)
        g3 = getattr(solv2_batch, "global_features", None)

        z1 = self._fuse(self.sol_f, z1_spaces, g1)  # (B,D)
        z2 = self._fuse(self.solv_f, z2_0, g2)      # (B,D)
        z3 = self._fuse(self.solv_f, z3_0, g3)      # (B,D)

        # 2) Ideal mixing in embedding space
        ratio = ratio_batch.view(-1, 1)
        zv = ratio * z2 + (1.0 - ratio) * z3  # (B,D)

        y_ideal = self.combine(z1, zv, temps)
        if y_ideal.dim() == 1:
            y_ideal = y_ideal.view(-1, 1)

        # 3) ChemSpace x ChemSpace interaction matrix
        a = self.space_head_s1(z2_0).squeeze(-1)  # (B,K)
        b = self.space_head_s2(z3_0).squeeze(-1)  # (B,K)
        base_M = a.unsqueeze(-1) * b.unsqueeze(-2)  # (B,K,K)

        if self.use_attn_for_residual:
            _, _, attn_a2b, _, g = self.attends(z2_0, z3_0, ratio_batch)
            B = base_M.size(0)
            g_expanded = g.view(B, 1, 1)
            M = attn_a2b * base_M * g_expanded
        else:
            M = base_M

        # 4) Scalar residual
        lambda_cross = torch.exp(self.cross_scale)
        dy = lambda_cross * M.sum(dim=(-2, -1), keepdim=True)
        dy = dy.view(dy.size(0), 1)

        pred = y_ideal + dy
        return pred, z1, zv, M


# ============================================================
# build model helper
# ============================================================
def build_mix_model_subset(
    sample_s,
    sample_v,
    ckpt_paths: List[str],
    emb_dim: int,
    solute_dropout: float,
    solvent_dropout: float,
    mlp_hidden: Tuple[int, ...],
    mlp_dropout: float,
    t_embed_dim: int,
    device: torch.device,
    use_entmax: bool = False,
    ratio_dim: int = 1,
    ratio_hidden_dim: int = 16,
    use_attention: bool = True,
) -> BMixSolvationPredictor:
    """
    Build a BMixSolvationPredictor using a subset of pre-trained encoders.
    """
    sol_encs = load_pretrained_encoders(sample_s, ckpt_paths)
    solv_encs = load_pretrained_encoders(sample_v, ckpt_paths)

    sol_f = WeightedFusion(len(ckpt_paths), emb_dim, dropout=solute_dropout, pair=True).to(device)
    solv_f = WeightedFusion(len(ckpt_paths), emb_dim, dropout=solvent_dropout, pair=True).to(device)

    # probe fused dim (z_dim) for TempPairMLP
    with torch.no_grad():
        b1 = Batch.from_data_list([sample_s]).to(device)
        b2 = Batch.from_data_list([sample_v]).to(device)

        z1 = _stack_encode(sol_encs, b1)  # (1,K,D)
        z2 = _stack_encode(solv_encs, b2) # (1,K,D)

        g1 = getattr(b1, "global_features", None)
        g2 = getattr(b2, "global_features", None)

        z1 = BMixSolvationPredictor._fuse(sol_f, z1, g1)
        z2 = BMixSolvationPredictor._fuse(solv_f, z2, g2)

        z_dim = int(z1.shape[-1])

    comb = TempPairMLP(
        z_dim,
        t_embed_dim=t_embed_dim,
        hidden=mlp_hidden,
        dropout=mlp_dropout,
    ).to(device)

    model = BMixSolvationPredictor(
        sol_encs=sol_encs,
        solv_encs=solv_encs,
        sol_fusion=sol_f,
        solv_fusion=solv_f,
        combiner=comb,
        d_model=emb_dim,
        use_entmax=use_entmax,
        ratio_dim=ratio_dim,
        ratio_hidden_dim=ratio_hidden_dim,
        use_attn_for_residual=use_attention,
    ).to(device)
    return model


# ============================================================
# mixture dataset & loaders
# ============================================================
def smiles_graph_inputs_from_mixture_csv(csv_path: str):
    """
    Expected columns:
      solute_smiles, solvent1_smiles, solvent2_smiles,
      temperature, logS, Xsolvent1, Xsolvent2
    """
    df = pd.read_csv(csv_path)

    required = [
        "solute_smiles",
        "solvent1_smiles",
        "solvent2_smiles",
        "temperature",
        "logS",
        "Xsolvent1",
        "Xsolvent2",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")

    solute_smiles = df["solute_smiles"].astype(str).tolist()
    solv1_smiles = df["solvent1_smiles"].astype(str).tolist()
    solv2_smiles = df["solvent2_smiles"].astype(str).tolist()

    y = df["logS"].astype(float).to_numpy()
    T = df["temperature"].astype(float).to_numpy()
    frac1 = df["Xsolvent1"].astype(float).to_numpy()
    frac2 = df["Xsolvent2"].astype(float).to_numpy()

    d1 = smiles2graph(solute_smiles, y=y, properties=T)
    d2_1 = smiles2graph(solv1_smiles, y=y, properties=T)
    d2_2 = smiles2graph(solv2_smiles, y=y, properties=T)

    return d1, d2_1, d2_2, y, T, frac1, frac2, solute_smiles, solv1_smiles, solv2_smiles


class MixtureDataset(Dataset):
    """
    Each sample returns:
      (solute_graph, solvent1_graph, solvent2_graph,
       temp, label, frac1, frac2,
       solute_smiles, solv1_smiles, solv2_smiles)
    """
    def __init__(
        self,
        s, v1, v2,
        temps, labels,
        frac1, frac2,
        solute_smiles, solv1_smiles, solv2_smiles,
    ):
        n = len(labels)
        assert len(s) == len(v1) == len(v2) == len(temps) == len(frac1) == len(frac2) == n
        assert len(solute_smiles) == len(solv1_smiles) == len(solv2_smiles) == n

        self.s = s
        self.v1 = v1
        self.v2 = v2
        self.temps = temps
        self.labels = labels
        self.frac1 = frac1
        self.frac2 = frac2
        self.solute_smiles = solute_smiles
        self.solv1_smiles = solv1_smiles
        self.solv2_smiles = solv2_smiles

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.s[idx],
            self.v1[idx],
            self.v2[idx],
            float(self.temps[idx]),
            float(self.labels[idx]),
            float(self.frac1[idx]),
            float(self.frac2[idx]),
            self.solute_smiles[idx],
            self.solv1_smiles[idx],
            self.solv2_smiles[idx],
        )


def collate_mixture(batch):
    (
        sol, solv1, solv2,
        temps, labels,
        frac1, frac2,
        solute_smiles, solv1_smiles, solv2_smiles,
    ) = zip(*batch)

    b1 = Batch.from_data_list(list(sol))
    b2 = Batch.from_data_list(list(solv1))
    b3 = Batch.from_data_list(list(solv2))

    temps_t = torch.tensor(temps, dtype=torch.float32).view(-1, 1)
    labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    w1 = torch.tensor(frac1, dtype=torch.float32).view(-1, 1)
    w2 = torch.tensor(frac2, dtype=torch.float32).view(-1, 1)

    return (
        b1, b2, b3,
        temps_t, labels_t,
        w1, w2,
        list(solute_smiles), list(solv1_smiles), list(solv2_smiles),
    )


def split_indices_by_solvent_pair(
    solv1_smiles: List[str],
    solv2_smiles: List[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int = 42,
    canonicalize_pair: bool = True,
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "train/val/test 比例之和必须为 1"
    n = len(solv1_smiles)
    assert len(solv2_smiles) == n

    keys = []
    for s1, s2 in zip(solv1_smiles, solv2_smiles):
        if canonicalize_pair:
            key = tuple(sorted((str(s1), str(s2))))
        else:
            key = (str(s1), str(s2))
        keys.append(key)

    unique_pairs = list(set(keys))
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_pairs)

    n_pairs = len(unique_pairs)
    n_train = int(round(train_frac * n_pairs))
    n_val = int(round(val_frac * n_pairs))
    n_train = max(0, min(n_pairs, n_train))
    n_val = max(0, min(n_pairs - n_train, n_val))

    train_pairs = set(unique_pairs[:n_train])
    val_pairs = set(unique_pairs[n_train:n_train + n_val])
    test_pairs = set(unique_pairs[n_train + n_val:])

    idx_train, idx_val, idx_test = [], [], []
    for i, key in enumerate(keys):
        if key in train_pairs:
            idx_train.append(i)
        elif key in val_pairs:
            idx_val.append(i)
        else:
            idx_test.append(i)
    return idx_train, idx_val, idx_test


def make_mixture_loaders_by_solvent_pair(
    csv_path: str,
    batch_size: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int = 42,
    canonicalize_pair: bool = True,
    num_workers: int = 0,
):
    (
        d1, d2_1, d2_2,
        y, T, frac1, frac2,
        solute_smiles, solv1_smiles, solv2_smiles
    ) = smiles_graph_inputs_from_mixture_csv(csv_path)

    dataset = MixtureDataset(
        d1, d2_1, d2_2,
        T, y,
        frac1, frac2,
        solute_smiles, solv1_smiles, solv2_smiles
    )

    idx_train, idx_val, idx_test = split_indices_by_solvent_pair(
        solv1_smiles,
        solv2_smiles,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        canonicalize_pair=canonicalize_pair,
    )

    train_loader = DataLoader(
        Subset(dataset, idx_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_mixture,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, idx_val),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mixture,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, idx_test),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mixture,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# checkpoint loading helpers
# ============================================================
def load_checkpoint_into_bmix(
    model: BMixSolvationPredictor,
    ckpt_path: str,
    device: torch.device,
    freeze_encoders: bool = True,
    freeze_fusions: bool = True,
    freeze_combiner: bool = True,
):
    """
    Load a checkpoint into BMix, supporting both naming conventions:

    A) single-solvent checkpoint keys:
        solute_encs.*, solvent_encs.*,
        solute_fusion.*, solvent_fusion.*,
        combiner.*

    B) BMix/SolvationPredictor style keys:
        sol_encs.*, solv_encs.*, sol_f.*, solv_f.*, combine.*

    BMix-specific keys (attends/cross_scale/space_head_*) are absent -> remain random init.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # detect whether remapping is needed
    needs_remap = any(k.startswith("solute_encs.") or k.startswith("solvent_encs.") or
                      k.startswith("solute_fusion.") or k.startswith("solvent_fusion.") or
                      k.startswith("combiner.")
                      for k in state.keys())

    if needs_remap:
        remapped = {}
        for k, v in state.items():
            new_k = None
            if k.startswith("solute_encs."):
                new_k = k.replace("solute_encs", "sol_encs", 1)
            elif k.startswith("solvent_encs."):
                new_k = k.replace("solvent_encs", "solv_encs", 1)
            elif k.startswith("solute_fusion."):
                new_k = k.replace("solute_fusion", "sol_f", 1)
            elif k.startswith("solvent_fusion."):
                new_k = k.replace("solvent_fusion", "solv_f", 1)
            elif k.startswith("combiner."):
                new_k = k.replace("combiner", "combine", 1)

            if new_k is not None:
                remapped[new_k] = v
        state_to_load = remapped
    else:
        state_to_load = state

    missing, unexpected = model.load_state_dict(state_to_load, strict=False)
    print(f"[load_checkpoint_into_bmix] loaded: {ckpt_path}")
    print("  missing keys:", missing)
    print("  unexpected keys:", unexpected)

    freeze_prefixes: List[str] = []
    if freeze_encoders:
        freeze_prefixes += ["sol_encs.", "solv_encs."]
    if freeze_fusions:
        freeze_prefixes += ["sol_f.", "solv_f."]
    if freeze_combiner:
        freeze_prefixes += ["combine."]

    n_frozen = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in freeze_prefixes):
            p.requires_grad = False
            n_frozen += 1
    print(f"[load_checkpoint_into_bmix] frozen params: {n_frozen}")
    return model


# ============================================================
# train + eval
# ============================================================
def train_and_eval_mix_model(
    model: BMixSolvationPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    out_dir: str,
    device: torch.device,
    patience: int = 0,
    save_best_ckpt: bool = True,
):
    """
    Train on mixture train_loader, select best by val MSE, then evaluate on test_loader.
    Saves:
      - plots/train_val_loss_curve.png
      - plots/test_scatter.png
      - test_predictions.csv
      - M_heatmaps/*.png (mean M per solvent-pair & ratio bin)
      - training_stats.npz
      - (optional) model_best.pth
    """
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    m_dir = os.path.join(out_dir, "M_heatmaps")
    ensure_dir(plots_dir)
    ensure_dir(m_dir)

    model = model.to(device)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        # ----- train -----
        model.train()
        run_loss = 0.0
        n_samples = 0

        for (sol_b, solv1_b, solv2_b, temps, labels, w1, w2,
             solute_smiles, solv1_smiles, solv2_smiles) in train_loader:
            sol_b = sol_b.to(device)
            solv1_b = solv1_b.to(device)
            solv2_b = solv2_b.to(device)
            temps = temps.to(device)
            labels = labels.to(device)
            ratio = w1.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred, _, _, _ = model(sol_b, solv1_b, solv2_b, ratio, temps)
            if pred.dim() == 1:
                pred = pred.view(-1, 1)

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item()) * labels.size(0)
            n_samples += labels.size(0)

        tr = run_loss / max(1, n_samples)
        train_losses.append(tr)

        # ----- val -----
        model.eval()
        run_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for (sol_b, solv1_b, solv2_b, temps, labels, w1, w2,
                 solute_smiles, solv1_smiles, solv2_smiles) in val_loader:
                sol_b = sol_b.to(device)
                solv1_b = solv1_b.to(device)
                solv2_b = solv2_b.to(device)
                temps = temps.to(device)
                labels = labels.to(device)
                ratio = w1.to(device)

                pred, _, _, _ = model(sol_b, solv1_b, solv2_b, ratio, temps)
                if pred.dim() == 1:
                    pred = pred.view(-1, 1)

                loss = criterion(pred, labels)
                run_loss += float(loss.item()) * labels.size(0)
                n_samples += labels.size(0)

        va = run_loss / max(1, n_samples)
        val_losses.append(va)

        print(f"Epoch {epoch:03d} | Train MSE: {tr:.6f} | Val MSE: {va:.6f}")

        # best tracking + early stop
        if va < best_val - 1e-12:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"[EarlyStop] patience={patience} reached at epoch {epoch}. Best val={best_val:.6f}")
                break

    # save loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train MSE")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "train_val_loss_curve.png"), dpi=300)
    plt.close()

    # load best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    if save_best_ckpt:
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(out_dir, "model_best.pth"))

    # ===== test =====
    model.eval()
    all_preds, all_labels = [], []
    all_temps, all_w1, all_w2 = [], [], []
    all_solute_smiles, all_s1_smiles, all_s2_smiles = [], [], []

    # M heatmap dict: M_dict[(s1,s2)][ratio_bin] -> list of (K,K)
    M_dict: Dict[Tuple[str, str], Dict[float, List[np.ndarray]]] = {}

    with torch.no_grad():
        for (sol_b, solv1_b, solv2_b, temps, labels, w1, w2,
             solute_smiles, solv1_smiles, solv2_smiles) in test_loader:
            sol_b = sol_b.to(device)
            solv1_b = solv1_b.to(device)
            solv2_b = solv2_b.to(device)
            temps_gpu = temps.to(device)
            ratio = w1.to(device)

            pred, _, _, M = model(sol_b, solv1_b, solv2_b, ratio, temps_gpu)
            if pred.dim() == 1:
                pred = pred.view(-1, 1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.numpy())
            all_temps.append(temps.numpy())
            all_w1.append(w1.numpy())
            all_w2.append(w2.numpy())
            all_solute_smiles.extend(solute_smiles)
            all_s1_smiles.extend(solv1_smiles)
            all_s2_smiles.extend(solv2_smiles)

            M_np = M.cpu().numpy()
            ratio_np = w1.numpy().reshape(-1)
            for i in range(M_np.shape[0]):
                key = (solv1_smiles[i], solv2_smiles[i])
                r = float(ratio_np[i])
                r_key = round(r, 3)
                M_dict.setdefault(key, {})
                M_dict[key].setdefault(r_key, [])
                M_dict[key][r_key].append(M_np[i])

    all_preds = np.concatenate(all_preds, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
    all_temps = np.concatenate(all_temps, axis=0).reshape(-1)
    all_w1 = np.concatenate(all_w1, axis=0).reshape(-1)
    all_w2 = np.concatenate(all_w2, axis=0).reshape(-1)

    mse = float(((all_preds - all_labels) ** 2).mean())
    rmse = float(np.sqrt(mse))
    pct1 = pct_within_1_np(all_preds, all_labels)

    print(f"[Test] RMSE: {rmse:.6f}")
    print(f"[Test] pct(|err|<=1): {pct1*100:.2f}%")

    # scatter
    plt.figure()
    x = all_labels
    y = all_preds
    plt.scatter(x, y, s=15, alpha=0.6)
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    plt.plot(x_line, x_line, "k--", linewidth=1.0, label="y=x")
    plt.fill_between(x_line, x_line - 1.0, x_line + 1.0, alpha=0.2, label="|y-x|<=1")
    plt.xlabel("True logS")
    plt.ylabel("Predicted logS")
    plt.title(f"RMSE={rmse:.3f}, pct<=1={pct1*100:.1f}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "test_scatter.png"), dpi=300)
    plt.close()

    # predictions csv
    df_pred = pd.DataFrame({
        "solute_smiles": all_solute_smiles,
        "solvent1_smiles": all_s1_smiles,
        "solvent2_smiles": all_s2_smiles,
        "Xsolvent1": all_w1,
        "Xsolvent2": all_w2,
        "temperature": all_temps,
        "logS_true": all_labels,
        "logS_pred": all_preds,
        "abs_err": np.abs(all_preds - all_labels),
    })
    df_pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # M heatmaps (mean)
    for (s1, s2), ratio_map in M_dict.items():
        s1_safe = safe_filename(s1)
        s2_safe = safe_filename(s2)
        for r_key, mats in ratio_map.items():
            mats_np = np.stack(mats, axis=0)
            mean_M = mats_np.mean(axis=0)

            plt.figure()
            plt.imshow(mean_M, origin="lower", aspect="equal")
            plt.colorbar(label="M value")
            K = mean_M.shape[0]
            plt.xticks(range(K), [f"sp{j}" for j in range(K)])
            plt.yticks(range(K), [f"sp{i}" for i in range(K)])
            plt.xlabel("solvent2 ChemSpaces (B)")
            plt.ylabel("solvent1 ChemSpaces (A)")
            plt.title(f"{s1}\n{s2}\nXsolvent1≈{r_key:.3f}")
            plt.tight_layout()

            fname = f"{s1_safe}__{s2_safe}__x1_{r_key:.3f}.png"
            plt.savefig(os.path.join(m_dir, fname), dpi=300)
            plt.close()

    np.savez(
        os.path.join(out_dir, "training_stats.npz"),
        train_losses=np.array(train_losses, dtype=float),
        val_losses=np.array(val_losses, dtype=float),
        test_rmse=rmse,
        test_pct_le1=pct1,
        best_val_mse=best_val,
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_rmse": rmse,
        "test_pct_le1": pct1,
        "best_val_mse": best_val,
    }


# ============================================================
# experiment runners
# ============================================================
def make_ckpt_paths_from_pretrain(pretrain_base_dir: str, task_dirs: List[str], ckpt_name: str) -> List[str]:
    ckpts = []
    for td in task_dirs:
        p = os.path.join(pretrain_base_dir, td, ckpt_name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing encoder ckpt: {p}")
        ckpts.append(p)
    return ckpts


def get_sample_graphs_from_mixture_loader(train_loader: DataLoader):
    base_ds = unwrap_subset_dataset(train_loader.dataset)
    sample = base_ds[0]
    return sample[0], sample[1]  # solute graph, solvent1 graph


def run_all_mix_experiments(
    mix_csv: str,
    ckpt_paths: List[str],
    leeds_init_ckpt: str,
    emb_dim: int,
    out_root: str,
    batch_size: int,
    num_runs: int,
    device: torch.device,
    num_epochs: int,
    lr: float,
    patience: int,
    solute_dropout: float,
    solvent_dropout: float,
    mlp_hidden: Tuple[int, ...],
    mlp_dropout: float,
    t_embed_dim: int,
):
    """
    Runs 4 experiments x num_runs:
      - split 0.3/0.1/0.6 and 0.8/0.1/0.1
      - for each split: attention=True and attention=False
    """
    ensure_dir(out_root)
    split_configs = [
        ("0.3_0.1_0.6", 0.3, 0.1, 0.6),
        ("0.8_0.1_0.1", 0.8, 0.1, 0.1),
    ]

    all_stats: Dict[str, Dict[str, Any]] = {}

    for split_name, train_frac, val_frac, test_frac in split_configs:
        print("\n" + "=" * 80)
        print(f"Data split: train/val/test = {train_frac}/{val_frac}/{test_frac}")
        print("=" * 80)

        for run_idx in range(num_runs):
            seed = 42 + run_idx
            set_seed(seed)

            train_loader, val_loader, test_loader = make_mixture_loaders_by_solvent_pair(
                csv_path=mix_csv,
                batch_size=batch_size,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                random_state=seed,
            )

            sample_s, sample_v = get_sample_graphs_from_mixture_loader(train_loader)

            for use_attention in (True, False):
                tag = "att" if use_attention else "noatt"
                exp_name = f"{tag}_{split_name}"
                if exp_name not in all_stats:
                    all_stats[exp_name] = {"runs": [], "rmse": [], "pct": []}

                print(f"\n[{exp_name}] Run {run_idx+1}/{num_runs} (seed={seed}, use_attention={use_attention})")

                model = build_mix_model_subset(
                    sample_s=sample_s,
                    sample_v=sample_v,
                    ckpt_paths=ckpt_paths,
                    emb_dim=emb_dim,
                    solute_dropout=solute_dropout,
                    solvent_dropout=solvent_dropout,
                    mlp_hidden=mlp_hidden,
                    mlp_dropout=mlp_dropout,
                    t_embed_dim=t_embed_dim,
                    device=device,
                    use_entmax=False,
                    ratio_dim=1,
                    ratio_hidden_dim=16,
                    use_attention=use_attention,
                )

                model = load_checkpoint_into_bmix(
                    model,
                    ckpt_path=leeds_init_ckpt,
                    device=device,
                    freeze_encoders=True,
                    freeze_fusions=True,
                    freeze_combiner=False,  # usually you want to train combiner a bit
                )

                out_dir = os.path.join(out_root, exp_name, f"run_{run_idx+1}")
                stats = train_and_eval_mix_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    num_epochs=num_epochs,
                    lr=lr,
                    out_dir=out_dir,
                    device=device,
                    patience=patience,
                    save_best_ckpt=True,
                )

                all_stats[exp_name]["runs"].append(stats)
                all_stats[exp_name]["rmse"].append(stats["test_rmse"])
                all_stats[exp_name]["pct"].append(stats["test_pct_le1"])

    # summarize
    print("\n" + "=" * 80)
    print("Summary over experiments (mean / var across runs)")
    print("=" * 80)

    for exp_name, rec in all_stats.items():
        rmse_arr = np.array(rec["rmse"], dtype=float)
        pct_arr = np.array(rec["pct"], dtype=float)
        rec["rmse_mean"] = float(rmse_arr.mean())
        rec["rmse_var"] = float(rmse_arr.var())
        rec["pct_mean"] = float(pct_arr.mean())
        rec["pct_var"] = float(pct_arr.var())

        print(f"\n[Experiment: {exp_name}]")
        print(f"  RMSE runs      : {rmse_arr}")
        print(f"  RMSE mean/var  : {rec['rmse_mean']:.4f} / {rec['rmse_var']:.6f}")
        print(f"  pct<=1 runs    : {pct_arr}")
        print(f"  pct<=1 mean/var: {rec['pct_mean']*100:.2f}% / {rec['pct_var']*10000:.4f} (in %^2)")

    # save json summary
    with open(os.path.join(out_root, "summary_all_stats.json"), "w", encoding="utf-8") as f:
        json_safe = {}
        for k, v in all_stats.items():
            json_safe[k] = {
                "rmse": v["rmse"],
                "pct": v["pct"],
                "rmse_mean": v["rmse_mean"],
                "rmse_var": v["rmse_var"],
                "pct_mean": v["pct_mean"],
                "pct_var": v["pct_var"],
            }
        import json
        json.dump(json_safe, f, indent=2)

    return all_stats


def run_mix_ablation(
    mix_csv: str,
    ablation_dir: str,
    subset_names: List[str],
    pretrain_base_dir: str,
    task_dirs: List[str],
    ckpt_name: str,
    out_root: str,
    leeds_ckpt_name: str,
    emb_dim: int,
    batch_size: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int,
    device: torch.device,
    use_attention: bool,
    num_epochs: int,
    lr: float,
    patience: int,
    solute_dropout: float,
    solvent_dropout: float,
    mlp_hidden: Tuple[int, ...],
    mlp_dropout: float,
    t_embed_dim: int,
    freeze_encoders: bool,
    freeze_fusions: bool,
    freeze_combiner: bool,
):
    """
    Mixture ablation (typically ATT only) using your saved Leeds ablation checkpoints.
    """
    ensure_dir(out_root)

    train_loader, val_loader, test_loader = make_mixture_loaders_by_solvent_pair(
        csv_path=mix_csv,
        batch_size=batch_size,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        canonicalize_pair=True,
    )
    sample_s, sample_v = get_sample_graphs_from_mixture_loader(train_loader)
    ckpt_paths_all = make_ckpt_paths_from_pretrain(pretrain_base_dir, task_dirs, ckpt_name)

    rows = []
    for subset_name in subset_names:
        idxs = [int(x) for x in subset_name.split("_") if x.strip() != ""]
        if any(i < 0 or i >= len(ckpt_paths_all) for i in idxs):
            raise ValueError(f"subset '{subset_name}' has indices out of range 0..{len(ckpt_paths_all)-1}")

        ckpt_paths_sub = [ckpt_paths_all[i] for i in idxs]

        print("\n" + "=" * 90)
        print(f"[Mixture Ablation] subset={subset_name} idxs={idxs} split={train_frac}/{val_frac}/{test_frac}")
        print("=" * 90)

        model = build_mix_model_subset(
            sample_s=sample_s,
            sample_v=sample_v,
            ckpt_paths=ckpt_paths_sub,
            emb_dim=emb_dim,
            solute_dropout=solute_dropout,
            solvent_dropout=solvent_dropout,
            mlp_hidden=mlp_hidden,
            mlp_dropout=mlp_dropout,
            t_embed_dim=t_embed_dim,
            device=device,
            use_entmax=False,
            ratio_dim=1,
            ratio_hidden_dim=16,
            use_attention=use_attention,
        )

        leeds_ckpt = os.path.join(ablation_dir, f"subset_{subset_name}", leeds_ckpt_name)
        if not os.path.isfile(leeds_ckpt):
            raise FileNotFoundError(f"Missing leeds ablation ckpt: {leeds_ckpt}")

        model = load_checkpoint_into_bmix(
            model,
            ckpt_path=leeds_ckpt,
            device=device,
            freeze_encoders=freeze_encoders,
            freeze_fusions=freeze_fusions,
            freeze_combiner=freeze_combiner,
        )

        out_dir = os.path.join(out_root, f"subset_{subset_name}")
        stats = train_and_eval_mix_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            lr=lr,
            out_dir=out_dir,
            device=device,
            patience=patience,
            save_best_ckpt=True,
        )

        rows.append({
            "subset": subset_name,
            "idxs": "_".join(map(str, idxs)),
            "split": f"{train_frac}/{val_frac}/{test_frac}",
            "use_attention": bool(use_attention),
            "mix_test_rmse": stats["test_rmse"],
            "mix_test_pct_le1": stats["test_pct_le1"],
            "leeds_init_ckpt": leeds_ckpt,
            "out_dir": out_dir,
            "freeze_encoders": bool(freeze_encoders),
            "freeze_fusions": bool(freeze_fusions),
            "freeze_combiner": bool(freeze_combiner),
        })

    df = pd.DataFrame(rows).sort_values("mix_test_rmse", ascending=True)
    summary_path = os.path.join(out_root, "summary_mix_ablation.csv")
    df.to_csv(summary_path, index=False)
    print("\n✅ Done. Summary saved to:", summary_path)
    print(df)
    return df


# ============================================================
# main
# ============================================================
def parse_tuple_int(s: str) -> Tuple[int, ...]:
    # e.g. "1024,128,64" -> (1024,128,64)
    if s.strip() == "":
        return tuple()
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def main():
    parser = argparse.ArgumentParser(description="BMix mixture solvation runner (consolidated).")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---------- common args ----------
    def add_common(p):
        p.add_argument("--mix_csv", type=str, default="mix_noaq.csv")
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--emb_dim", type=int, default=512)
        p.add_argument("--num_workers", type=int, default=0)

        p.add_argument("--solute_dropout", type=float, default=0.1)
        p.add_argument("--solvent_dropout", type=float, default=0.1)
        p.add_argument("--mlp_hidden", type=str, default="512,512")
        p.add_argument("--mlp_dropout", type=float, default=0.1)
        p.add_argument("--t_embed_dim", type=int, default=16)

        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--patience", type=int, default=0)

        p.add_argument("--seed", type=int, default=2025)

    # ---------- mix_runs ----------
    p1 = sub.add_parser("mix_runs", help="Run repeated mixture experiments (two splits, att/noatt).")
    add_common(p1)
    p1.add_argument("--out_root", type=str, default="mix_result_solv")
    p1.add_argument("--num_runs", type=int, default=3)

    p1.add_argument("--pretrain_base_dir", type=str, default="ckpt/tau_0.1__scale_0.25")
    p1.add_argument("--task_dirs", type=str, default="task_0,task_1,task_2,task_none")
    p1.add_argument("--encoder_ckpt_name", type=str, default="best_model.pth")

    p1.add_argument("--leeds_init_ckpt", type=str, default="ckpt/leeds.pth")

    # ---------- mix_ablation ----------
    p2 = sub.add_parser("mix_ablation", help="Run mixture ablation using saved Leeds ablation checkpoints.")
    add_common(p2)
    p2.add_argument("--out_root", type=str, default="mix_abl_results")
    p2.add_argument("--train_frac", type=float, default=0.3)
    p2.add_argument("--val_frac", type=float, default=0.1)
    p2.add_argument("--test_frac", type=float, default=0.6)
    p2.add_argument("--ablation_dir", type=str, default="ablation_solvation")
    p2.add_argument("--leeds_ckpt_name", type=str, default="model_best.pth")

    p2.add_argument("--pretrain_base_dir", type=str, default="ckpt/tau_0.1__scale_0.25")
    p2.add_argument("--task_dirs", type=str, default="task_0,task_1,task_2,task_none")
    p2.add_argument("--encoder_ckpt_name", type=str, default="best_model.pth")

    p2.add_argument("--subsets", type=str, default="0_1_2_3",
                    help="Comma-separated subset names, e.g. '0_1_2_3,0_1_2,0_1_3'")

    p2.add_argument("--use_attention", action="store_true", help="Use attention in residual (ATT mode).")
    p2.add_argument("--no_attention", action="store_true", help="Force no-attention (light mode).")

    p2.add_argument("--freeze_encoders", action="store_true", help="Freeze encoders when training on mixture.")
    p2.add_argument("--freeze_fusions", action="store_true", help="Freeze fusion modules when training on mixture.")
    p2.add_argument("--freeze_combiner", action="store_true", help="Freeze combiner(TempPairMLP) when training on mixture.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    mlp_hidden = parse_tuple_int(args.mlp_hidden)
    task_dirs = [x.strip() for x in args.task_dirs.split(",") if x.strip() != ""]

    if args.mode == "mix_runs":
        ckpt_paths = make_ckpt_paths_from_pretrain(args.pretrain_base_dir, task_dirs, args.encoder_ckpt_name)
        run_all_mix_experiments(
            mix_csv=args.mix_csv,
            ckpt_paths=ckpt_paths,
            leeds_init_ckpt=args.leeds_init_ckpt,
            emb_dim=args.emb_dim,
            out_root=args.out_root,
            batch_size=args.batch_size,
            num_runs=args.num_runs,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            solute_dropout=args.solute_dropout,
            solvent_dropout=args.solvent_dropout,
            mlp_hidden=mlp_hidden,
            mlp_dropout=args.mlp_dropout,
            t_embed_dim=args.t_embed_dim,
        )

    elif args.mode == "mix_ablation":
        subsets = [x.strip() for x in args.subsets.split(",") if x.strip() != ""]

        if args.no_attention:
            use_attention = False
        elif args.use_attention:
            use_attention = True
        else:
            use_attention = True  # default ATT on

        run_mix_ablation(
            mix_csv=args.mix_csv,
            ablation_dir=args.ablation_dir,
            subset_names=subsets,
            pretrain_base_dir=args.pretrain_base_dir,
            task_dirs=task_dirs,
            ckpt_name=args.encoder_ckpt_name,
            out_root=args.out_root,
            leeds_ckpt_name=args.leeds_ckpt_name,
            emb_dim=args.emb_dim,
            batch_size=args.batch_size,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=args.seed,
            device=device,
            use_attention=use_attention,
            num_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            solute_dropout=args.solute_dropout,
            solvent_dropout=args.solvent_dropout,
            mlp_hidden=mlp_hidden,
            mlp_dropout=args.mlp_dropout,
            t_embed_dim=args.t_embed_dim,
            freeze_encoders=args.freeze_encoders,
            freeze_fusions=args.freeze_fusions,
            freeze_combiner=args.freeze_combiner,
        )
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
