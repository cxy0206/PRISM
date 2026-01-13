#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(Updated) Multi-task classification pipeline for:
  - sider (27 tasks)
  - clintox (2 tasks)
  - tox21 (12 tasks)
  - muv (17 tasks)

Your new requirements:
1) First scan ckpt/ for all (tau, scale) pretrained folders, and select the BEST config
   on each dataset by validation mean ROC-AUC (all4 spaces + static fusion).
2) Then, under that best config, run:
   - Our method (all4, pretrained, static fusion)
   - Drop-one-space ablation (pretrained, static fusion)
   - No-pretraining ablation (scratch Chemprop MPN encoders, static fusion)
   - Aggregation ablation: WeightedFusion (static) vs WeightedFusion_new (adaptive)
   - Baselines: VSA+FP classical (gboost/rf/mlp) + MPN+VSA+FP

Compatibility with your CURRENT code:
- smiles2graph creates Data.global_features with dim=8
- WeightedFusion forward requires (embs, global_features)
- WeightedFusion_new forward supports global_features optional, but we still pass it.

IMPORTANT: your fusion classes output 1 logit by default (nn.Linear(64,1)).
This script AUTOMATICALLY patches the last Linear to output num_tasks for multi-task datasets.
No need to modify model/fusion.py.

Files expected:
  data/sider.csv, data/clintox.csv, data/tox21.csv, data/muv.csv
  Each CSV: first column "smiles", remaining columns are task labels (0/1 or -1 for missing).
  ckpt/ contains folders like tau_0.1__scale_0.5/task_0/best_model.pth ... task_none/best_model.pth

Outputs:
  results_aug_new/multitask_cls_suite/
    - best_configs.json
    - selection_<dataset>.csv
    - summary_all.csv
"""

import os
import re
import json
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader as GeoDataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =========================================================
# Your project imports (must exist)
# =========================================================
from model.getdata import smiles2graph
from model.mpn_vas_st import GNNModelWithNewLoss as PretrainedEncoder
from model.fusion import WeightedFusion, WeightedFusion_new

# chemprop
from argparse import Namespace
try:
    from chemprop.models.mpn import MPN
except Exception as e:
    raise RuntimeError("Cannot import chemprop.models.mpn.MPN. Please install chemprop.") from e


# =========================================================
# Global config
# =========================================================
device = '1'
DATA_DIR = "data"
CKPT_ROOT = "ckpt"

OUT_ROOT = "results_aug_new/clintox"
os.makedirs(OUT_ROOT, exist_ok=True)

DATASETS = ["clintox"]

# fixed split seed (same split used for all runs)
SPLIT_SEED = 42

# run repeats (mean ± std). Split is fixed, only init/training randomness varies.
RUNS = 3

# model/training hyperparams
LEARNING_RATE = 1e-4
PATIENCE_ES = 20
PATIENCE_LR = 10
LR_FACTOR = 0.5
EMB_DIM = 512

# per-dataset training hyperparams 
DATASET_PARAMS = {
    "clintox":{"batch_size": 32, "epochs": 200, "dropout": 0.1, "weight_decay": 1e-5, "val_split": 0.1, "test_split": 0.1},
}

# spaces
SPACE_NAMES = ["task_0", "task_1", "task_2", "task_none"]
ALL4 = [0, 1, 2, 3]
DROP_ONE_SUBSETS = [
    [0, 1, 2],  # w/o task_none
    [0, 1, 3],  # w/o task_2
    [0, 2, 3],  # w/o task_1
    [1, 2, 3],  # w/o task_0
]

# Selection stage settings
SEARCH_FREEZE_ENCODERS = True         # freeze encoders during ckpt config selection (recommended)
SEARCH_EPOCHS_CAP = 200               # max epochs during selection (early stop still applies)
SEARCH_FUSION_KIND = "static"         # selection uses static fusion
SEARCH_SUBSET = ALL4                  # selection uses all4 spaces
SEARCH_SINGLE_RUN = True              # selection uses only 1 run (seed=42) for speed

# Ablation stage settings
FINETUNE_ENCODERS_IN_ABLATION = True  # during final runs, train encoders + fusion

# Baseline features
FP_BITS = 2048
FP_RADIUS = 2
VSA_DIMS = (10, 12, 14)
GLOBAL_DIM = sum(VSA_DIMS) + FP_BITS  # 2084


# =========================================================
# Seed util
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Scaffold split
# =========================================================
def generate_scaffold(smiles: str, include_chirality: bool = True) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)

def scaffold_split_indices(smiles_list: List[str], val_split: float, test_split: float, seed: int):
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

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


# =========================================================
# Multi-task labels: y + mask
# =========================================================
def coerce_label_matrix(df: pd.DataFrame, label_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    y: [N,T] float32 with missing filled as 0
    mask: [N,T] float32, 1 if label present else 0
    Missing: NaN or -1
    """
    y = df[label_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    mask = np.ones_like(y, dtype=np.float32)

    nan_mask = np.isnan(y)
    mask[nan_mask] = 0.0

    minus1_mask = (y == -1.0)
    mask[minus1_mask] = 0.0

    y = np.nan_to_num(y, nan=0.0)
    y[minus1_mask] = 0.0

    # If labels are not strictly {0,1}, threshold at 0.5 on valid entries
    valid_vals = y[mask > 0]
    if valid_vals.size > 0:
        uniq = np.unique(valid_vals)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            y = (y > 0.5).astype(np.float32)

    return y.astype(np.float32), mask.astype(np.float32)


# =========================================================
# Dataset wrapper
# =========================================================
class MoleculeListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self._data = list(data_list)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(idx, (list, tuple, np.ndarray)):
            return MoleculeListDataset([self._data[i] for i in idx])
        return self._data[idx]


def load_multitask_dataset(ds_name: str, batch_size: int, val_split: float, test_split: float, seed: int):
    """
    Uses YOUR smiles2graph (global_features dim=8).
    Adds .y_mask per graph.
    """
    path = os.path.join(DATA_DIR, f"{ds_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"{ds_name}.csv must have a 'smiles' column.")

    label_cols = [c for c in df.columns if c != "smiles"]
    smiles_all = df["smiles"].astype(str).tolist()

    # pre-filter invalid smiles to keep alignment stable
    keep = []
    for i, smi in enumerate(smiles_all):
        if Chem.MolFromSmiles(smi) is not None:
            keep.append(i)
    if len(keep) < len(smiles_all):
        print(f"[{ds_name}] drop invalid smiles: {len(smiles_all)-len(keep)} / {len(smiles_all)}")

    df = df.iloc[keep].reset_index(drop=True)
    smiles = df["smiles"].astype(str).tolist()

    y, m = coerce_label_matrix(df, label_cols)  # [N,T]
    N, T = y.shape

    # smiles2graph expects y list; each label row can be vector
    y_list = [y[i].tolist() for i in range(N)]
    data_list = smiles2graph(smiles, y=y_list)

    # smiles2graph might still skip some; realign if needed
    if len(data_list) != N:
        got_smiles = [getattr(g, "smiles", None) for g in data_list]
        got_smiles = [s for s in got_smiles if s is not None]
        smi_to_idx = {s: i for i, s in enumerate(smiles)}
        keep2 = [smi_to_idx[s] for s in got_smiles if s in smi_to_idx]
        keep2 = np.array(keep2, dtype=np.int64)
        smiles = [smiles[i] for i in keep2.tolist()]
        y = y[keep2]
        m = m[keep2]
        N, T = y.shape
        print(f"[{ds_name}] WARNING: smiles2graph skipped graphs; keep={N}")

    # attach y_mask
    for i, g in enumerate(data_list):
        g.y = torch.tensor(y[i], dtype=torch.float32).view(1, -1)        # [1,T]
        g.y_mask = torch.tensor(m[i], dtype=torch.float32).view(1, -1)   # [1,T]

    tr_idx, vl_idx, te_idx = scaffold_split_indices(smiles, val_split, test_split, seed)

    base = MoleculeListDataset(data_list)

    def subset(ds, idx):
        if idx.size == 0:
            return MoleculeListDataset([])
        return ds[torch.tensor(idx, dtype=torch.long)]

    tr_ds = subset(base, tr_idx)
    vl_ds = subset(base, vl_idx)
    te_ds = subset(base, te_idx)

    tr_loader = GeoDataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = GeoDataLoader(vl_ds, batch_size=batch_size, shuffle=False)
    te_loader = GeoDataLoader(te_ds, batch_size=batch_size, shuffle=False)

    return {
        "smiles": smiles,
        "label_cols": label_cols,
        "num_tasks": T,
        "loaders": (tr_loader, vl_loader, te_loader),
        "splits": (tr_idx, vl_idx, te_idx),
    }


# =========================================================
# AUC metric (mean over tasks with defined AUC)
# =========================================================
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None

def multitask_mean_auc(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    T = y_true.shape[1]
    aucs = []
    for t in range(T):
        mt = mask[:, t] > 0.5
        if mt.sum() == 0:
            continue
        a = safe_auc(y_true[mt, t], y_prob[mt, t])
        if a is not None:
            aucs.append(a)
    if len(aucs) == 0:
        return 0.5, 0
    return float(np.mean(aucs)), int(len(aucs))


# =========================================================
# Find ckpt configs under ckpt/
# =========================================================
def find_ckpt_configs(ckpt_root: str) -> List[Dict]:
    """
    Return list of configs:
      {"name": folder_name, "paths": [p0,p1,p2,p3]}  (must all exist)
    """
    configs = []
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"Missing ckpt root: {ckpt_root}")

    pat = re.compile(r"^tau_[0-9.]+__scale_[0-9.]+$")
    for fname in sorted(os.listdir(ckpt_root)):
        fpath = os.path.join(ckpt_root, fname)
        if not os.path.isdir(fpath):
            continue
        if not pat.match(fname):
            continue

        paths = []
        ok = True
        for sp in SPACE_NAMES:
            p = os.path.join(fpath, sp, "best_model.pth")
            if not os.path.exists(p):
                ok = False
                break
            paths.append(p)
        if ok:
            configs.append({"name": fname, "paths": paths})

    if len(configs) == 0:
        raise RuntimeError(
            f"No valid configs found under {ckpt_root}. "
            f"Expected folders like tau_0.1__scale_0.5 with task_0..task_none/best_model.pth"
        )
    print(f"[CKPT] Found {len(configs)} valid configs.")
    return configs


# =========================================================
# Model building utilities
# =========================================================
def patch_fusion_out_dim(fusion: nn.Module, num_tasks: int) -> nn.Module:
    """
    Your WeightedFusion / WeightedFusion_new ends with nn.Linear(64, 1).
    We replace the last Linear out_features -> num_tasks for multi-task.
    """
    if not hasattr(fusion, "mlp"):
        return fusion
    mlp = fusion.mlp
    if not isinstance(mlp, nn.Sequential):
        return fusion

    # find last Linear index
    last_linear_idx = None
    for i in range(len(mlp) - 1, -1, -1):
        if isinstance(mlp[i], nn.Linear):
            last_linear_idx = i
            break
    if last_linear_idx is None:
        return fusion

    in_features = mlp[last_linear_idx].in_features
    out_features = mlp[last_linear_idx].out_features
    if out_features == num_tasks:
        return fusion

    mlp[last_linear_idx] = nn.Linear(in_features, num_tasks)
    fusion.mlp = mlp
    return fusion


def encode_one(encoder: nn.Module, batch) -> torch.Tensor:
    out = encoder(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def load_pretrained_encoders(sample_graph, subset: List[int], ckpt_paths: List[str]) -> List[nn.Module]:
    encs = []
    for i in subset:
        p = ckpt_paths[i]
        enc = PretrainedEncoder(
            num_node_features=sample_graph.x.shape[1],
            num_edge_features=sample_graph.edge_attr.shape[1],
            num_global_features=0,
            cov_num=3,
            hidden_dim=EMB_DIM
        ).to(device)

        ckpt = torch.load(p, map_location=device)
        state = ckpt.get("encoder_state_dict", ckpt)
        enc.load_state_dict(state, strict=True)
        encs.append(enc)
    return encs


class ScratchMPNNEncoder(nn.Module):
    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_dim=512, depth=3, dropout=0.1):
        super().__init__()
        mpn_args = Namespace(
            hidden_size=int(hidden_dim),
            atom_messages=True,
            bias=True,
            depth=max(1, int(depth)),
            dropout=float(dropout),
            undirected=False,
            activation="ReLU",
            no_cache=False,
            features_only=False,
            use_input_features=False,
        )
        self.mpn = MPN(mpn_args, atom_fdim=atom_fdim, bond_fdim=bond_fdim)

    def forward(self, batch):
        if hasattr(batch, "to_data_list"):
            pyg_list = batch.to_data_list()
        else:
            pyg_list = [batch]
        try:
            out = self.mpn(prompt=False, batch=pyg_list)
        except TypeError:
            try:
                out = self.mpn(batch=pyg_list)
            except TypeError:
                out = self.mpn(pyg_list)
        return out


def build_scratch_encoders(sample_graph, subset: List[int], dropout: float) -> List[nn.Module]:
    atom_fdim = sample_graph.x.shape[1]
    bond_fdim = sample_graph.edge_attr.shape[1]
    all_enc = []
    for _ in range(4):
        all_enc.append(ScratchMPNNEncoder(atom_fdim, bond_fdim, hidden_dim=EMB_DIM, depth=3, dropout=dropout).to(device))
    return [all_enc[i] for i in subset]


class EncFusionMultiTask(nn.Module):
    def __init__(self, encoders: List[nn.Module], fusion: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fusion = fusion

    def forward(self, batch):
        # embs: [B,N,D]
        embs = [encode_one(enc, batch) for enc in self.encoders]
        embs = torch.stack(embs, dim=1)
        gf = getattr(batch, "global_features", None)
        out = self.fusion(embs, gf)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        # ensure [B,T]
        if logits.dim() == 1:
            logits = logits.view(-1, 1)
        return logits


# =========================================================
# Training / Eval (multitask)
# =========================================================
def compute_pos_weight_vector(train_loader, num_tasks: int) -> torch.Tensor:
    pos = np.zeros((num_tasks,), dtype=np.float64)
    neg = np.zeros((num_tasks,), dtype=np.float64)

    for batch in train_loader:
        y = batch.y
        m = batch.y_mask
        y = y.view(y.size(0), -1).numpy()
        m = m.view(m.size(0), -1).numpy() > 0.5

        for t in range(num_tasks):
            mt = m[:, t]
            if mt.sum() == 0:
                continue
            yt = y[mt, t]
            pos[t] += float((yt == 1).sum())
            neg[t] += float((yt == 0).sum())

    pos = np.maximum(pos, 1.0)
    pw = neg / pos
    pw = np.clip(pw, 1.0, 1000.0)
    return torch.tensor(pw, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_mean_auc(model: nn.Module, loader, num_tasks: int) -> Tuple[float, int]:
    model.eval()
    logits_all, y_all, m_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).detach().cpu()
        y = batch.y.view(batch.y.size(0), -1).detach().cpu()
        m = batch.y_mask.view(batch.y_mask.size(0), -1).detach().cpu()
        logits_all.append(logits)
        y_all.append(y)
        m_all.append(m)

    if len(logits_all) == 0:
        return 0.5, 0

    logits = torch.cat(logits_all, dim=0).numpy()
    y = torch.cat(y_all, dim=0).numpy()
    m = torch.cat(m_all, dim=0).numpy()

    prob = 1.0 / (1.0 + np.exp(-logits))
    return multitask_mean_auc(y, prob, m)


def train_multitask(
    model: nn.Module,
    train_loader,
    val_loader,
    num_tasks: int,
    epochs: int,
    weight_decay: float,
    lr: float = LEARNING_RATE,
):
    pos_weight = compute_pos_weight_vector(train_loader, num_tasks)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=PATIENCE_LR
    )

    best_val = -1e9
    best_state = None
    patience = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        denom_steps = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)  # [B,T]
            y = batch.y.view(batch.y.size(0), -1).float().to(device)
            m = batch.y_mask.view(batch.y_mask.size(0), -1).float().to(device)

            loss_mat = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight, reduction="none"
            )
            loss = (loss_mat * m).sum()
            denom = m.sum().item()
            if denom > 0:
                loss = loss / denom
            else:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            denom_steps += 1.0

        val_auc, val_valid = eval_mean_auc(model, val_loader, num_tasks)
        scheduler.step(val_auc)

        if val_auc > best_val + 1e-8:
            best_val = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_ES:
                break

        if ep % 10 == 0 or ep == epochs - 1:
            avg_loss = total_loss / max(denom_steps, 1.0)
            print(f"  [Ep {ep:03d}] train_loss={avg_loss:.4f} val_auc={val_auc:.4f} (valid_tasks={val_valid})")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.to(device)
    return best_val


# =========================================================
# Build fusion model (pretrained/scratch + static/adaptive)
# =========================================================
def build_fusion_model(
    sample_graph,
    num_tasks: int,
    subset: List[int],
    ckpt_paths: Optional[List[str]],
    mode: str,            # "pretrained" or "scratch"
    fusion_kind: str,     # "static" or "adaptive"
    dropout: float,
    finetune_encoders: bool,
):
    if mode == "pretrained":
        assert ckpt_paths is not None
        encoders = load_pretrained_encoders(sample_graph, subset, ckpt_paths)
    elif mode == "scratch":
        encoders = build_scratch_encoders(sample_graph, subset, dropout=dropout)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not finetune_encoders:
        for enc in encoders:
            for p in enc.parameters():
                p.requires_grad = False

    if fusion_kind == "static":
        fusion = WeightedFusion(num_inputs=len(subset), emb_dim=EMB_DIM, dropout=dropout).to(device)
    else:
        fusion = WeightedFusion_new(num_inputs=len(subset), emb_dim=EMB_DIM, dropout=dropout, global_in_dim=8).to(device)

    fusion = patch_fusion_out_dim(fusion, num_tasks=num_tasks).to(device)

    model = EncFusionMultiTask(encoders, fusion).to(device)
    return model


# =========================================================
# Run one experiment (train on train, select by val, report test)
# =========================================================
def run_one(
    loaders,
    num_tasks: int,
    subset: List[int],
    ckpt_paths: Optional[List[str]],
    mode: str,
    fusion_kind: str,
    dropout: float,
    weight_decay: float,
    epochs: int,
    finetune_encoders: bool,
    seed: int,
) -> Tuple[float, int]:
    set_seed(seed)
    tr_loader, vl_loader, te_loader = loaders
    sample = tr_loader.dataset[0]

    model = build_fusion_model(
        sample_graph=sample,
        num_tasks=num_tasks,
        subset=subset,
        ckpt_paths=ckpt_paths,
        mode=mode,
        fusion_kind=fusion_kind,
        dropout=dropout,
        finetune_encoders=finetune_encoders,
    )

    train_multitask(model, tr_loader, vl_loader, num_tasks=num_tasks, epochs=epochs, weight_decay=weight_decay, lr=LEARNING_RATE)
    test_auc, test_valid = eval_mean_auc(model, te_loader, num_tasks=num_tasks)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(test_auc), int(test_valid)


# =========================================================
# Selection: choose best (tau, scale) config by val AUC
# =========================================================
def select_best_config_for_dataset(
    ds_name: str,
    configs: List[Dict],
    loaders,
    num_tasks: int,
    dropout: float,
    weight_decay: float,
    epochs: int,
) -> Dict:
    """
    Evaluate each config on VAL mean AUC (train fusion; optionally freeze encoders).
    Returns best config dict.
    """
    tr_loader, vl_loader, te_loader = loaders
    sample = tr_loader.dataset[0]

    rows = []

    # use fewer epochs in selection
    sel_epochs = min(int(epochs), int(SEARCH_EPOCHS_CAP))

    for cfg in configs:
        ckpt_paths = cfg["paths"]

        # selection uses all4 + static fusion
        set_seed(42)
        model = build_fusion_model(
            sample_graph=sample,
            num_tasks=num_tasks,
            subset=SEARCH_SUBSET,
            ckpt_paths=ckpt_paths,
            mode="pretrained",
            fusion_kind=SEARCH_FUSION_KIND,
            dropout=dropout,
            finetune_encoders=(not SEARCH_FREEZE_ENCODERS),
        )

        # train and get best val auc
        best_val = train_multitask(
            model, tr_loader, vl_loader, num_tasks=num_tasks,
            epochs=sel_epochs, weight_decay=weight_decay, lr=LEARNING_RATE
        )
        # also compute current val_auc precisely (optional)
        val_auc, val_valid = eval_mean_auc(model, vl_loader, num_tasks=num_tasks)

        rows.append({
            "config": cfg["name"],
            "val_auc": float(val_auc),
            "val_valid_tasks": int(val_valid),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # if you want ultra-fast selection: stop after first run only
        if SEARCH_SINGLE_RUN:
            pass

    sel_df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    sel_csv = os.path.join(OUT_ROOT, f"selection_{ds_name}.csv")
    sel_df.to_csv(sel_csv, index=False)
    print(f"[{ds_name}] Selection table -> {sel_csv}")

    best_name = sel_df.iloc[0]["config"]
    best = next(c for c in configs if c["name"] == best_name)
    print(f" [{ds_name}] Best config by val AUC: {best_name}  (val_auc={sel_df.iloc[0]['val_auc']:.4f})")
    return best


# =========================================================
# Baselines: VSA+FP features (for classical + MPN+VSA+FP)
# =========================================================
def calc_vsa_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        smr = np.array([float(getattr(Descriptors, f"SMR_VSA{i}")(mol)) for i in range(1, 11)], dtype=np.float32)
        slogp = np.array([float(getattr(Descriptors, f"SlogP_VSA{i}")(mol)) for i in range(1, 13)], dtype=np.float32)  # 12 dims
        peoe = np.array([float(getattr(Descriptors, f"PEOE_VSA{i}")(mol)) for i in range(1, 15)], dtype=np.float32)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp = arr.astype(np.float32)

        smr = np.nan_to_num(smr, nan=0.0, posinf=0.0, neginf=0.0)
        slogp = np.nan_to_num(slogp, nan=0.0, posinf=0.0, neginf=0.0)
        peoe = np.nan_to_num(peoe, nan=0.0, posinf=0.0, neginf=0.0)
        fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.concatenate([smr, slogp, peoe, fp], axis=0).astype(np.float32)
        return x
    except Exception:
        return None


def load_baseline_matrices(ds_name: str, label_cols: List[str], split_seed: int, val_split: float, test_split: float):
    """
    Re-load csv, compute X (VSA+FP), y/mask, and same scaffold split as fusion pipeline.
    """
    path = os.path.join(DATA_DIR, f"{ds_name}.csv")
    df = pd.read_csv(path)
    smiles_all = df["smiles"].astype(str).tolist()

    # filter invalid smiles + those failing vsa/fp
    keep = []
    X_list = []
    for i, smi in enumerate(smiles_all):
        if Chem.MolFromSmiles(smi) is None:
            continue
        x = calc_vsa_fp(smi)
        if x is None:
            continue
        keep.append(i)
        X_list.append(x)

    df = df.iloc[keep].reset_index(drop=True)
    smiles = df["smiles"].astype(str).tolist()
    X = np.stack(X_list, axis=0).astype(np.float32)

    y, m = coerce_label_matrix(df, label_cols)

    tr_idx, vl_idx, te_idx = scaffold_split_indices(smiles, val_split, test_split, seed=split_seed)

    return (X, y.astype(np.int32), m.astype(np.float32)), (tr_idx, vl_idx, te_idx)


def baseline_classical_multitask(X, y, m, tr_idx, vl_idx, te_idx, seed: int) -> Dict[str, Tuple[float, int]]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    X_tr, X_vl, X_te = X[tr_idx], X[vl_idx], X[te_idx]
    y_tr, y_vl, y_te = y[tr_idx], y[vl_idx], y[te_idx]
    m_tr, m_vl, m_te = m[tr_idx], m[vl_idx], m[te_idx]

    T = y.shape[1]
    results = {}

    def task_mean_auc(probs_by_task: List[Optional[np.ndarray]]) -> Tuple[float, int]:
        aucs = []
        valid = 0
        for t in range(T):
            if probs_by_task[t] is None:
                continue
            mt = m_te[:, t] > 0.5
            if mt.sum() == 0:
                continue
            a = safe_auc(y_te[mt, t], probs_by_task[t][mt])
            if a is not None:
                aucs.append(a)
                valid += 1
        if len(aucs) == 0:
            return 0.5, 0
        return float(np.mean(aucs)), valid

    # gboost
    probs_tasks = [None] * T
    try:
        import xgboost as xgb
        for t in range(T):
            mt = m_tr[:, t] > 0.5
            mv = m_vl[:, t] > 0.5
            if mt.sum() < 50 or mv.sum() < 20:
                continue
            ytr_t = y_tr[mt, t]
            yvl_t = y_vl[mv, t]
            if len(np.unique(ytr_t)) < 2 or len(np.unique(yvl_t)) < 2:
                continue
            pos = float((ytr_t == 1).sum())
            neg = float((ytr_t == 0).sum())
            spw = neg / max(pos, 1.0)

            model = xgb.XGBClassifier(
                n_estimators=5000,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                tree_method="hist",
                eval_metric="auc",
                scale_pos_weight=spw,
            )
            model.fit(
                X_tr[mt], ytr_t,
                eval_set=[(X_vl[mv], yvl_t)],
                verbose=False,
                early_stopping_rounds=100
            )
            probs_tasks[t] = model.predict_proba(X_te)[:, 1]
        results["gboost"] = task_mean_auc(probs_tasks)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        probs_tasks = [None] * T
        for t in range(T):
            mt = m_tr[:, t] > 0.5
            if mt.sum() < 50:
                continue
            ytr_t = y_tr[mt, t]
            if len(np.unique(ytr_t)) < 2:
                continue
            model = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=800, random_state=seed)
            model.fit(X_tr[mt], ytr_t)
            if hasattr(model, "predict_proba"):
                probs_tasks[t] = model.predict_proba(X_te)[:, 1]
            else:
                scores = model.decision_function(X_te)
                probs_tasks[t] = 1.0 / (1.0 + np.exp(-scores))
        results["gboost"] = task_mean_auc(probs_tasks)

    # RF
    probs_tasks = [None] * T
    for t in range(T):
        mt = m_tr[:, t] > 0.5
        if mt.sum() < 50:
            continue
        ytr_t = y_tr[mt, t]
        if len(np.unique(ytr_t)) < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=1000,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced"
        )
        rf.fit(X_tr[mt], ytr_t)
        probs_tasks[t] = rf.predict_proba(X_te)[:, 1]
    results["rf"] = task_mean_auc(probs_tasks)

    # MLP
    probs_tasks = [None] * T
    for t in range(T):
        mt = m_tr[:, t] > 0.5
        if mt.sum() < 100:
            continue
        ytr_t = y_tr[mt, t]
        if len(np.unique(ytr_t)) < 2:
            continue
        mlp = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(1024, 512, 256),
                activation="relu",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed,
            ))
        ])
        mlp.fit(X_tr[mt], ytr_t)
        probs_tasks[t] = mlp.predict_proba(X_te)[:, 1]
    results["mlp"] = task_mean_auc(probs_tasks)

    return results


class MPNWithGlobalMultiTask(nn.Module):
    def __init__(self, atom_fdim: int, bond_fdim: int, global_dim: int, num_tasks: int, hidden=512, depth=3, dropout=0.2):
        super().__init__()
        self.num_tasks = num_tasks
        mpn_args = Namespace(
            hidden_size=hidden,
            depth=depth,
            dropout=0.0,
            undirected=True,
            atom_messages=True,
            features_only=False,
            use_input_features=False,
            activation="ReLU",
            bias=True
        )
        self.mpn = MPN(mpn_args, atom_fdim=atom_fdim, bond_fdim=bond_fdim)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, batch):
        if hasattr(batch, "to_data_list"):
            pyg_list = batch.to_data_list()
        else:
            pyg_list = [batch]
        try:
            mol_vec = self.mpn(prompt=False, batch=pyg_list)
        except TypeError:
            try:
                mol_vec = self.mpn(batch=pyg_list)
            except TypeError:
                mol_vec = self.mpn(pyg_list)

        gf = batch.global_features
        if gf.dim() == 1:
            gf = gf.unsqueeze(0)
        if gf.dim() == 3 and gf.size(1) == 1:
            gf = gf.squeeze(1)
        gf = self.global_proj(gf.to(mol_vec.device))
        logits = self.head(torch.cat([mol_vec, gf], dim=-1))
        return logits


def baseline_mpn_global_multitask(
    ds_name: str,
    loaders,
    num_tasks: int,
    epochs: int,
    weight_decay: float,
    seed: int,
):
    """
    For MPN+VSA+FP baseline we need Data.global_features to be 2084-dim.
    We rebuild the loaders by overwriting global_features for each graph.
    """
    set_seed(seed)
    tr_loader, vl_loader, te_loader = loaders

    def rebuild_loader(old_loader, shuffle: bool):
        old_ds = old_loader.dataset
        data_iter = getattr(old_ds, "_data", None)
        if data_iter is None:
            data_iter = [old_ds[i] for i in range(len(old_ds))]

        # 空集直接返回一个不shuffle的loader（且batch_size=1也行）
        if len(data_iter) == 0:
            return GeoDataLoader(
                MoleculeListDataset([]),
                batch_size=1,
                shuffle=False
            )

        new_list = []
        for g in data_iter:
            smi = getattr(g, "smiles", None)
            x = calc_vsa_fp(smi) if smi is not None else None
            if x is None:
                x = np.zeros((GLOBAL_DIM,), dtype=np.float32)

            g.global_features = torch.tensor(x, dtype=torch.float32).view(1, -1)
            new_list.append(g)

        return GeoDataLoader(
            MoleculeListDataset(new_list),
            batch_size=old_loader.batch_size,
            shuffle=shuffle,  
            num_workers=getattr(old_loader, "num_workers", 0),
            pin_memory=getattr(old_loader, "pin_memory", False),
            drop_last=False,
        )


    tr2 = rebuild_loader(tr_loader, shuffle=True)
    vl2 = rebuild_loader(vl_loader, shuffle=False)
    te2 = rebuild_loader(te_loader, shuffle=False)

    sample = tr2.dataset[0]
    atom_fdim = sample.x.shape[1]
    bond_fdim = sample.edge_attr.shape[1]

    model = MPNWithGlobalMultiTask(atom_fdim, bond_fdim, GLOBAL_DIM, num_tasks=num_tasks, hidden=512, depth=3, dropout=0.2).to(device)

    pos_weight = compute_pos_weight_vector(tr2, num_tasks=num_tasks)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=LR_FACTOR, patience=PATIENCE_LR)

    best_val = -1e9
    best_state = None
    patience = 0

    for ep in range(epochs):
        model.train()
        for batch in tr2:
            batch = batch.to(device)
            logits = model(batch).view(batch.y.size(0), -1)
            y = batch.y.view(batch.y.size(0), -1).float().to(device)
            m = batch.y_mask.view(batch.y_mask.size(0), -1).float().to(device)

            loss_mat = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
            loss = (loss_mat * m).sum()
            denom = m.sum().item()
            if denom > 0:
                loss = loss / denom
            else:
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_auc, val_valid = eval_mean_auc(model, vl2, num_tasks=num_tasks)
        sched.step(val_auc)

        if val_auc > best_val + 1e-8:
            best_val = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_ES:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_auc, test_valid = eval_mean_auc(model, te2, num_tasks=num_tasks)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(test_auc), int(test_valid)


# =========================================================
# Main
# =========================================================
configs = find_ckpt_configs(CKPT_ROOT)
best_configs = {}
all_records = []

for ds in DATASETS:
    print("\n" + "=" * 80)
    print(f"DATASET = {ds}")
    print("=" * 80)

    params = DATASET_PARAMS[ds]
    bundle = load_multitask_dataset(
        ds_name=ds,
        batch_size=params["batch_size"],
        val_split=params["val_split"],
        test_split=params["test_split"],
        seed=SPLIT_SEED,
    )
    loaders = bundle["loaders"]
    num_tasks = bundle["num_tasks"]
    label_cols = bundle["label_cols"]

    print(f"[{ds}] num_tasks={num_tasks}  batch_size={params['batch_size']}  epochs={params['epochs']}")

    # -------------------------
    # 1) Select best ckpt config for THIS dataset
    # -------------------------
    best_cfg = select_best_config_for_dataset(
        ds_name=ds,
        configs=configs,
        loaders=loaders,
        num_tasks=num_tasks,
        dropout=params["dropout"],
        weight_decay=params["weight_decay"],
        epochs=params["epochs"],
    )
    best_configs[ds] = {"config": best_cfg["name"], "paths": best_cfg["paths"]}

    # -------------------------
    # helper to run multi-run experiment and record
    # -------------------------
    def run_multi(exp_name: str, subset: List[int], mode: str, fusion_kind: str, ckpt_paths: Optional[List[str]]):
        aucs, valids = [], []
        for r in range(RUNS):
            seed = 42 + r
            print(f"\n[{ds}] {exp_name} | subset={'+'.join([SPACE_NAMES[i] for i in subset])} | run={r+1}/{RUNS} seed={seed}")
            auc, valid = run_one(
                loaders=loaders,
                num_tasks=num_tasks,
                subset=subset,
                ckpt_paths=ckpt_paths,
                mode=mode,
                fusion_kind=fusion_kind,
                dropout=params["dropout"],
                weight_decay=params["weight_decay"],
                epochs=params["epochs"],
                finetune_encoders=(FINETUNE_ENCODERS_IN_ABLATION if mode == "pretrained" else True),
                seed=seed,
            )
            aucs.append(auc)
            valids.append(valid)
            print(f"  -> test mean AUC={auc:.4f} (valid_tasks={valid})")

        rec = {
            "dataset": ds,
            "experiment": exp_name,
            "mode": mode,
            "fusion": fusion_kind,
            "subset": "+".join([SPACE_NAMES[i] for i in subset]),
            "runs": RUNS,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "valid_tasks_mean": float(np.mean(valids)),
            "num_tasks_total": int(num_tasks),
            "best_pretrain_config": best_cfg["name"],
        }
        all_records.append(rec)
        print(f" [{ds}] {exp_name}: AUC={rec['auc_mean']:.4f} ± {rec['auc_std']:.4f}")

    # -------------------------
    # 2) Our method (best config): all4 + pretrained + static
    # -------------------------
    run_multi(
        exp_name="ours_pretrained_static_all4",
        subset=ALL4,
        mode="pretrained",
        fusion_kind="static",
        ckpt_paths=best_cfg["paths"],
    )

    # -------------------------
    # 3) Drop-one-space ablation (best config): pretrained + static
    # -------------------------
    for subset in DROP_ONE_SUBSETS:
        run_multi(
            exp_name="ablation_drop_one_pretrained_static",
            subset=subset,
            mode="pretrained",
            fusion_kind="static",
            ckpt_paths=best_cfg["paths"],
        )

    # -------------------------
    # 4) Aggregation ablation (best config): pretrained + (static vs adaptive) on all4
    # -------------------------
    run_multi(
        exp_name="ablation_aggregation_pretrained_static",
        subset=ALL4,
        mode="pretrained",
        fusion_kind="static",
        ckpt_paths=best_cfg["paths"],
    )
    run_multi(
        exp_name="ablation_aggregation_pretrained_adaptive",
        subset=ALL4,
        mode="pretrained",
        fusion_kind="adaptive",
        ckpt_paths=best_cfg["paths"],
    )

    # -------------------------
    # 5) No-pretraining ablation: scratch MPN encoders + static (all4)
    # -------------------------
    run_multi(
        exp_name="ablation_no_pretrain_scratch_mpn_static_all4",
        subset=ALL4,
        mode="scratch",
        fusion_kind="static",
        ckpt_paths=None,
    )

    # -------------------------
    # 6) Baselines (VSA+FP)
    # -------------------------
    # Classical baselines
    (X, y, m), (tr_idx, vl_idx, te_idx) = load_baseline_matrices(
        ds_name=ds,
        label_cols=label_cols,
        split_seed=SPLIT_SEED,
        val_split=params["val_split"],
        test_split=params["test_split"],
    )

    base_bucket = defaultdict(list)  # model -> list of (auc, valid)
    for r in range(RUNS):
        seed = 42 + r
        print(f"\n[{ds}] baseline_vsa_fp_classical run {r+1}/{RUNS} seed={seed}")
        res = baseline_classical_multitask(X, y, m, tr_idx, vl_idx, te_idx, seed=seed)
        for k, (auc, valid) in res.items():
            base_bucket[k].append((auc, valid))
            print(f"  {k:>8s} -> test mean AUC={auc:.4f} (valid_tasks={valid})")

    for k, vals in base_bucket.items():
        aucs = [v[0] for v in vals]
        valids = [v[1] for v in vals]
        rec = {
            "dataset": ds,
            "experiment": "baseline_vsa_fp_classical",
            "model": k,
            "runs": RUNS,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "valid_tasks_mean": float(np.mean(valids)),
            "num_tasks_total": int(num_tasks),
            "best_pretrain_config": best_cfg["name"],
        }
        all_records.append(rec)
        print(f" [{ds}] baseline {k}: AUC={rec['auc_mean']:.4f} ± {rec['auc_std']:.4f}")

    # MPN + VSA+FP baseline
    mpn_aucs, mpn_valids = [], []
    for r in range(RUNS):
        seed = 42 + r
        print(f"\n[{ds}] baseline_mpn_global_vsa_fp run {r+1}/{RUNS} seed={seed}")
        auc, valid = baseline_mpn_global_multitask(
            ds_name=ds,
            loaders=loaders,
            num_tasks=num_tasks,
            epochs=params["epochs"],
            weight_decay=params["weight_decay"],
            seed=seed,
        )
        mpn_aucs.append(auc)
        mpn_valids.append(valid)
        print(f"  -> test mean AUC={auc:.4f} (valid_tasks={valid})")

    rec = {
        "dataset": ds,
        "experiment": "baseline_mpn_global_vsa_fp",
        "model": "chemprop_mpn",
        "runs": RUNS,
        "auc_mean": float(np.mean(mpn_aucs)),
        "auc_std": float(np.std(mpn_aucs)),
        "valid_tasks_mean": float(np.mean(mpn_valids)),
        "num_tasks_total": int(num_tasks),
        "best_pretrain_config": best_cfg["name"],
    }
    all_records.append(rec)
    print(f" [{ds}] baseline chemprop_mpn: AUC={rec['auc_mean']:.4f} ± {rec['auc_std']:.4f}")

# write outputs
best_json = os.path.join(OUT_ROOT, "best_configs.json")
with open(best_json, "w", encoding="utf-8") as f:
    json.dump(best_configs, f, indent=2, ensure_ascii=False)
print(f"\n Best configs saved to: {best_json}")

summary_csv = os.path.join(OUT_ROOT, "summary_all.csv")
pd.DataFrame(all_records).to_csv(summary_csv, index=False)
print(f" Summary saved to: {summary_csv}")
