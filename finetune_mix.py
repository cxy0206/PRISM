#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PRISM mixed-solvent fine-tuning.

Prior-specific encoders feed trainable fusion modules, followed by
composition-weighted solvent pooling and temperature-dependent regression.
Encoders are frozen by default. Optional pure-solvent warm starts transfer only
the fusion modules. CSV column names can be configured through the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch

from model.getdata import smiles2graph
from finetune_pair import TempPairMLP, _stack_encode, load_pretrained_encoders
from model.fusion import make_fusion
from model.checkpoints import encoder_output_dim


# ============================================================
# Basic utilities
# ============================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_int_tuple(s: str) -> Tuple[int, ...]:
    if s.strip() == "":
        return tuple()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def pct_within_1_np(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return float((np.abs(preds - labels) <= 1.0).mean())


def rmse_np(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def unwrap_subset_dataset(ds: Dataset) -> Dataset:
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def canonical_pair(s1: str, s2: str) -> Tuple[str, str]:
    return tuple(sorted((str(s1), str(s2))))


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Data handling
# ============================================================


@dataclass
class MixColumns:
    solute: str = "solute_smiles"
    solvent1: str = "solvent1_smiles"
    solvent2: str = "solvent2_smiles"
    x1: str = "Xsolvent1"
    x2: str = "Xsolvent2"
    temperature: str = "temperature"
    label: str = "logS"


def _check_columns(df: pd.DataFrame, columns: MixColumns, csv_path: str) -> None:
    required = asdict(columns)
    missing = [col for col in required.values() if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            "Use --solute_col, --solvent1_col, --solvent2_col, --x1_col, "
            "--x2_col, --temperature_col, or --label_col to override names."
        )


def load_mixture_dataframe(
    csv_path: str,
    columns: MixColumns,
    normalize_composition: bool = True,
    drop_invalid_smiles: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _check_columns(df, columns, csv_path)

    keep = [
        columns.solute,
        columns.solvent1,
        columns.solvent2,
        columns.x1,
        columns.x2,
        columns.temperature,
        columns.label,
    ]
    df = df[keep].copy()

    # Basic type normalization.
    for col in [columns.solute, columns.solvent1, columns.solvent2]:
        df[col] = df[col].astype(str)
    for col in [columns.x1, columns.x2, columns.temperature, columns.label]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=keep).reset_index(drop=True)
    if len(df) < before:
        print(f"[Data] Dropped {before - len(df)} rows with missing/non-numeric values.")

    if normalize_composition:
        denom = df[columns.x1].astype(float) + df[columns.x2].astype(float)
        bad = denom <= 0
        if bad.any():
            raise ValueError("Found mixture rows with Xsolvent1 + Xsolvent2 <= 0.")
        df[columns.x1] = df[columns.x1] / denom
        df[columns.x2] = df[columns.x2] / denom

    # This flag is kept only for compatibility. smiles2graph usually handles
    # conversion in the existing project; if invalid SMILES must be removed,
    # do it in the preprocessing script to keep split files deterministic.
    if drop_invalid_smiles:
        print("[Warning] drop_invalid_smiles=True is not implemented here; preprocess invalid SMILES upstream.")

    return df


class MixtureDataset(Dataset):
    """Mixed-solvent graph dataset.

    Each item returns:
        solute graph, solvent1 graph, solvent2 graph,
        temperature, label, Xsolvent1, Xsolvent2,
        solute SMILES, solvent1 SMILES, solvent2 SMILES
    """

    def __init__(
        self,
        solute_graphs: Sequence[Any],
        solvent1_graphs: Sequence[Any],
        solvent2_graphs: Sequence[Any],
        temps: Sequence[float],
        labels: Sequence[float],
        x1: Sequence[float],
        x2: Sequence[float],
        solute_smiles: Sequence[str],
        solvent1_smiles: Sequence[str],
        solvent2_smiles: Sequence[str],
    ):
        n = len(labels)
        assert len(solute_graphs) == len(solvent1_graphs) == len(solvent2_graphs) == n
        assert len(temps) == len(x1) == len(x2) == n
        assert len(solute_smiles) == len(solvent1_smiles) == len(solvent2_smiles) == n

        self.solute_graphs = list(solute_graphs)
        self.solvent1_graphs = list(solvent1_graphs)
        self.solvent2_graphs = list(solvent2_graphs)
        self.temps = np.asarray(temps, dtype=float)
        self.labels = np.asarray(labels, dtype=float)
        self.x1 = np.asarray(x1, dtype=float)
        self.x2 = np.asarray(x2, dtype=float)
        self.solute_smiles = list(solute_smiles)
        self.solvent1_smiles = list(solvent1_smiles)
        self.solvent2_smiles = list(solvent2_smiles)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.solute_graphs[idx],
            self.solvent1_graphs[idx],
            self.solvent2_graphs[idx],
            float(self.temps[idx]),
            float(self.labels[idx]),
            float(self.x1[idx]),
            float(self.x2[idx]),
            self.solute_smiles[idx],
            self.solvent1_smiles[idx],
            self.solvent2_smiles[idx],
        )


def collate_mixture(batch: Sequence[Tuple[Any, ...]]):
    (
        solute_graphs,
        solvent1_graphs,
        solvent2_graphs,
        temps,
        labels,
        x1,
        x2,
        solute_smiles,
        solvent1_smiles,
        solvent2_smiles,
    ) = zip(*batch)

    solute_batch = Batch.from_data_list(list(solute_graphs))
    solvent1_batch = Batch.from_data_list(list(solvent1_graphs))
    solvent2_batch = Batch.from_data_list(list(solvent2_graphs))

    temps_t = torch.tensor(temps, dtype=torch.float32).view(-1, 1)
    labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    x1_t = torch.tensor(x1, dtype=torch.float32).view(-1, 1)
    x2_t = torch.tensor(x2, dtype=torch.float32).view(-1, 1)

    return (
        solute_batch,
        solvent1_batch,
        solvent2_batch,
        temps_t,
        labels_t,
        x1_t,
        x2_t,
        list(solute_smiles),
        list(solvent1_smiles),
        list(solvent2_smiles),
    )


def build_mixture_dataset(csv_path: str, columns: MixColumns, normalize_composition: bool = True) -> MixtureDataset:
    df = load_mixture_dataframe(csv_path, columns, normalize_composition=normalize_composition)

    solute_smiles = df[columns.solute].astype(str).tolist()
    solvent1_smiles = df[columns.solvent1].astype(str).tolist()
    solvent2_smiles = df[columns.solvent2].astype(str).tolist()
    y = df[columns.label].astype(float).to_numpy()
    temps = df[columns.temperature].astype(float).to_numpy()
    x1 = df[columns.x1].astype(float).to_numpy()
    x2 = df[columns.x2].astype(float).to_numpy()

    # The existing project uses smiles2graph(smiles, y=..., properties=...).
    solute_graphs = smiles2graph(solute_smiles, y=y, properties=temps)
    solvent1_graphs = smiles2graph(solvent1_smiles, y=y, properties=temps)
    solvent2_graphs = smiles2graph(solvent2_smiles, y=y, properties=temps)

    return MixtureDataset(
        solute_graphs=solute_graphs,
        solvent1_graphs=solvent1_graphs,
        solvent2_graphs=solvent2_graphs,
        temps=temps,
        labels=y,
        x1=x1,
        x2=x2,
        solute_smiles=solute_smiles,
        solvent1_smiles=solvent1_smiles,
        solvent2_smiles=solvent2_smiles,
    )


# ============================================================
# Split handling
# ============================================================


def split_indices_random(
    n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = max(0, min(n, n_train))
    n_val = max(0, min(n - n_train, n_val))

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train + n_val].tolist()
    test_idx = idx[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def split_indices_by_solvent_pair(
    solvent1_smiles: Sequence[str],
    solvent2_smiles: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    canonicalize: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    assert len(solvent1_smiles) == len(solvent2_smiles)

    keys: List[Tuple[str, str]] = []
    for s1, s2 in zip(solvent1_smiles, solvent2_smiles):
        key = canonical_pair(s1, s2) if canonicalize else (str(s1), str(s2))
        keys.append(key)

    unique_pairs = sorted(set(keys))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pairs)

    n_pairs = len(unique_pairs)
    n_train = int(round(train_frac * n_pairs))
    n_val = int(round(val_frac * n_pairs))
    n_train = max(0, min(n_pairs, n_train))
    n_val = max(0, min(n_pairs - n_train, n_val))

    train_pairs = set(unique_pairs[:n_train])
    val_pairs = set(unique_pairs[n_train:n_train + n_val])
    test_pairs = set(unique_pairs[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for i, key in enumerate(keys):
        if key in train_pairs:
            train_idx.append(i)
        elif key in val_pairs:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


def read_split_file(path: str) -> Tuple[List[int], List[int], List[int]]:
    """Read a split CSV with columns: index, split.

    split must be one of train/val/test.
    """
    df = pd.read_csv(path)
    if "index" not in df.columns or "split" not in df.columns:
        raise ValueError("Split file must contain columns: index, split")

    split = df["split"].astype(str).str.lower()
    train_idx = df.loc[split == "train", "index"].astype(int).tolist()
    val_idx = df.loc[split == "val", "index"].astype(int).tolist()
    test_idx = df.loc[split == "test", "index"].astype(int).tolist()
    return train_idx, val_idx, test_idx


def save_split_file(path: str, train_idx: Sequence[int], val_idx: Sequence[int], test_idx: Sequence[int]) -> None:
    rows = []
    rows.extend({"index": int(i), "split": "train"} for i in train_idx)
    rows.extend({"index": int(i), "split": "val"} for i in val_idx)
    rows.extend({"index": int(i), "split": "test"} for i in test_idx)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_mixture_loaders(
    dataset: MixtureDataset,
    batch_size: int,
    split_strategy: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    num_workers: int = 0,
    split_file: Optional[str] = None,
    save_split_path: Optional[str] = None,
    canonicalize_pair_split: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    if split_file:
        train_idx, val_idx, test_idx = read_split_file(split_file)
        split_meta = {"split_strategy": "file", "split_file": split_file}
    elif split_strategy == "random":
        train_idx, val_idx, test_idx = split_indices_random(
            n=len(dataset),
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
        split_meta = {"split_strategy": "random"}
    elif split_strategy in {"pair", "pair_disjoint", "solvent_pair_disjoint"}:
        train_idx, val_idx, test_idx = split_indices_by_solvent_pair(
            solvent1_smiles=dataset.solvent1_smiles,
            solvent2_smiles=dataset.solvent2_smiles,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            canonicalize=canonicalize_pair_split,
        )
        split_meta = {"split_strategy": "solvent_pair_disjoint"}
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    if save_split_path:
        ensure_dir(os.path.dirname(save_split_path) or ".")
        save_split_file(save_split_path, train_idx, val_idx, test_idx)

    def pair_count(indices: Sequence[int]) -> int:
        return len({canonical_pair(dataset.solvent1_smiles[i], dataset.solvent2_smiles[i]) for i in indices})

    split_meta.update(
        {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_pair_count": pair_count(train_idx),
            "val_pair_count": pair_count(val_idx),
            "test_pair_count": pair_count(test_idx),
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "seed": seed,
        }
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_mixture,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mixture,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mixture,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, split_meta


def get_sample_graphs_from_loader(loader: DataLoader):
    base_ds = unwrap_subset_dataset(loader.dataset)
    sample = base_ds[0]
    return sample[0], sample[1]


# ============================================================
# PRISM mixed-solvent model
# ============================================================


class PRISMMixPredictor(nn.Module):
    """Composition-weighted mixed-solvent predictor.

    For a solute m and two solvents v1, v2 with mole fractions x1, x2:

        z_m      = Fusion({G_s(m)}_s)
        z_v1     = Fusion({G_s(v1)}_s)
        z_v2     = Fusion({G_s(v2)}_s)
        z_env    = x1 * z_v1 + x2 * z_v2
        y_hat    = TempPairMLP(z_m, z_env, T)

    The molecular encoders remain frozen when --freeze_encoders is enabled.
    """

    def __init__(
        self,
        solute_encs: Sequence[nn.Module],
        solvent_encs: Sequence[nn.Module],
        solute_fusion: nn.Module,
        solvent_fusion: nn.Module,
        combiner: TempPairMLP,
    ):
        super().__init__()
        self.solute_encs = nn.ModuleList(solute_encs)
        self.solvent_encs = nn.ModuleList(solvent_encs)
        self.solute_fusion = solute_fusion
        self.solvent_fusion = solvent_fusion
        self.combiner = combiner
        self.freeze_encoders = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoders:
            for encoder in list(self.solute_encs) + list(self.solvent_encs):
                encoder.eval()
        return self

    @staticmethod
    def _fuse(fusion: nn.Module, embeddings: torch.Tensor, global_features: Optional[torch.Tensor]):
        out = fusion(embeddings, global_features=global_features)
        return out[0] if isinstance(out, tuple) else out

    def encode_solute(self, solute_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        z_spaces = _stack_encode(self.solute_encs, solute_batch)
        g = getattr(solute_batch, "global_features", None)
        z = self._fuse(self.solute_fusion, z_spaces, g)
        return z, z_spaces

    def encode_solvent(self, solvent_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        z_spaces = _stack_encode(self.solvent_encs, solvent_batch)
        g = getattr(solvent_batch, "global_features", None)
        z = self._fuse(self.solvent_fusion, z_spaces, g)
        return z, z_spaces

    def forward(
        self,
        solute_batch: Batch,
        solvent1_batch: Batch,
        solvent2_batch: Batch,
        x1: torch.Tensor,
        x2: torch.Tensor,
        temps: torch.Tensor,
        return_embeddings: bool = False,
    ):
        z_solute, z_solute_spaces = self.encode_solute(solute_batch)
        z_solv1, z_solv1_spaces = self.encode_solvent(solvent1_batch)
        z_solv2, z_solv2_spaces = self.encode_solvent(solvent2_batch)

        x1 = x1.view(-1, 1).to(z_solv1.dtype)
        x2 = x2.view(-1, 1).to(z_solv2.dtype)
        z_env = x1 * z_solv1 + x2 * z_solv2

        pred = self.combiner(z_solute, z_env, temps)
        if pred.dim() == 1:
            pred = pred.view(-1, 1)

        if return_embeddings:
            return {
                "pred": pred,
                "z_solute": z_solute,
                "z_solvent1": z_solv1,
                "z_solvent2": z_solv2,
                "z_env": z_env,
                "z_solute_spaces": z_solute_spaces,
                "z_solvent1_spaces": z_solv1_spaces,
                "z_solvent2_spaces": z_solv2_spaces,
            }
        return pred


def make_ckpt_paths_from_pretrain(pretrain_base_dir: str, task_dirs: Sequence[str], ckpt_name: str) -> List[str]:
    ckpts = []
    for task_dir in task_dirs:
        path = os.path.join(pretrain_base_dir, task_dir, ckpt_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing encoder checkpoint: {path}")
        ckpts.append(path)
    return ckpts


def build_mix_model(
    sample_solute_graph, sample_solvent_graph, encoder_ckpts, emb_dim,
    solute_dropout, solvent_dropout, mlp_hidden, mlp_dropout, t_embed_dim,
    device, fusion_type="adaptive", use_global_features=False,
):
    """Rebuild the encoder architecture from its saved configuration."""
    solute_encs = load_pretrained_encoders(
        sample_solute_graph, list(encoder_ckpts), use_global_features, device)
    solvent_encs = load_pretrained_encoders(
        sample_solvent_graph, list(encoder_ckpts), use_global_features, device)
    emb_dim = encoder_output_dim(solute_encs)
    solute_fusion = make_fusion(len(encoder_ckpts), emb_dim, solute_dropout,
                              True, fusion_type, use_global_features)
    solvent_fusion = make_fusion(len(encoder_ckpts), emb_dim, solvent_dropout,
                               True, fusion_type, use_global_features)
    combiner = TempPairMLP(emb_dim, t_embed_dim=t_embed_dim,
                          hidden=mlp_hidden, dropout=mlp_dropout)
    model = PRISMMixPredictor(solute_encs, solvent_encs, solute_fusion,
                             solvent_fusion, combiner).to(device)
    model.fusion_type = fusion_type
    model.use_global_features = use_global_features
    return model


def _remap_state_dict_keys_for_mix_model(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Support old and new naming conventions.

    New main model names:
        solute_encs.*, solvent_encs.*, solute_fusion.*, solvent_fusion.*, combiner.*

    Pure-solvent PRISM parameter names:
        sol_encs.*, solv_encs.*, sol_f.*, solv_f.*, combine.*
    """
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        if key.startswith("sol_encs."):
            new_key = key.replace("sol_encs", "solute_encs", 1)
        elif key.startswith("solv_encs."):
            new_key = key.replace("solv_encs", "solvent_encs", 1)
        elif key.startswith("sol_f."):
            new_key = key.replace("sol_f", "solute_fusion", 1)
        elif key.startswith("solv_f."):
            new_key = key.replace("solv_f", "solvent_fusion", 1)
        elif key.startswith("combine."):
            new_key = key.replace("combine", "combiner", 1)

        # Ignore modules unrelated to fusion transfer in legacy checkpoints.
        if new_key.startswith(("attends.", "cross_scale", "space_head_s1.", "space_head_s2.")):
            continue
        remapped[new_key] = value
    return remapped


def load_checkpoint_into_mix_model(
    model, ckpt_path, device, freeze_encoders=True,
    freeze_fusions=False, freeze_combiner=False,
):
    """Warm-start only the two fusion modules from pure-solvent training."""
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        for name in ("fusion_type", "use_global_features"):
            if name in checkpoint and checkpoint[name] != getattr(model, name):
                raise ValueError(f"Warm-start {name} mismatch: {checkpoint[name]} != {getattr(model, name)}")
        if "encoder_configs" in checkpoint:
            expected = [enc.property_index for enc in model.solute_encs]
            saved = [cfg["property_index"] for cfg in checkpoint["encoder_configs"]]
            if saved != expected:
                raise ValueError(f"Warm-start prior order mismatch: {saved} != {expected}")
        state = _remap_state_dict_keys_for_mix_model(
            checkpoint.get("model_state_dict", checkpoint))
        transfers = []
        for name in ("solute_fusion", "solvent_fusion"):
            module = getattr(model, name)
            prefix = name + "."
            subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            expected = module.state_dict()
            if set(subset) != set(expected) or any(
                subset[k].shape != expected[k].shape for k in expected if k in subset
            ):
                raise ValueError(f"Incompatible {name} checkpoint; match fusion type, global features, and branch dimensions.")
            transfers.append((module, subset))
        for module, subset in transfers:
            module.load_state_dict(subset, strict=True)
        print(f"[Checkpoint] Loaded both fusion modules from {ckpt_path}")

    model.freeze_encoders = freeze_encoders
    for group, frozen in ((model.solute_encs, freeze_encoders),
                          (model.solvent_encs, freeze_encoders),
                          (model.solute_fusion, freeze_fusions),
                          (model.solvent_fusion, freeze_fusions),
                          (model.combiner, freeze_combiner)):
        for name, parameter in group.named_parameters():
            parameter.requires_grad_(not frozen and not name.endswith("cached_zero_vector"))
    model.train(model.training)
    print(f"[Freeze] encoders={freeze_encoders}, fusions={freeze_fusions}, combiner={freeze_combiner}")
    return model


# ============================================================
# Training and evaluation
# ============================================================


def evaluate_mix_model(
    model: PRISMMixPredictor,
    loader: DataLoader,
    device: torch.device,
    save_predictions_path: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_temps: List[np.ndarray] = []
    all_x1: List[np.ndarray] = []
    all_x2: List[np.ndarray] = []
    all_solute_smiles: List[str] = []
    all_s1_smiles: List[str] = []
    all_s2_smiles: List[str] = []

    with torch.no_grad():
        for (
            sol_b,
            solv1_b,
            solv2_b,
            temps,
            labels,
            x1,
            x2,
            solute_smiles,
            solvent1_smiles,
            solvent2_smiles,
        ) in loader:
            sol_b = sol_b.to(device)
            solv1_b = solv1_b.to(device)
            solv2_b = solv2_b.to(device)
            temps = temps.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            pred = model(sol_b, solv1_b, solv2_b, x1, x2, temps)

            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(labels.numpy())
            all_temps.append(temps.detach().cpu().numpy())
            all_x1.append(x1.detach().cpu().numpy())
            all_x2.append(x2.detach().cpu().numpy())
            all_solute_smiles.extend(solute_smiles)
            all_s1_smiles.extend(solvent1_smiles)
            all_s2_smiles.extend(solvent2_smiles)

    preds = np.concatenate(all_preds, axis=0).reshape(-1)
    labels = np.concatenate(all_labels, axis=0).reshape(-1)
    temps = np.concatenate(all_temps, axis=0).reshape(-1)
    x1 = np.concatenate(all_x1, axis=0).reshape(-1)
    x2 = np.concatenate(all_x2, axis=0).reshape(-1)

    metrics = {
        "rmse": rmse_np(preds, labels),
        "mse": float(np.mean((preds - labels) ** 2)),
        "mae": float(np.mean(np.abs(preds - labels))),
        "pct_le1": pct_within_1_np(preds, labels),
        "n": int(len(labels)),
    }

    if save_predictions_path:
        pred_df = pd.DataFrame(
            {
                "solute_smiles": all_solute_smiles,
                "solvent1_smiles": all_s1_smiles,
                "solvent2_smiles": all_s2_smiles,
                "Xsolvent1": x1,
                "Xsolvent2": x2,
                "temperature": temps,
                "logS_true": labels,
                "logS_pred": preds,
                "abs_err": np.abs(preds - labels),
            }
        )
        pred_df.to_csv(save_predictions_path, index=False)

    return {"metrics": metrics, "preds": preds, "labels": labels}


def plot_training_curves(train_losses: Sequence[float], val_losses: Sequence[float], out_path: str) -> None:
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train MSE")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scatter(labels: np.ndarray, preds: np.ndarray, out_path: str, title: str) -> None:
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    plt.figure()
    plt.scatter(labels, preds, s=15, alpha=0.65)
    lo = float(min(labels.min(), preds.min()) - 0.5)
    hi = float(max(labels.max(), preds.max()) + 0.5)
    grid = np.linspace(lo, hi, 200)
    plt.plot(grid, grid, "k--", linewidth=1.0, label="y=x")
    plt.fill_between(grid, grid - 1.0, grid + 1.0, alpha=0.2, label="|error| <= 1")
    plt.xlabel("True logS")
    plt.ylabel("Predicted logS")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def train_mix_model(
    model: PRISMMixPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    save_best_ckpt: bool = True,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check freeze flags.")

    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    bad_epochs = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for (
            sol_b,
            solv1_b,
            solv2_b,
            temps,
            labels,
            x1,
            x2,
            *_rest,
        ) in train_loader:
            sol_b = sol_b.to(device)
            solv1_b = solv1_b.to(device)
            solv2_b = solv2_b.to(device)
            temps = temps.to(device)
            labels = labels.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(sol_b, solv1_b, solv2_b, x1, x2, temps)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            n_seen += labels.size(0)

        train_mse = running_loss / max(1, n_seen)
        train_losses.append(train_mse)

        val_eval = evaluate_mix_model(model, val_loader, device)
        val_mse = float(val_eval["metrics"]["mse"])
        val_losses.append(val_mse)

        print(f"Epoch {epoch:03d} | Train MSE {train_mse:.6f} | Val MSE {val_mse:.6f}")

        if val_mse < best_val - 1e-12:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"[EarlyStop] patience={patience} reached at epoch={epoch}; best_epoch={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    if save_best_ckpt:
        torch.save(
            {
                "format_version": 2,
                "model_state_dict": model.state_dict(),
                "fusion_type": model.fusion_type,
                "use_global_features": model.use_global_features,
                "encoder_configs": [e.encoder_config() for e in model.solute_encs],
                "best_val_mse": best_val,
                "best_epoch": best_epoch,
            },
            os.path.join(out_dir, "model_best.pth"),
        )

    plot_training_curves(train_losses, val_losses, os.path.join(plots_dir, "train_val_loss_curve.png"))

    test_path = os.path.join(out_dir, "test_predictions.csv")
    test_eval = evaluate_mix_model(model, test_loader, device, save_predictions_path=test_path)
    test_metrics = test_eval["metrics"]

    plot_scatter(
        labels=test_eval["labels"],
        preds=test_eval["preds"],
        out_path=os.path.join(plots_dir, "test_scatter.png"),
        title=f"RMSE={test_metrics['rmse']:.3f}, pct<=1={test_metrics['pct_le1']*100:.1f}%",
    )

    stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "test_rmse": test_metrics["rmse"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_pct_le1": test_metrics["pct_le1"],
        "test_n": test_metrics["n"],
    }
    save_json(stats, os.path.join(out_dir, "metrics.json"))
    np.savez(
        os.path.join(out_dir, "training_stats.npz"),
        train_losses=np.asarray(train_losses, dtype=float),
        val_losses=np.asarray(val_losses, dtype=float),
        best_val_mse=float(best_val),
        best_epoch=int(best_epoch),
        test_rmse=float(test_metrics["rmse"]),
        test_pct_le1=float(test_metrics["pct_le1"]),
    )

    print(f"[Test] RMSE={test_metrics['rmse']:.6f} | MAE={test_metrics['mae']:.6f} | pct<=1={test_metrics['pct_le1']*100:.2f}%")
    return stats


# ============================================================
# Experiment runners
# ============================================================


def run_single_experiment(
    dataset: MixtureDataset,
    encoder_ckpts: Sequence[str],
    init_ckpt: Optional[str],
    device: torch.device,
    out_dir: str,
    split_strategy: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    batch_size: int,
    emb_dim: int,
    solute_dropout: float,
    solvent_dropout: float,
    mlp_hidden: Tuple[int, ...],
    mlp_dropout: float,
    t_embed_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    freeze_encoders: bool,
    freeze_fusions: bool,
    freeze_combiner: bool,
    num_workers: int,
    split_file: Optional[str] = None,
    save_split: bool = True,
    fusion_type: str = "adaptive",
    use_global_features: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    ensure_dir(out_dir)

    split_path = os.path.join(out_dir, "split.csv") if save_split else None
    train_loader, val_loader, test_loader, split_meta = make_mixture_loaders(
        dataset=dataset,
        batch_size=batch_size,
        split_strategy=split_strategy,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        num_workers=num_workers,
        split_file=split_file,
        save_split_path=split_path,
    )
    print(f"[Split] {split_meta}")
    save_json(split_meta, os.path.join(out_dir, "split_meta.json"))

    sample_solute, sample_solvent = get_sample_graphs_from_loader(train_loader)
    model = build_mix_model(
        sample_solute_graph=sample_solute,
        sample_solvent_graph=sample_solvent,
        encoder_ckpts=encoder_ckpts,
        emb_dim=emb_dim,
        solute_dropout=solute_dropout,
        solvent_dropout=solvent_dropout,
        mlp_hidden=mlp_hidden,
        mlp_dropout=mlp_dropout,
        t_embed_dim=t_embed_dim,
        device=device,
        fusion_type=fusion_type,
        use_global_features=use_global_features,
    )

    model = load_checkpoint_into_mix_model(
        model=model,
        ckpt_path=init_ckpt.format(seed=seed) if init_ckpt else None,
        device=device,
        freeze_encoders=freeze_encoders,
        freeze_fusions=freeze_fusions,
        freeze_combiner=freeze_combiner,
    )

    save_json({"encoder_ckpts": list(encoder_ckpts),
               "init_ckpt": init_ckpt.format(seed=seed) if init_ckpt else None,
               "fusion_type": fusion_type, "use_global_features": use_global_features,
               "seed": seed, "epochs": epochs, "batch_size": batch_size,
               "lr": lr, "weight_decay": weight_decay, "patience": patience,
               "solute_dropout": solute_dropout, "solvent_dropout": solvent_dropout,
               "mlp_hidden": list(mlp_hidden), "mlp_dropout": mlp_dropout,
               "t_embed_dim": t_embed_dim, "freeze_encoders": freeze_encoders,
               "freeze_fusions": freeze_fusions, "freeze_combiner": freeze_combiner},
              os.path.join(out_dir, "config.json"))
    stats = train_mix_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        save_best_ckpt=True,
    )
    stats.update(split_meta)
    stats["seed"] = seed
    stats["out_dir"] = out_dir
    return stats


def summarize_runs(rows: List[Dict[str, Any]], out_path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    group_cols = [col for col in ["split_strategy", "experiment"] if col in df.columns]
    if group_cols:
        agg = (
            df.groupby(group_cols)
            .agg(
                rmse_mean=("test_rmse", "mean"),
                rmse_std=("test_rmse", lambda x: x.std(ddof=0)),
                rmse_var=("test_rmse", lambda x: x.var(ddof=0)),
                pct_le1_mean=("test_pct_le1", "mean"),
                pct_le1_std=("test_pct_le1", lambda x: x.std(ddof=0)),
                pct_le1_var=("test_pct_le1", lambda x: x.var(ddof=0)),
                n_runs=("test_rmse", "count"),
            )
            .reset_index()
        )
        agg.to_csv(out_path.replace(".csv", "_summary.csv"), index=False)
        print("\n[Summary]")
        print(agg)
    else:
        print("\n[Runs]")
        print(df)
    return df


def run_repeated_experiments(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    columns = MixColumns(
        solute=args.solute_col,
        solvent1=args.solvent1_col,
        solvent2=args.solvent2_col,
        x1=args.x1_col,
        x2=args.x2_col,
        temperature=args.temperature_col,
        label=args.label_col,
    )
    dataset = build_mixture_dataset(args.mix_csv, columns, normalize_composition=not args.no_normalize_composition)

    task_dirs = parse_list(args.task_dirs)
    encoder_ckpts = make_ckpt_paths_from_pretrain(args.pretrain_base_dir, task_dirs, args.encoder_ckpt_name)
    mlp_hidden = parse_int_tuple(args.mlp_hidden)
    ensure_dir(args.out_root)

    if args.split_strategy == "both":
        split_strategies = ["random", "solvent_pair_disjoint"]
    else:
        split_strategies = [args.split_strategy]

    rows: List[Dict[str, Any]] = []
    for split_strategy in split_strategies:
        for run_idx in range(args.num_runs):
            seed = args.seed + run_idx
            exp_name = f"{split_strategy}_seed{seed}"
            out_dir = os.path.join(args.out_root, exp_name)
            print("\n" + "=" * 90)
            print(f"[Run] split={split_strategy} seed={seed} out_dir={out_dir}")
            print("=" * 90)

            stats = run_single_experiment(
                dataset=dataset,
                encoder_ckpts=encoder_ckpts,
                init_ckpt=args.init_ckpt,
                device=device,
                out_dir=out_dir,
                split_strategy=split_strategy,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=seed,
                batch_size=args.batch_size,
                emb_dim=args.emb_dim,
                solute_dropout=args.solute_dropout,
                solvent_dropout=args.solvent_dropout,
                mlp_hidden=mlp_hidden,
                mlp_dropout=args.mlp_dropout,
                t_embed_dim=args.t_embed_dim,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                freeze_encoders=args.freeze_encoders,
                freeze_fusions=args.freeze_fusions,
                freeze_combiner=args.freeze_combiner,
                num_workers=args.num_workers,
                fusion_type=args.fusion_type,
                use_global_features=args.use_global_features,
            )
            stats["experiment"] = args.experiment_name
            rows.append(stats)

    summary_path = os.path.join(args.out_root, "summary_runs.csv")
    return summarize_runs(rows, summary_path)


def run_ablation(args: argparse.Namespace) -> pd.DataFrame:
    """Run leave-space/subset ablations under the same updated framework.

    Each subset chooses a subset of prior branches, e.g. 0_1_2_3 or 0_1_3.
    The model remains the same composition-weighted pooling model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    columns = MixColumns(
        solute=args.solute_col,
        solvent1=args.solvent1_col,
        solvent2=args.solvent2_col,
        x1=args.x1_col,
        x2=args.x2_col,
        temperature=args.temperature_col,
        label=args.label_col,
    )
    dataset = build_mixture_dataset(args.mix_csv, columns, normalize_composition=not args.no_normalize_composition)

    task_dirs = parse_list(args.task_dirs)
    all_encoder_ckpts = make_ckpt_paths_from_pretrain(args.pretrain_base_dir, task_dirs, args.encoder_ckpt_name)
    mlp_hidden = parse_int_tuple(args.mlp_hidden)
    subsets = parse_list(args.subsets)
    ensure_dir(args.out_root)

    rows: List[Dict[str, Any]] = []
    for subset in subsets:
        idxs = [int(x) for x in subset.split("_") if x.strip()]
        if any(i < 0 or i >= len(all_encoder_ckpts) for i in idxs):
            raise ValueError(f"Subset {subset} out of range for {len(all_encoder_ckpts)} encoders.")
        encoder_ckpts = [all_encoder_ckpts[i] for i in idxs]

        # Optional subset-specific pure-solvent initialization.
        init_ckpt = args.init_ckpt
        if args.ablation_init_dir:
            candidate = os.path.join(args.ablation_init_dir, f"subset_{subset}", args.ablation_init_name)
            if os.path.isfile(candidate):
                init_ckpt = candidate
            else:
                print(f"[Ablation] subset init not found, using --init_ckpt: {candidate}")

        for run_idx in range(args.num_runs):
            seed = args.seed + run_idx
            out_dir = os.path.join(args.out_root, f"subset_{subset}", f"seed{seed}")
            print("\n" + "=" * 90)
            print(f"[Ablation] subset={subset} seed={seed} init={init_ckpt}")
            print("=" * 90)

            stats = run_single_experiment(
                dataset=dataset,
                encoder_ckpts=encoder_ckpts,
                init_ckpt=init_ckpt,
                device=device,
                out_dir=out_dir,
                split_strategy=args.split_strategy,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=seed,
                batch_size=args.batch_size,
                emb_dim=args.emb_dim,
                solute_dropout=args.solute_dropout,
                solvent_dropout=args.solvent_dropout,
                mlp_hidden=mlp_hidden,
                mlp_dropout=args.mlp_dropout,
                t_embed_dim=args.t_embed_dim,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                freeze_encoders=args.freeze_encoders,
                freeze_fusions=args.freeze_fusions,
                freeze_combiner=args.freeze_combiner,
                num_workers=args.num_workers,
                fusion_type=args.fusion_type,
                use_global_features=args.use_global_features,
            )
            stats["experiment"] = f"subset_{subset}"
            stats["subset"] = subset
            stats["subset_indices"] = "_".join(map(str, idxs))
            stats["init_ckpt"] = init_ckpt or ""
            rows.append(stats)

    summary_path = os.path.join(args.out_root, "summary_ablation.csv")
    return summarize_runs(rows, summary_path)


# ============================================================
# CLI
# ============================================================


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mix_csv", type=str, default="data/pair_data/mix.csv")
    parser.add_argument("--solute_col", type=str, default="solute_smiles")
    parser.add_argument("--solvent1_col", type=str, default="solvent1_smiles")
    parser.add_argument("--solvent2_col", type=str, default="solvent2_smiles")
    parser.add_argument("--x1_col", type=str, default="Xsolvent1")
    parser.add_argument("--x2_col", type=str, default="Xsolvent2")
    parser.add_argument("--temperature_col", type=str, default="temperature")
    parser.add_argument("--label_col", type=str, default="logS")
    parser.add_argument("--no_normalize_composition", action="store_true")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrain_base_dir", type=str, default="ckpt/tau_0.1__scale_0.25")
    parser.add_argument("--task_dirs", type=str, default="task_0,task_1,task_2,task_none")
    parser.add_argument("--encoder_ckpt_name", type=str, default="best_model.pth")
    parser.add_argument("--init_ckpt", type=str, default="", help="Pure-solvent fusion checkpoint; {seed} is replaced for each run.")
    parser.add_argument("--emb_dim", type=int, default=512, help="Legacy argument; actual width is read from checkpoints.")
    parser.add_argument("--fusion_type", choices=["static", "adaptive"], default="adaptive")
    parser.add_argument("--use_global_features", action="store_true", help="Opt in to molecular global features.")
    parser.add_argument("--solute_dropout", type=float, default=0.3)
    parser.add_argument("--solvent_dropout", type=float, default=0.3)
    parser.add_argument("--mlp_hidden", type=str, default="1024,128,64")
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--t_embed_dim", type=int, default=32)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    # Default: freeze PRISM encoders, train only the
    # lightweight mixed-solvent head/fusion unless flags are changed.
    parser.add_argument("--freeze_encoders", action="store_true", default=True)
    parser.add_argument("--unfreeze_encoders", dest="freeze_encoders", action="store_false")
    parser.add_argument("--freeze_fusions", action="store_true", default=False)
    parser.add_argument("--freeze_combiner", action="store_true", default=False)


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="solvent_pair_disjoint",
        choices=["random", "solvent_pair_disjoint", "pair_disjoint", "pair", "both"],
    )
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRISM mixed-solvent fine-tuning.")
    sub = parser.add_subparsers(dest="mode", required=True)

    run_parser = sub.add_parser("run", help="Run repeated mixed-solvent experiments.")
    add_data_args(run_parser)
    add_model_args(run_parser)
    add_training_args(run_parser)
    add_split_args(run_parser)
    run_parser.add_argument("--num_runs", type=int, default=3)
    run_parser.add_argument("--out_root", type=str, default="results/mix_prism")
    run_parser.add_argument("--experiment_name", type=str, default="PRISM")

    ab_parser = sub.add_parser("ablation", help="Run branch-subset ablations.")
    add_data_args(ab_parser)
    add_model_args(ab_parser)
    add_training_args(ab_parser)
    add_split_args(ab_parser)
    ab_parser.add_argument("--num_runs", type=int, default=3)
    ab_parser.add_argument("--out_root", type=str, default="results/mix_prism_ablation")
    ab_parser.add_argument("--subsets", type=str, default="0_1_2_3,0_1_2,0_1_3,0_2_3,1_2_3")
    ab_parser.add_argument("--ablation_init_dir", type=str, default="")
    ab_parser.add_argument("--ablation_init_name", type=str, default="model_best.pth")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # argparse returns an empty string for optional ckpt paths by default.
    if hasattr(args, "init_ckpt") and args.init_ckpt == "":
        args.init_ckpt = None
    if hasattr(args, "ablation_init_dir") and args.ablation_init_dir == "":
        args.ablation_init_dir = None

    if args.mode == "run":
        run_repeated_experiments(args)
    elif args.mode == "ablation":
        run_ablation(args)
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
