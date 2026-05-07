#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChemSpaceA / PRISM prior-alignment encoder.

This module implements the model used for prior-alignment pre-training in the
current ChemSpaceA framework.

The training objective is a single population-level soft-neighborhood alignment
loss:

    prior distances  ->  soft prior neighborhoods
    embedding cosine similarities -> soft embedding neighborhoods
    loss = KL(prior neighborhood || embedding neighborhood)

Supported prior spaces:
    1. Descriptor prior spaces, e.g. SMR_VSA, SlogP_VSA, PEOE_VSA
       stored as property_0, property_1, property_2, ...
    2. Fingerprint prior space, using 1 - Tanimoto distance from SMILES.

The module intentionally removes earlier experimental traces such as
rank losses, distance-regression losses, and alternative loss-selection logic.
"""

from argparse import Namespace
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.models.mpn import MPN

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, RDKFingerprint
    _HAS_RDKIT = True
except Exception:
    Chem = None
    DataStructs = None
    AllChem = None
    RDKFingerprint = None
    _HAS_RDKIT = False


class FilterEmptyGraphs(torch.utils.data.Dataset):
    """Dataset wrapper that removes empty molecular graphs."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.filtered_dataset = self._filter_empty_graphs()

    def _filter_empty_graphs(self):
        return [
            data
            for data in self.dataset
            if data.x is not None
            and data.x.shape[0] > 0
            and data.edge_index is not None
            and data.edge_index.shape[1] > 0
        ]

    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, idx):
        return self.filtered_dataset[idx]


class ChemSpaceAPriorEncoder(nn.Module):
    """
    CMPNN-style molecular graph encoder for ChemSpaceA prior-alignment pre-training.

    Parameters
    ----------
    num_node_features:
        Atom feature dimension.
    num_edge_features:
        Bond feature dimension.
    num_global_features:
        Optional graph-level feature dimension. If positive, global features are
        encoded and concatenated to the graph representation before projection.
    property_index:
        Which descriptor prior to use. For example:
            0 -> property_0, usually SMR_VSA
            1 -> property_1, usually SlogP_VSA
            2 -> property_2, usually PEOE_VSA
        Use None for fingerprint prior alignment.
    tau:
        Temperature for embedding-neighborhood softmax.
    tau_physical:
        Temperature for prior-space neighborhoods. If None, it is estimated from
        the median off-diagonal prior distance in the current batch.
    tau_phys_scale:
        Scale factor for the automatically estimated prior-space temperature.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_global_features: int,
        cov_num: int = 3,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        batch_size: int = 512,
        device: Optional[torch.device] = None,
        property_index: Optional[Union[int, str]] = 0,
        save_path: str = "models",
        fp_type: str = "morgan",
        fp_radius: int = 2,
        fp_nbits: int = 2048,
        fp_use_chirality: bool = True,
        tau: Optional[float] = None,
        tau_physical: Optional[float] = None,
        tau_phys_scale: float = 0.5,
        projection_dim: int = 64,
    ):
        super().__init__()

        self.num_node_features = int(num_node_features)
        self.num_edge_features = int(num_edge_features)
        self.num_global_features = int(num_global_features)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.batch_size = int(batch_size)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.property_index = self._normalize_property_index(property_index)
        self.save_path = save_path

        self.fp_type = fp_type
        self.fp_radius = int(fp_radius)
        self.fp_nbits = int(fp_nbits)
        self.fp_use_chirality = bool(fp_use_chirality)
        self._fp_cache = {}

        self.tau = tau
        self.tau_physical = tau_physical
        self.tau_phys_scale = float(tau_phys_scale)

        mpn_args = Namespace(
            hidden_size=self.hidden_dim,
            atom_messages=True,
            bias=True,
            depth=max(1, int(cov_num)),
            dropout=self.dropout_rate,
            undirected=False,
            activation="ReLU",
            no_cache=False,
        )
        self._chemprop_args = mpn_args
        self.mpn = MPN(
            mpn_args,
            atom_fdim=self.num_node_features,
            bond_fdim=self.num_edge_features,
        )

        self.global_encoder = (
            nn.Linear(self.num_global_features, 32)
            if self.num_global_features > 0
            else None
        )

        proj_input_dim = self.hidden_dim + (32 if self.global_encoder is not None else 0)
        self.projection_head = nn.Sequential(
            nn.Linear(proj_input_dim, max(1, self.hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(max(1, self.hidden_dim // 2), int(projection_dim)),
        )

    # ------------------------------------------------------------------
    # Batch field helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_property_index(property_index):
        if property_index is None:
            return None
        if isinstance(property_index, str):
            name_to_idx = {
                "smr": 0,
                "smr_vsa": 0,
                "slogp": 1,
                "slogp_vsa": 1,
                "peoe": 2,
                "peoe_vsa": 2,
                "fp": None,
                "fingerprint": None,
                "morgan": None,
            }
            key = property_index.lower()
            return name_to_idx.get(key, int(property_index) if key.isdigit() else property_index)
        return int(property_index)

    @staticmethod
    def _as_data_list(batch):
        if hasattr(batch, "to_data_list"):
            return batch.to_data_list()
        if isinstance(batch, list):
            return batch
        return [batch]

    @staticmethod
    def _stack_field_from_list(batch, field_name: str):
        """
        Stack a tensor-like graph field from a list[Data], PyG Batch, or single Data.
        Returns a tensor of shape [B, ...] when possible.
        """
        if hasattr(batch, "to_data_list"):
            batch = batch.to_data_list()

        if isinstance(batch, list):
            values = []
            for data in batch:
                value = getattr(data, field_name, None)
                if value is None:
                    return None
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                elif not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)

                if value.dim() >= 2 and value.size(0) == 1:
                    value = value.squeeze(0)
                values.append(value)

            return torch.stack(values, dim=0)

        return getattr(batch, field_name, None)

    def get_property(self, batch, which=None, field: Optional[str] = None):
        """
        Read a descriptor prior from a batch.

        By default, the field is property_{self.property_index}.
        """
        if field is None:
            if which is None:
                which = self.property_index
            if which is None:
                return None
            which = self._normalize_property_index(which)
            field = f"property_{int(which)}"

        props = self._stack_field_from_list(batch, field)
        if props is None:
            return None

        props = torch.as_tensor(props, dtype=torch.float32)
        if props.dim() == 1:
            props = props.unsqueeze(0)

        return props

    @staticmethod
    def _get_smiles_list(batch) -> Optional[List[str]]:
        """
        Read SMILES strings from list[Data], PyG Batch, or single Data.
        """
        if hasattr(batch, "to_data_list"):
            batch = batch.to_data_list()

        if isinstance(batch, list):
            smiles = []
            for data in batch:
                smi = getattr(data, "smiles", None)
                if smi is None:
                    return None
                if isinstance(smi, str):
                    smiles.append(smi)
                elif isinstance(smi, (list, tuple)):
                    smiles.append(str(smi[0]))
                elif isinstance(smi, np.ndarray):
                    smiles.append(str(smi.flatten()[0]))
                else:
                    smiles.append(str(smi))
            return smiles

        if hasattr(batch, "smiles"):
            smi = batch.smiles
            if isinstance(smi, str):
                return [smi]
            if isinstance(smi, (list, tuple)):
                return [str(x) for x in smi]
            if isinstance(smi, np.ndarray):
                return [str(x) for x in smi.tolist()]

        return None

    # ------------------------------------------------------------------
    # Prior-space distance construction
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + eps)

    @staticmethod
    def _pairwise_euclid(x: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        d2 = x2 + x2.t() - 2.0 * (x @ x.t())
        return (d2.clamp_min(0.0) + 1e-12).sqrt()

    @staticmethod
    def _auto_tau_phys_from_distance(
        distance: torch.Tensor,
        scale: float = 0.5,
        min_tau: float = 1e-3,
        max_tau: float = 1e6,
    ) -> torch.Tensor:
        eye = torch.eye(distance.size(0), device=distance.device, dtype=torch.bool)
        mask = torch.isfinite(distance) & (~eye)
        values = distance[mask]

        if values.numel() == 0:
            tau = torch.tensor(float(scale), device=distance.device)
        else:
            tau = torch.median(values) * float(scale)

        return torch.clamp(tau, min=min_tau, max=max_tau)

    def _fingerprint(self, smiles: str):
        if smiles in self._fp_cache:
            return self._fp_cache[smiles]

        if not _HAS_RDKIT:
            raise RuntimeError("RDKit is required for fingerprint prior alignment.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self._fp_cache[smiles] = None
            return None

        if self.fp_type.lower() == "rdk":
            fp = RDKFingerprint(mol, fpSize=self.fp_nbits)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.fp_radius,
                nBits=self.fp_nbits,
                useChirality=self.fp_use_chirality,
            )

        self._fp_cache[smiles] = fp
        return fp

    def _fingerprint_distance_from_smiles(self, smiles_list: Sequence[str]) -> torch.Tensor:
        if not _HAS_RDKIT:
            raise RuntimeError("RDKit is required for fingerprint prior alignment.")

        batch_size = len(smiles_list)
        distance = torch.full(
            (batch_size, batch_size),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )

        fps = [self._fingerprint(smi) for smi in smiles_list]
        for i, fp_i in enumerate(fps):
            if fp_i is None:
                continue

            sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps)
            row = [
                0.0 if fp_j is None else float(sims[j])
                for j, fp_j in enumerate(fps)
            ]
            distance[i] = torch.tensor(1.0 - np.asarray(row, dtype=np.float32), device=self.device)

        distance.fill_diagonal_(0.0)
        return distance

    def _descriptor_distance(
        self,
        props: torch.Tensor,
        standardize: bool = True,
    ) -> torch.Tensor:
        props = props.float()
        if props.dim() == 1:
            props = props.unsqueeze(1)
        if standardize:
            props = self._zscore(props)
        return self._pairwise_euclid(props)

    # ------------------------------------------------------------------
    # Soft-neighborhood alignment objective
    # ------------------------------------------------------------------
    def _soft_targets_from_distance(
        self,
        distance: torch.Tensor,
        tau_phys: Optional[float] = None,
        tau_phys_scale: Optional[float] = None,
        topk: Optional[int] = 64,
        topk_ratio: Optional[float] = None,
    ):
        """
        Convert a prior-space distance matrix into row-wise soft neighborhoods.
        The diagonal is excluded.
        """
        distance = distance.clone()
        device = distance.device
        batch_size = distance.size(0)

        eye = torch.eye(batch_size, device=device, dtype=torch.bool)
        distance = distance.masked_fill(eye, float("inf"))

        if topk is None and topk_ratio is not None:
            topk = max(1, int((batch_size - 1) * float(topk_ratio)))

        if topk is not None and topk < batch_size - 1:
            nn_idx = torch.topk(distance, k=topk, dim=1, largest=False).indices
            mask = torch.zeros_like(distance, dtype=torch.bool, device=device)
            rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(nn_idx)
            mask[rows, nn_idx] = True
            distance = distance.masked_fill(~mask, float("inf"))

        if tau_phys is None:
            scale = self.tau_phys_scale if tau_phys_scale is None else float(tau_phys_scale)
            tau_value = self._auto_tau_phys_from_distance(distance, scale=scale)
        else:
            tau_value = torch.tensor(max(float(tau_phys), 1e-6), device=device)

        weights = torch.exp(-distance / tau_value)
        weights = weights.masked_fill(eye, 0.0)

        # Numerical guard: if all distances are invalid for a row, assign a tiny
        # off-diagonal mass and renormalize.
        weights = weights + (~eye).float() * 1e-12
        targets = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        return targets, tau_value

    def _alignment_loss_from_distance(
        self,
        z: torch.Tensor,
        prior_distance: torch.Tensor,
        tau: Optional[float] = None,
        tau_phys: Optional[float] = None,
        tau_phys_scale: Optional[float] = None,
        topk: Optional[int] = 64,
        topk_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        KL(prior-neighborhood || embedding-neighborhood), implemented as
        cross-entropy with detached prior soft targets.
        """
        batch_size = z.size(0)
        if batch_size < 2:
            return self._zero_like_params()

        tau_value = self.tau if self.tau is not None else (0.1 if tau is None else float(tau))
        tau_phys_value = self.tau_physical if self.tau_physical is not None else tau_phys

        targets, _ = self._soft_targets_from_distance(
            prior_distance,
            tau_phys=tau_phys_value,
            tau_phys_scale=tau_phys_scale,
            topk=topk,
            topk_ratio=topk_ratio,
        )

        z = F.normalize(z, dim=1)
        logits = (z @ z.t()) / max(float(tau_value), 1e-6)

        eye = torch.eye(batch_size, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, -1e9)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets.detach() * log_probs).sum(dim=1).mean()
        return loss

    def _alignment_loss_from_descriptor(
        self,
        z: torch.Tensor,
        props: torch.Tensor,
        tau: Optional[float] = None,
        tau_phys: Optional[float] = None,
        tau_phys_scale: Optional[float] = None,
        topk: Optional[int] = 64,
        topk_ratio: Optional[float] = None,
        standardize: bool = True,
    ) -> torch.Tensor:
        prior_distance = self._descriptor_distance(props.to(z.device), standardize=standardize)
        return self._alignment_loss_from_distance(
            z,
            prior_distance,
            tau=tau,
            tau_phys=tau_phys,
            tau_phys_scale=tau_phys_scale,
            topk=topk,
            topk_ratio=topk_ratio,
        )

    # ------------------------------------------------------------------
    # Forward and loss
    # ------------------------------------------------------------------
    def forward(self, data):
        """
        Return the graph-level encoder representation before the projection head.
        This is the representation reused by downstream fine-tuning scripts.
        """
        data_list = self._as_data_list(data)
        graph_embedding = self.mpn(prompt=False, batch=data_list)

        if self.global_encoder is not None:
            global_features = self._stack_field_from_list(data_list, "global_features")
            if global_features is not None:
                global_features = global_features.to(graph_embedding.device)

                if global_features.dim() == 1:
                    global_features = global_features.view(1, -1).expand(graph_embedding.size(0), -1)
                elif global_features.dim() == 3 and global_features.size(1) == 1:
                    global_features = global_features.squeeze(1)
                elif global_features.dim() == 2 and global_features.size(0) == 1 and graph_embedding.size(0) > 1:
                    global_features = global_features.expand(graph_embedding.size(0), -1)

                graph_embedding = torch.cat(
                    [graph_embedding, self.global_encoder(global_features)],
                    dim=1,
                )

        return graph_embedding

    def encode_projected(self, data) -> torch.Tensor:
        """
        Return the projected representation used by the prior-alignment objective.
        """
        return self.projection_head(self.forward(data))

    def _zero_like_params(self) -> torch.Tensor:
        return next(self.parameters()).sum() * 0.0

    def get_loss(
        self,
        batch,
        temperature: Optional[float] = 0.1,
        tau_phys: Optional[float] = None,
        tau_phys_scale: Optional[float] = None,
        topk: Optional[int] = 64,
        topk_ratio: Optional[float] = None,
        standardize_props: bool = True,
    ) -> torch.Tensor:
        """
        Compute the ChemSpaceA prior-alignment loss for a batch.

        If property_index is an integer, descriptor prior alignment is used.
        If property_index is None, fingerprint prior alignment is used.
        """
        z = self.encode_projected(batch)
        batch_size = int(z.size(0))
        if batch_size < 2:
            return self._zero_like_params()

        if self.property_index is None:
            smiles_list = self._get_smiles_list(batch)
            if smiles_list is None:
                return self._zero_like_params()

            prior_distance = self._fingerprint_distance_from_smiles(smiles_list).to(z.device)
            return self._alignment_loss_from_distance(
                z,
                prior_distance,
                tau=temperature,
                tau_phys=tau_phys,
                tau_phys_scale=tau_phys_scale,
                topk=topk,
                topk_ratio=topk_ratio,
            )

        props = self.get_property(batch)
        if props is None:
            return self._zero_like_params()
        if props.size(0) != batch_size:
            return self._zero_like_params()

        return self._alignment_loss_from_descriptor(
            z,
            props,
            tau=temperature,
            tau_phys=tau_phys,
            tau_phys_scale=tau_phys_scale,
            topk=topk,
            topk_ratio=topk_ratio,
            standardize=standardize_props,
        )

GNNModelWithNewLoss = ChemSpaceAPriorEncoder