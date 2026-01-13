import os
import math
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from argparse import Namespace
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from torch_geometric.data import Data, DataLoader

# === RDKit ===
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RDKFingerprint
_HAS_RDKIT = True

from chemprop.models.mpn import MPN


class FilterEmptyGraphs(torch.utils.data.Dataset):
    """Dataset wrapper to filter out empty graphs"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.filtered_dataset = self._filter_empty_graphs()

    def _filter_empty_graphs(self):
        return [data for data in self.dataset 
                if data.x is not None and data.x.shape[0] > 0 
                and data.edge_index is not None and data.edge_index.shape[1] > 0]

    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, idx):
        return self.filtered_dataset[idx]


class GNNModelWithNewLoss(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features,
                 cov_num=3, hidden_dim=256, dropout_rate=0.3, batch_size=512,
                 datasize=False, device=None, property_index=0,
                 loss_weights={'mse':1, 'rank':0}, save_path="models",
                 # fingerprint config
                 fp_type='morgan', fp_radius=2, fp_nbits=2048, fp_use_chirality=True,
                 tau = None, tau_physical = None, tau_phys_scale = 0.5,
                 ):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.batch_size = int(batch_size)
        self.datasize = bool(datasize)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.property_index = property_index
        self.loss_weights = loss_weights
        self.save_path = save_path
        self.cov_num = int(cov_num)

        self.rdkit_ok = _HAS_RDKIT
        self.fp_type = fp_type
        self.fp_radius = int(fp_radius)
        self.fp_nbits = int(fp_nbits)
        self.fp_use_chirality = bool(fp_use_chirality)
        self._fp_cache = {}   # {smiles: ExplicitBitVect}

        self.tau = tau
        self.tau_physical = tau_physical
        self.tau_phys_scale = tau_phys_scale

        # ----- encoder: chemprop MPN -----
        mpn_args = Namespace(
            hidden_size=self.hidden_dim,
            atom_messages=True,     
            bias=True,
            depth=max(1, self.cov_num),
            dropout=self.dropout_rate,
            undirected=False,
            activation='ReLU',
            no_cache=False
        )
        self._chemprop_args = mpn_args
        self.mpn = MPN(mpn_args, atom_fdim=num_node_features, bond_fdim=num_edge_features)

        self.global_encoder = nn.Linear(num_global_features, 32) if num_global_features > 0 else None

        proj_input_dim = self.hidden_dim + (32 if num_global_features > 0 else 0)
        self.projection_head = nn.Sequential(
            nn.Linear(proj_input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 64)
        )

        self.loss_method = "sampling" if self.datasize else "full_combination"
        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_weights = []

    # -------------------- Utilities --------------------
    def get_property(self, batch, which=None, field=None):

        if field is None:
            if which is None:
                which = self.property_index
            if which is None:
                return None  # fingerprint 
            if isinstance(which, str):
                map_name = {'smr': 0, 'slogp': 1, 'peoe': 2}
                which = map_name.get(which.lower(), which)
            field = f"property_{int(which)}"

        if isinstance(batch, list):
            props = self._stack_field_from_list(batch, field) 
        else:
            props = getattr(batch, field, None)
            if props is None and hasattr(batch, "to_data_list"):
                props = self._stack_field_from_list(batch.to_data_list(), field)

        if props is None:
            print(f"[warn] get_property: missing '{field}'")
            return None

        props = torch.as_tensor(props, dtype=torch.float32)
        if props.dim() == 1:
            props = props.unsqueeze(0)                          # [D] -> [1, D]
        elif props.dim() >= 2 and props.size(0) == 1 and isinstance(batch, list) and len(batch) > 1:
           
            props = props.expand(len(batch), -1)

        return props


    def _zscore(self, X, eps=1e-6):
        mean = X.mean(dim=0, keepdim=True)
        std  = X.std(dim=0, keepdim=True)
        return (X - mean) / (std + eps)

    def _pairwise_euclid(self, X):
        X2 = (X * X).sum(dim=1, keepdim=True)         # [B,1]
        d2 = X2 + X2.t() - 2.0 * (X @ X.t())
        return (d2.clamp_min(0.0) + 1e-12).sqrt()

    def _auto_tau_phys_from_D(self, D, scale=0.5, min_tau=1e-3, max_tau=1e6):
        mask = torch.isfinite(D) & (~torch.eye(D.size(0), device=D.device, dtype=torch.bool))
        vals = D[mask]
        if vals.numel() == 0:
            tau = max(min_tau, 1.0 * scale)
        else:
            med = torch.median(vals)
            tau = torch.clamp(med * scale, min=min_tau, max=max_tau)
        return tau

    # ---------- SMILES -> Tanimoto distance ----------
    def _get_smiles_list(self, batch):
        if hasattr(batch, 'smiles'):
            smi = batch.smiles
            if isinstance(smi, (list, tuple)):
                return list(smi)
            if isinstance(smi, np.ndarray):
                return [str(x) for x in smi.tolist()]
            if isinstance(smi, str):
                return [smi]
        return None

    def _fingerprint(self, smi):
        if smi in self._fp_cache:
            return self._fp_cache[smi]
        mol = Chem.MolFromSmiles(smi) if self.rdkit_ok else None
        if mol is None:
            self._fp_cache[smi] = None
            return None
        if self.fp_type.lower() == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_radius, nBits=self.fp_nbits,
                useChirality=self.fp_use_chirality
            )
        elif self.fp_type.lower() == 'rdk':
            fp = RDKFingerprint(mol, fpSize=self.fp_nbits)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_radius, nBits=self.fp_nbits,
                useChirality=self.fp_use_chirality
            )
        self._fp_cache[smi] = fp
        return fp

    def _chemspace_distance_from_smiles(self, smiles_list):
        B = len(smiles_list)
        D = torch.full((B, B), float('inf'), device=self.device)
        if not self.rdkit_ok:
            raise RuntimeError("RDKit uninstalled")

        fps = [self._fingerprint(s) for s in smiles_list]
        for i in range(B):
            fpi = fps[i]
            if fpi is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fpi, fps)
            sims = [0.0 if (fpj is None) else sims[j] for j, fpj in enumerate(fps)]
            row = 1.0 - np.asarray(sims, dtype=np.float32)
            D[i] = torch.tensor(row, device=self.device)

        D.fill_diagonal_(0.0)
        return D

    # ---------- Soft labels ----------
    def _soft_labels_from_distance(self, D, tau_phys=None, tau_phys_scale=0.5,
                                   topk=None, topk_ratio=None):
        tau_phys_scale = self.tau_phys_scale
        tau_phys = self.tau_physical
        B = D.size(0)
        dev = D.device
        D = D.clone()
        D.fill_diagonal_(float('inf'))

        if topk is None and topk_ratio is not None:
            k = max(1, int((B - 1) * float(topk_ratio)))
        else:
            k = topk
        if k is not None and k < B - 1:
            nn_idx = torch.topk(D, k=k, dim=1, largest=False).indices  # [B, k]
            mask = torch.zeros_like(D, dtype=torch.bool, device=dev)
            rows = torch.arange(B, device=dev).unsqueeze(1).expand_as(nn_idx)
            mask[rows, nn_idx] = True
            D = D.masked_fill(~mask, float('inf'))

        if tau_phys is None:
            tau_val = self._auto_tau_phys_from_D(D, scale=tau_phys_scale)
        else:
            tau_val = max(tau_phys, 1e-6)

        W = torch.exp(-D / tau_val)
        W = W.masked_fill(torch.eye(B, device=dev, dtype=torch.bool), 0.0)
        W = W + (1.0 - torch.eye(B, device=dev)) * 1e-12
        Y = W / (W.sum(dim=1, keepdim=True))
        return Y, tau_val

    def _soft_nce_loss_from_distance(self, z, D_phys, tau=0.1,
                                     tau_phys=None, tau_phys_scale=0.5,
                                     topk=None, topk_ratio=None):
        tau_phys_scale = self.tau_phys_scale
        tau_phys = self.tau_physical
        tau = self.tau if self.tau is not None else tau
        B = z.size(0)
        if B < 2:
            return torch.zeros((), device=z.device)
        Y, _ = self._soft_labels_from_distance(
            D_phys, tau_phys=tau_phys, tau_phys_scale=tau_phys_scale,
            topk=topk, topk_ratio=topk_ratio
        )
        z = F.normalize(z, dim=1)
        logits = (z @ z.t()) / max(tau, 1e-6)
        logits = logits.masked_fill(torch.eye(B, device=z.device, dtype=torch.bool), -1e9)
        logP = F.log_softmax(logits, dim=1)
        return -(Y.detach() * logP).sum(dim=1).mean()

    def _soft_labels_from_props(self, props, tau_phys=None, tau_phys_scale=0.5,
                                topk=None, topk_ratio=None, standardize=True):
        tau_phys_scale = self.tau_phys_scale
        tau_phys = self.tau_physical
        dev = props.device
        props = props.float()
        if standardize:
            props = self._zscore(props)
        D = self._pairwise_euclid(props)                     # [B,B]
        return self._soft_labels_from_distance(
            D, tau_phys=tau_phys, tau_phys_scale=tau_phys_scale,
            topk=topk, topk_ratio=topk_ratio
        )

    def _soft_nce_loss_from_props(self, z, props, tau=0.1, tau_phys=None, tau_phys_scale=0.5,
                                  topk=None, topk_ratio=None, standardize=True):
        tau_phys_scale = self.tau_phys_scale
        tau_phys = self.tau_physical
        tau = self.tau if self.tau is not None else tau
        B = z.size(0)
        if B < 2:
            return torch.zeros((), device=z.device)
        Y, _ = self._soft_labels_from_props(
            props, tau_phys=tau_phys, tau_phys_scale=tau_phys_scale,
            topk=topk, topk_ratio=topk_ratio, standardize=standardize
        )
        z = F.normalize(z, dim=1)
        logits = (z @ z.t()) / max(tau, 1e-6)
        logits = logits.masked_fill(torch.eye(B, device=z.device, dtype=torch.bool), -1e9)
        logP   = F.log_softmax(logits, dim=1)
        return -(Y.detach() * logP).sum(dim=1).mean()

    def _metric_reg_loss(self, z, props=None, D_pre=None, mode='sampled',
                         num_near=8, num_far=8, num_rand=8,
                         standardize=True, alpha=None):
        dev = z.device
        B   = z.size(0)
        if B < 2:
            return torch.zeros((), device=dev)

        if D_pre is None:
            assert props is not None, "D_pre must be provided if props is None"
            P = props.float()
            if standardize:
                P = self._zscore(P)
            Dv = self._pairwise_euclid(P)                       # [B,B]
        else:
            Dv = D_pre.clone()
        Dz = self._pairwise_euclid(z)                           # [B,B]

        if mode == 'full':
            iu, ju = torch.triu_indices(B, B, offset=1, device=dev)
            dv = Dv[iu, ju]
            dz = Dz[iu, ju]
        else:
            idx_sorted = Dv.argsort(dim=1)                      # [B,B]
            k_near = min(num_near, max(B-1, 1))
            k_far  = min(num_far,  max(B-1, 1))
            near_idx = idx_sorted[:, 1:1+k_near]
            far_idx  = idx_sorted[:, -k_far:]
            if num_rand > 0:
                ridx = torch.randint(low=1, high=B, size=(B, num_rand), device=dev)
                ridx = (ridx + torch.arange(B, device=dev).unsqueeze(1)) % B
            else:
                ridx = torch.empty(B, 0, dtype=torch.long, device=dev)
            pairs = torch.cat([near_idx, far_idx, ridx], dim=1)  # [B,K]
            rows  = torch.arange(B, device=dev).unsqueeze(1).expand_as(pairs)
            dv = Dv[rows.reshape(-1), pairs.reshape(-1)]
            dz = Dz[rows.reshape(-1), pairs.reshape(-1)]

        if alpha is None:
            alpha = ((dz * dv) / (dv.pow(2) + 1e-9)).mean().detach()
        return F.mse_loss(dz, alpha * dv)

    # -------------------- Encoder and Projection --------------------
    def forward(self, data):
        if hasattr(data, "to_data_list"):         # PyG Batch
            pyg_list = data.to_data_list()
        elif isinstance(data, list):              # 你自定义 collate 返回 list[Data]
            pyg_list = data
        else:                                     # 单个 Data
            pyg_list = [data]

        graph_embedding = self.mpn(prompt=False, batch=pyg_list)   # [B, hidden]

        if self.global_encoder is not None:
            gf = self._stack_field_from_list(pyg_list, "global_features")
            if gf is not None:
                gf = gf.to(graph_embedding.device)
                if gf.dim() == 1:
                    gf = gf.view(1, -1).expand(graph_embedding.size(0), -1)
                elif gf.dim() == 3 and gf.size(1) == 1:
                    gf = gf.squeeze(1)
                elif gf.dim() == 2 and gf.size(0) == 1 and graph_embedding.size(0) > 1:
                    gf = gf.expand(graph_embedding.size(0), -1)
                graph_embedding = torch.cat([graph_embedding, self.global_encoder(gf)], dim=1)

        return graph_embedding



    # -------------------- New Loss: property space or fingerprint space --------------------
    def _zero_like_params(self):
        # 值为 0，但和参数图连通，可反传；避免 no-grad 报错
        return next(self.parameters()).sum() * 0.0

    def _stack_field_from_list(self, batch, field_name):
        """支持 list[Data] / DataBatch / Data 三种输入，把每个图的张量字段拼成 [B, ...]"""
        import numpy as np
        if isinstance(batch, list):
            items = []
            for d in batch:
                val = getattr(d, field_name, None)
                if val is None:
                    return None
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                if not isinstance(val, torch.Tensor):
                    # 单个标量/列表等，尽量转 tensor
                    val = torch.as_tensor(val)
                # 常见是 [1, D]，这里统一挤掉 batch 维再堆叠
                if val.dim() >= 2 and val.size(0) == 1:
                    val = val.squeeze(0)
                items.append(val)
            # 堆成 [B, ...]
            return torch.stack(items, dim=0)
        else:
            # DataBatch 或 Data 直接取
            return getattr(batch, field_name, None)

    def _get_smiles_list(self, batch):
        """在 list[Data] / Batch / Data 上统一取 smiles，返回 List[str]"""
        import numpy as np
        if isinstance(batch, list):
            out = []
            for d in batch:
                s = getattr(d, "smiles", None)
                if s is None:
                    return None
                if isinstance(s, str):
                    out.append(s)
                elif isinstance(s, (list, tuple)):
                    out.append(str(s[0]))
                elif isinstance(s, np.ndarray):
                    out.append(str(s.flatten()[0]))
                else:
                    out.append(str(s))
            return out
        # DataBatch / Data 的老逻辑
        if hasattr(batch, 'smiles'):
            smi = batch.smiles
            if isinstance(smi, (list, tuple)):
                return list(smi)
            if isinstance(smi, np.ndarray):
                return [str(x) for x in smi.tolist()]
            if isinstance(smi, str):
                return [smi]
        return None

    def get_loss(self, batch, temperature=0.1,
                tau_phys=None, tau_phys_scale=0.5,
                topk=64, topk_ratio=None,
                lambda_metric=0.1, metric_mode='sampled',
                num_near=8, num_far=8, num_rand=8,
                standardize_props=True):
        # 先算 z，用它来判断 B，兼容 list[Data]
        temperature = self.tau
        tau_phys = self.tau_physical
        tau_phys_scale = self.tau_phys_scale
        z = self.projection_head(self.forward(batch))  # [B, Demb]
        B = int(z.size(0))
        if B < 2:
            return self._zero_like_params()

        if self.property_index is None:
            # ---- fingerprint 路径 ----
            smiles_list = self._get_smiles_list(batch)
            if (not self.rdkit_ok) or (smiles_list is None):
                return self._zero_like_params()
            D_fp = self._chemspace_distance_from_smiles(smiles_list).to(z.device)  # [B, B]
            loss_soft = self._soft_nce_loss_from_distance(
                z, D_fp, tau=temperature, tau_phys=tau_phys,
                tau_phys_scale=tau_phys_scale, topk=topk, topk_ratio=topk_ratio
            )
            loss_metric = self._metric_reg_loss(
                z, D_pre=D_fp, mode=metric_mode,
                num_near=num_near, num_far=num_far, num_rand=num_rand
            )
        else:
            # ---- property 路径 ----
            props = self.get_property(batch)
            if props is None:
                return self._zero_like_params()
            if not isinstance(props, torch.Tensor):
                props = torch.as_tensor(props)
            if props.dim() == 1:
                props = props.unsqueeze(1)         # [B] -> [B,1]
            if props.size(0) != B:
                return self._zero_like_params()
            props = props.to(z.device).float()
            loss_soft = self._soft_nce_loss_from_props(
                z, props, tau=temperature, tau_phys=tau_phys, tau_phys_scale=tau_phys_scale,
                topk=topk, topk_ratio=topk_ratio, standardize=standardize_props
            )
            loss_metric = self._metric_reg_loss(
                z, props=props, mode=metric_mode,
                num_near=num_near, num_far=num_far, num_rand=num_rand,
                standardize=standardize_props
            )

        print(f"[dbg] soft={loss_soft.item():.4f} metric={loss_metric.item():.4f} |z|={z.norm(dim=1).mean().item():.3f}")

        loss = loss_soft + lambda_metric * loss_metric * 0
        return loss

