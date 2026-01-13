from argparse import Namespace
from typing import List, Union, Tuple, Optional, Sequence
from functools import reduce
import numpy as np
import torch
import torch.nn as nn

from chemprop.nn_utils import index_select_ND, get_activation_function

# =========================
# New: PyG list -> simple batched "mol-graph" adapter
# =========================

class SimpleBatchGraph:
    def __init__(self, data_list, ensure_undirected=True,
                 atom_fdim=None, bond_fdim=None, device=None):
        self.device = device or torch.device("cpu")

        atom_feats, bond_feats = [], []
        a_scope, b_scope = [], []

        # 1-based 全局偏移；0 留给 dummy
        atom_offset = 1
        bond_offset = 1

        # 先收集每个分子的 per-atom 邻接（待会儿一次性 pad）
        a2b_rows_all = []   # 将包含：第0行dummy + 每个原子一行
        a2a_rows_all = []

        # 先放一个 dummy 行（稍后 pad 会补0，不用填内容）
        a2b_rows_all.append([])   # 行0
        a2a_rows_all.append([])   # 行0

        for mol_idx, data in enumerate(data_list):
            x = data.x
            assert x.dim() == 2
            Na = x.size(0)
            if atom_fdim is not None:
                assert x.size(1) == atom_fdim

            edge_index = data.edge_index
            E = edge_index.size(1)
            edge_attr = getattr(data, "edge_attr", None)
            if edge_attr is None:
                if bond_fdim is None:
                    raise ValueError("edge_attr is None and bond_fdim is not specified.")
                edge_attr = torch.zeros((E, bond_fdim), dtype=torch.float32, device=x.device)
            else:
                if bond_fdim is not None:
                    assert edge_attr.size(1) == bond_fdim

            # 构造有向边（如需补反向）
            ei_src = edge_index[0].tolist()
            ei_dst = edge_index[1].tolist()
            pairs  = [(ei_src[i], ei_dst[i]) for i in range(E)]
            attrs  = [edge_attr[i] for i in range(E)]
            if ensure_undirected:
                s = set(pairs)
                for (u,v), attr in list(zip(pairs, attrs)):
                    if (v,u) not in s:
                        pairs.append((v,u))
                        attrs.append(attr.clone())

            Em = len(pairs)

            # 局部 bond 序号 -> 反向 bond 序号
            pair_to_local = {p:i for i,p in enumerate(pairs)}

            # per-atom 的局部列表
            a2b_local = [[] for _ in range(Na)]
            a2a_local = [[] for _ in range(Na)]
            for lb, (u,v) in enumerate(pairs):
                a2b_local[u].append(lb)
                a2a_local[u].append(v)

            # 特征先原样缓存，稍后统一在前面加 dummy 行
            atom_feats.append(x.cpu())
            bond_feats.append(torch.stack(attrs, dim=0).cpu() if Em>0
                              else torch.zeros((0, edge_attr.size(1)), dtype=edge_attr.dtype))

            # === 关键：把局部索引转成 1-based 的全局索引 ===
            # 原子全局 id：atom_offset .. atom_offset+Na-1  -> 1-based
            # 键   全局 id：bond_offset .. bond_offset+Em-1 -> 1-based
            for u in range(Na):
                # bond 索引 +1 再加偏移：1-based
                a2b_rows_all.append([bond_offset + lb for lb in a2b_local[u]])
                # atom 索引 +1 再加偏移：1-based
                a2a_rows_all.append([atom_offset + v  for v  in a2a_local[u]])

            # b2a / b2revb 也使用 1-based
            # 注意：这里先临时存，循环外再一次性 tensor 化
            if mol_idx == 0:
                b2a_list = []
                b2revb_list = []
            for lb, (u,v) in enumerate(pairs):
                b2a_list.append(atom_offset + v)                 # 1-based
                rev_local = pair_to_local.get((v,u), lb)
                b2revb_list.append(bond_offset + rev_local)      # 1-based

            # scope 也从 1 开始
            a_scope.append((atom_offset, Na))
            b_scope.append((bond_offset, Em))

            atom_offset += Na
            bond_offset += Em

        # 拼接特征，并在最前面加 dummy 行
        f_atoms = torch.cat(
            [torch.zeros((1, atom_feats[0].size(1))), *atom_feats], dim=0
        ).to(self.device)
        f_bonds = (torch.cat(
            [torch.zeros((1, bond_feats[0].size(1))), *bond_feats], dim=0
        ) if len(bond_feats)>0 else torch.zeros((1, bond_fdim))).to(self.device)

        # pad 到 [N_atoms_total+1, max_deg]，padding=0
        def pad_and_stack_rows(rows, pad_val=0):
            max_len = max((len(r) for r in rows), default=1)
            if max_len == 0: max_len = 1
            padded = [r + [pad_val]*(max_len-len(r)) for r in rows]
            return torch.tensor(padded, dtype=torch.long)

        a2b = pad_and_stack_rows(a2b_rows_all, pad_val=0).to(self.device)  # 1-based + 0 pad
        a2a = pad_and_stack_rows(a2a_rows_all, pad_val=0).to(self.device)  # 1-based + 0 pad

        b2a    = torch.tensor(b2a_list,    dtype=torch.long, device=self.device) if len(b2a_list)>0 else torch.zeros((0,), dtype=torch.long, device=self.device)
        b2revb = torch.tensor(b2revb_list, dtype=torch.long, device=self.device) if len(b2revb_list)>0 else torch.zeros((0,), dtype=torch.long, device=self.device)

        self._f_atoms  = f_atoms
        self._f_bonds  = f_bonds
        self._a2b      = a2b
        self._a2a      = a2a
        self._b2a      = b2a
        self._b2revb   = b2revb
        self._a_scope  = a_scope
        self._b_scope  = b_scope

    def get_components(self):
        return (self._f_atoms, self._f_bonds, self._a2b, self._b2a, self._b2revb,
                self._a_scope, self._b_scope, None)

    def get_a2a(self):
        return self._a2a



def to_mol_graph(batch, args: Namespace, prompt: bool):
    """
    Adapter used by MPN.forward: if `batch` is already a BatchMolGraph-like object (has get_components),
    return as is; otherwise if it's a list of PyG Data, build SimpleBatchGraph.
    """
    # If it already looks like a chemprop BatchMolGraph-like object
    if hasattr(batch, "get_components") and hasattr(batch, "get_a2a"):
        return batch

    if isinstance(batch, (list, tuple)) and len(batch) > 0 and hasattr(batch[0], "x") and hasattr(batch[0], "edge_index"):
        # Infer device from first Data.x (kept on CPU; MPNEncoder will .cuda() as in your current code)
        atom_fdim = args.atom_fdim if hasattr(args, "atom_fdim") and args.atom_fdim is not None else None
        bond_fdim = args.bond_fdim if hasattr(args, "bond_fdim") and args.bond_fdim is not None else None
        return SimpleBatchGraph(
            data_list=batch,
            ensure_undirected=True,
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            device=torch.device("cpu")
        )

    raise TypeError("Unsupported batch type: expected BatchMolGraph-like or list[PyG Data].")


# =========================
# MPN & Encoder (unchanged core math; only minor robustness tweaks)
# =========================

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.atom_messages = args.atom_messages
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.aggregation = 'mean'
        self.aggregation_norm = 100
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph, feature_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs (from SimpleBatchGraph or BatchMolGraph-like).

        Returns:
            Tensor of shape (num_molecules, hidden_size), i.e., per-molecule embeddings.
        """
        # Unpack components (CPU), then move to GPU (keeping your original .cuda() behavior)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, _ = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        if self.atom_messages:
            a2a = mol_graph.get_a2a().cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # [N_atom or N_bond] x hidden_size

        # Message passing
        for _ in range(self.depth - 1):
            if self.atom_messages:
                # Gather neighbor atom messages and bond features
                nei_a_message = index_select_ND(message, a2a)     # [num_atoms, max_deg, hidden]
                nei_f_bonds  = index_select_ND(f_bonds, a2b)      # [num_atoms, max_deg, bond_fdim]
                nei_message  = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # [num_atoms, max_deg, hidden + bond_fdim]
                message      = nei_message.sum(dim=1)              # [num_atoms, hidden + bond_fdim]
            else:
                # Bond messages
                nei_a_message = index_select_ND(message, a2b)      # [num_atoms, max_deg, hidden]
                a_message     = nei_a_message.sum(dim=1)           # [num_atoms, hidden]
                rev_message   = message[b2revb]                    # [num_bonds, hidden]
                message       = a_message[b2a] - rev_message       # [num_bonds, hidden]

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # [num_atoms, max_deg, hidden]
        a_message = nei_a_message.sum(dim=1)           # [num_atoms, hidden]
        a_input = torch.cat([f_atoms, a_message], dim=1)  # [num_atoms, atom_fdim + hidden]
        atom_hiddens = self.act_func(self.W_o(a_input))   # [num_atoms, hidden]
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout: per-molecule
        mol_vecs = []
        for (a_start, a_size) in a_scope:
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # [num_molecules, hidden]
        return mol_vecs


class MPN(nn.Module):
    """A wrapper around MPNEncoder which now accepts a list[PyG Data] batch directly (no PyBatchMolGraph)."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        super(MPN, self).__init__()
        self.args = args
        # If args doesn't carry atom_fdim/bond_fdim explicitly, you can pass them in ctor.
        # Otherwise, infer from first seen batch (not recommended). Here we rely on ctor args.
        assert atom_fdim is not None and bond_fdim is not None, \
            "Please provide atom_fdim and bond_fdim explicitly to MPN(...) when using PyG Data."
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        # So the encoder knows expected dimensions
        setattr(self.args, "atom_fdim", atom_fdim)
        setattr(self.args, "bond_fdim", bond_fdim)

        self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)

    def forward(self, prompt: bool, batch, features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        # batch is a list[Data] due to your collate_fn=lambda xs: xs
        batch_graph = to_mol_graph(batch, self.args, prompt)
        output = self.encoder.forward(batch_graph, features_batch)
        return output
