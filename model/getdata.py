import enum
if not hasattr(enum, 'StrEnum'):
    from enum import Enum
    class StrEnum(str, Enum):
        pass
    enum.StrEnum = StrEnum

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from model.atom import get_multi_hot_atom_featurizer
from model.bond import MultiHotBondFeaturizer

def get_global_features(mol):
    """
    Extract global molecular features from an RDKit mol object.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object

    Returns:
    --------
    np.ndarray
        Array of molecular descriptors
    """
    from rdkit.Chem import Descriptors, Lipinski

    if mol is None:
        return None

    try:
        features = []
        # Physical properties
        features.append(Descriptors.MolWt(mol))  # Molecular weight
        features.append(Descriptors.MolLogP(mol))  # LogP
        features.append(Descriptors.TPSA(mol))  # Topological polar surface area

        # Hydrogen bonding
        features.append(Lipinski.NumHDonors(mol))  # Number of H-bond donors
        features.append(Lipinski.NumHAcceptors(mol))  # Number of H-bond acceptors

        # Structural features
        features.append(Lipinski.NumRotatableBonds(mol))  # Number of rotatable bonds
        features.append(Descriptors.NumAromaticRings(mol))  # Number of aromatic rings
        features.append(Descriptors.NumAliphaticRings(mol))  # Number of aliphatic rings

        # Complexity and drug-likeness
        # features.append(Descriptors.FractionCSP3(mol))  # Fraction of sp3 hybridized carbons
        # features.append(Descriptors.NumHeteroatoms(mol))  # Number of heteroatoms

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error calculating descriptors for molecule: {e}")
        # Return zeros array with expected length if calculation fails
        return np.zeros(10, dtype=np.float32)
    
def calculate_vsa_vectors(smiles, vsa_type='smr'):
    """Compute SMR_VSA1-10, SlogP_VSA1-10, PEOE_VSA1-14 vectors and return as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        smr = [float(getattr(Descriptors, f"SMR_VSA{i}")(mol)) for i in range(1, 11)]
        slogp = [float(getattr(Descriptors, f"SlogP_VSA{i}")(mol)) for i in range(1, 11)]
        peoe = [float(getattr(Descriptors, f"PEOE_VSA{i}")(mol)) for i in range(1, 15)]
    except Exception:
        return None
    if vsa_type == 'smr':
        return np.array(smr)
    elif vsa_type == 'slogp':
        return np.array(slogp)
    else:
        return np.array(peoe)

def mol2graph(mol, atom_featurizer, bond_featurizer):
    x = torch.tensor(
        np.stack([atom_featurizer(atom) for atom in mol.GetAtoms()]),
        dtype=torch.float
    )
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_featurizer(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_featurizer.__len__()), dtype=torch.float)

    global_features = get_global_features(mol)
    global_features = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)

def smiles2graph(x_smiles, atom_featurizer=get_multi_hot_atom_featurizer('V1'), bond_featurizer=MultiHotBondFeaturizer(), y=None, properties=None):
    data_list = []
    if properties is None:
        properties = [None] * len(x_smiles)
    else:
        # Ensure properties is a list of lists
        if not isinstance(properties[0], (list, tuple)):
            properties = [[prop] for prop in properties]
    
    # Check if labels are provided
    if y is None:
        y = [None] * len(x_smiles)

    for smiles, label, prop in zip(x_smiles, y, properties):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[Warning] Failed to parse SMILES: {smiles}")
            continue
        graph = mol2graph(mol, atom_featurizer, bond_featurizer)
        graph.smiles = smiles

        prop_vals = prop
        if prop_vals is not None:
            for i, prop_val in enumerate(prop_vals):
                if prop_val is not None:
                    prop_tensor = torch.tensor(prop_val, dtype=torch.float).view(1, -1)
                    setattr(graph, f"property_{i}", prop_tensor)  # Dynamically set property attributes
        
        if label is not None:
            graph.y = torch.tensor(label, dtype=torch.float32).view(1, -1)
        
        data_list.append(graph)

    return data_list
