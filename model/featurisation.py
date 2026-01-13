import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch_geometric.data import Data

# Step 1: Atom Featurisation
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    fearure_vector = [atom_type_enc, n_heavy_neighbors_enc, formal_charge_enc, hybridisation_type_enc, is_in_a_ring_enc, is_aromatic_enc, atomic_mass_scaled, vdw_radius_scaled, covalent_radius_scaled]
    atom_feature_vector = []
    for i in fearure_vector:
        atom_feature_vector += i

    # atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

# Step 2: Bond Featurisation
def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    if bond is None:
        return np.zeros(10)
    
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

def get_global_features(mol):
    """
    Extract global molecular features (like molecular weight, LogP, etc.) from an RDKit mol object.
    """
    from rdkit.Chem import Descriptors
    
    mol_weight = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    num_h_bond_donors = Descriptors.NumHDonors(mol)
    num_h_bond_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)  # Topological polar surface area

    return np.array([mol_weight, logP, num_h_bond_donors, num_h_bond_acceptors, tpsa])



target_scaler = StandardScaler()

def smiles2graph(
    x_smiles, y=None, cluster=None, properties=None, test=False
):
    """
    Converts a list of SMILES strings into a list of PyTorch Geometric graph data objects.

    Args:
    - x_smiles: List of SMILES strings.
    - y: List of labels (optional). Default is None. Each label should be a 4-dimensional vector.
    - cluster: List of cluster labels (optional). Default is None. Each value corresponds to a cluster ID.
    - properties: List of property lists (optional). Default is None. Each property list corresponds to a set of properties.
    - test: Boolean flag indicating whether to skip global features for test data.

    Returns:
    - data_list: List of PyTorch Geometric Data objects.
    """
    data_list = []
    
    # Check if clustering information is provided
    if cluster is None:
        cluster = [None] * len(x_smiles)  # If no clusters, set it to None for all SMILES
    
    # Check if properties are provided
    if properties is None:
        properties = [None] * len(x_smiles)
    else:
        # Ensure properties is a list of lists
        if not isinstance(properties[0], (list, tuple)):
            properties = [[prop] for prop in properties]
    
    # Check if labels are provided
    if y is None:
        y = [None] * len(x_smiles)
    
    # Iterate over the SMILES strings and their corresponding labels, clusters, and properties
    for smiles, prop_vals, y_val, cluster_val in zip(x_smiles, properties, y, cluster):
        # Convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # Skip invalid SMILES
        
        # Get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        
        # Example of unrelated molecule for feature dimensions
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
        
        # Node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype=torch.float)
        
        # Edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)
        
        # Edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        EF = torch.tensor(EF, dtype=torch.float)
        
        # Global features (e.g., molecular descriptors)
        if not test:
            global_features = get_global_features(mol)
            global_features = torch.tensor(np.array(global_features), dtype=torch.float)
        
        # Create the Data object
        if not test:
            data = Data(x=X, edge_index=E, edge_attr=EF, global_features=global_features)
        else:
            data = Data(x=X, edge_index=E, edge_attr=EF)
        
        # Store the SMILES string and cluster ID in the data object for future reference
        data.smiles = smiles
        data.cluster = cluster_val  # Store the cluster information separately
        
        # Add properties to the data object
        if prop_vals is not None:
            for i, prop_val in enumerate(prop_vals):
                if prop_val is not None:
                    prop_tensor = torch.tensor(prop_val, dtype=torch.float).view(1, -1)
                    setattr(data, f"property_{i}", prop_tensor)  # Dynamically set property attributes
                    if test:
                        global_features.append(prop_val)
        
        # Add labels to the data object
        if y_val is not None:
            y_tensor = torch.tensor(y_val, dtype=torch.float).view(1, -1)
            data.y = y_tensor
        
        # Add global features for test data
        if test:
            global_features = np.array(global_features)
            global_features = torch.tensor(np.array(global_features), dtype=torch.float).view(1, -1)
            data.global_features = global_features
        
        # Append the data object to the list
        data_list.append(data)
    
    return data_list