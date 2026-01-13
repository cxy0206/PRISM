from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Bond, BondType
from typing import Optional, Sequence, Any
from abc import abstractmethod
from collections.abc import Sized
from typing import Generic, TypeVar

import numpy as np


# from model.data.molgraph import MolGraph

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of
    type ``T``."""

    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


# class GraphFeaturizer(Featurizer[S, MolGraph]):
#     @property
#     @abstractmethod
#     def shape(self) -> tuple[int, int]:
#         ...



class MultiHotBondFeaturizer(VectorFeaturizer):
    """
    Featurizes bonds based on:
      - null-ity (is bond None?)
      - bond type
      - conjugated
      - in ring
      - stereochemistry

    Feature vector layout (default):
      [0]          : null flag
      [1 : 1+T]    : one-hot bond type (T types)
      [1+T]        : conjugated?
      [1+T+1]      : in ring?
      [1+T+2 : end]: one-hot stereo type (S types + unknown)
    """
    def __init__(
        self,
        bond_types: Optional[Sequence[BondType]] = None,
        stereos: Optional[Sequence[int]] = None,
    ):
        # 默认 bond types 和 stereo types
        self.bond_types = list(bond_types) if bond_types is not None else [
            BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC
        ]
        self.stereos = list(stereos) if stereos is not None else list(range(6))
        # 特征长度：null + bond types + conjugated + in_ring + (stereo + unknown)
        self._size = (
            1
            + len(self.bond_types)
            + 1
            + 1
            + (len(self.stereos) + 1)
        )

    def __len__(self) -> int:
        return self._size

    def __call__(self, b) -> np.ndarray:
        x = np.zeros(self._size, dtype=int)
        # null-ity
        if b is None:
            x[0] = 1
            return x
        # bond type one-hot
        bond_type = b.GetBondType()
        idx_bt = self.bond_types.index(bond_type) if bond_type in self.bond_types else len(self.bond_types)
        x[1 + idx_bt] = 1
        # conjugated flag
        x[1 + len(self.bond_types)] = int(b.GetIsConjugated())
        # in ring flag
        x[1 + len(self.bond_types) + 1] = int(b.IsInRing())
        # stereo one-hot (including unknown)
        base = 1 + len(self.bond_types) + 2
        stereo_val = int(b.GetStereo())
        idx_st = self.stereos.index(stereo_val) if stereo_val in self.stereos else len(self.stereos)
        x[base + idx_st] = 1
        return x

    @classmethod
    def one_hot_index(cls, x, xs):
        """Returns (index, length+1) for x in xs or (length, length+1) if not in xs."""
        n = len(xs)
        if x in xs:
            return xs.index(x), n + 1
        else:
            return n, n + 1


class RIGRBondFeaturizer(VectorFeaturizer[Bond]):
    """A :class:`RIGRBondFeaturizer` feauturizes bonds based on only the resonance-invariant features:

    * ``null``-ity (i.e., is the bond ``None``?)
    * in ring?
    """

    def __len__(self):
        return 2

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        x[1] = int(b.IsInRing())

        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence):
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1