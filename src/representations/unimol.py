"""Frozen Uni-Mol molecular embeddings.

Uses a pretrained Uni-Mol v1 encoder to generate 512-dimensional
molecular representations from SMILES.
"""

from typing import ClassVar
from rdkit.Chem import AllChem

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation


_OUTPUT_DIM = 512
_BATCH_SIZE = 32


class UniMol(Representation):
    """Generate 512-dimensional embeddings using pretrained Uni-Mol."""

    name: ClassVar[str] = "unimol"

    def __init__(self) -> None:
        self._model = None 

    def _ensure_loaded(self) -> None:
        """Load the pretrained Uni-Mol model only when needed."""
        if self._model is None:
            import torch
            from unimol_tools import UniMolRepr

            self._model = UniMolRepr(
                data_type="molecule",
                model_name="unimolv1",
                use_cuda=torch.cuda.is_available(),
            )
    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return deterministic 512-dimensional Uni-Mol embeddings."""
        self._ensure_loaded()

        smiles_list = smiles.to_list()
        out = np.full(
            (len(smiles_list), _OUTPUT_DIM),
            np.nan,
            dtype=np.float64,
        )

        failed_conformers = 0

        for start in tqdm(
            range(0, len(smiles_list), _BATCH_SIZE),
            desc="Uni-Mol embeddings",
            unit="batch",
        ):
            chunk = smiles_list[start:start + _BATCH_SIZE]

            valid_idx = []
            atoms_list = []
            coordinates_list = []

            for i, smi in enumerate(chunk):
                if not isinstance(smi, str):
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                try:
                    mol = Chem.AddHs(mol)

                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42

                    if AllChem.EmbedMolecule(mol, params) != 0:
                        failed_conformers += 1
                        continue

                    if not AllChem.MMFFHasAllMoleculeParams(mol):
                        failed_conformers += 1
                        continue

                    if AllChem.MMFFOptimizeMolecule(
                        mol, maxIters=1000
                    ) != 0:
                        failed_conformers += 1
                        continue

                    atoms = [
                        atom.GetSymbol()
                        for atom in mol.GetAtoms()
                    ]

                    coordinates = (
                        mol.GetConformer()
                        .GetPositions()
                        .astype(np.float32)
                    )

                    if not np.isfinite(coordinates).all():
                        failed_conformers += 1
                        continue

                    valid_idx.append(i)
                    atoms_list.append(atoms)
                    coordinates_list.append(coordinates)

                except (ValueError, RuntimeError, TypeError):
                    failed_conformers += 1
                    continue

            if not valid_idx:
                continue

            embeddings = np.asarray(
                self._model.get_repr({
                    "atoms": atoms_list,
                    "coordinates": coordinates_list,
                }),
                dtype=np.float64,
            )

            if embeddings.shape != (len(valid_idx), _OUTPUT_DIM):
                raise ValueError(
                    f"Unexpected Uni-Mol embedding shape: {embeddings.shape}"
                )

            for row, i in enumerate(valid_idx):
                if np.isfinite(embeddings[row]).all():
                    out[start + i] = embeddings[row]

        print(f"Failed 3D conformers: {failed_conformers}/{len(smiles_list)}")

        return out