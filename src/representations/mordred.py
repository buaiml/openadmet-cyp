"""2D Mordred descriptors using the optional mordredcommunity package."""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import polars as pl
from rdkit import Chem

from representations.base import Representation

if TYPE_CHECKING:
    from mordred import Calculator


class MordredDescriptors(Representation):
    """All configured 2D descriptors in a fixed calculator-defined order.

    Invalid, null, and blank SMILES produce an all-NaN row. Missing or
    non-finite descriptor values remain NaN without dropping any columns.
    The calculator is loaded on first use and reused for later calls.
    """

    name: ClassVar[str] = "mordred"

    def __init__(self) -> None:
        self._calculator: "Calculator | None" = None

    def _get_calculator(self) -> "Calculator":
        """Load the optional dependency and initialize the 2D calculator once."""
        if self._calculator is None:
            try:
                from mordred import Calculator, descriptors
            except ModuleNotFoundError as exc:
                if exc.name != "mordred":
                    raise
                raise ImportError(
                    "Mordred descriptors require mordredcommunity. "
                    "Install it with: python -m pip install mordredcommunity"
                ) from exc
            self._calculator = Calculator(descriptors, ignore_3D=True)
        return self._calculator

    @property
    def feature_names(self) -> list[str]:
        """Return descriptor names in the same order as the feature columns."""
        return [str(descriptor) for descriptor in self._get_calculator().descriptors]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, n_descriptors) float64 matrix in input order."""
        calculator = self._get_calculator()
        out = np.full((len(smiles), len(calculator.descriptors)), np.nan, dtype=np.float64)
        for i, smi in enumerate(smiles.to_list()):
            if smi is None or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                result = calculator(mol).fill_missing(np.nan)
                out[i] = np.asarray(list(result), dtype=np.float64)
        out[~np.isfinite(out)] = np.nan
        return out
