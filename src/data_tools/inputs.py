"""Registry of molecular input featurizations for the CYP challenge."""

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from representations.chemeleon import CheMeleon
from representations.rdkit_descriptors import RDKitDescriptors

_rdkit = RDKitDescriptors()
_chemeleon = CheMeleon()


def _rdkit_features(df: pl.DataFrame) -> np.ndarray:
    return _rdkit.transform(df["SMILES"])


def _chemeleon_features(df: pl.DataFrame) -> np.ndarray:
    return _chemeleon.transform(df["SMILES"])


INPUT_REGISTRY: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
    _rdkit.name: _rdkit_features,
    _chemeleon.name: _chemeleon_features,
}


def _featurize_one(df: pl.DataFrame, input_name: str, cache_dir: Path) -> np.ndarray:
    """Return features for df, computing only molecules not already in the cache.

    Cache is content-addressed by SMILES, not by split/seed — a molecule's
    representation doesn't depend on how the data was split, so the same
    cache is reused across every split type and seed.

    Cache layout (one pair of files per representation):
      {input_name}.npy  — float64 array of shape (N_cached, D); grows as new molecules arrive
      {input_name}.idx  — text file, one SMILES per line; line i = row i of .npy
    """
    smiles_req = df["SMILES"].to_list()
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{input_name}.npy"
    idx_path = cache_dir / f"{input_name}.idx"

    if idx_path.exists():
        indexed = idx_path.read_text().splitlines()
        pos: dict[str, int] = {s: i for i, s in enumerate(indexed)}
    else:
        indexed, pos = [], {}

    missing = list(dict.fromkeys(s for s in smiles_req if s not in pos))
    if missing:
        new_features = INPUT_REGISTRY[input_name](pl.DataFrame({"SMILES": missing}))
        full = np.vstack([np.load(str(npy_path)), new_features]) if npy_path.exists() else new_features
        np.save(str(npy_path), full)
        for s in missing:
            pos[s] = len(indexed)
            indexed.append(s)
        idx_path.write_text("\n".join(indexed))
        print(f"[{input_name}] appended {len(missing)} molecules (cache now {len(indexed)} total)")
    else:
        print(f"[{input_name}] loaded from cache ({len(indexed)} molecules indexed)")

    return np.load(str(npy_path))[[pos[s] for s in smiles_req]]


def featurize(
    df: pl.DataFrame,
    input_names: str | list[str],
    cache_dir: Path = Path("data/features"),
) -> np.ndarray:
    """Return feature matrix for df, computing only uncached molecules.

    Pass a single name or a list; multiple inputs are hstacked after loading
    each from its own cache file.
    """
    names = [input_names] if isinstance(input_names, str) else input_names
    unknown = [n for n in names if n not in INPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")
    arrays = [_featurize_one(df, name, cache_dir) for name in names]
    return arrays[0] if len(arrays) == 1 else np.hstack(arrays)
