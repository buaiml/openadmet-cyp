"""Shared preprocessing utilities for CYP models."""

from typing import Optional, Tuple

import numpy as np


def drop_nan_rows(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    label: str = "",
    *,
    return_mask: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]] | Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Drop rows with any non-finite feature, or (if y given) a non-finite target.

    CYP target columns are sparse — most compounds were only assayed against
    a subset of the four enzymes — so a non-finite/missing y is expected and
    must be filtered here, not just non-finite X (invalid SMILES).
    Prints a message if any rows are dropped.
    Returns (X_clean, y_clean) — y_clean is None when y is None.
    With return_mask=True, also return the same row mask for aligned metadata.
    """
    mask = np.isfinite(X).all(axis=1)
    if y is not None:
        mask &= np.isfinite(y)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        tag = f" ({label})" if label else ""
        print(f"Dropped {n_dropped} of {len(X)} examples with non-finite features/target{tag}")
    cleaned = (X[mask], y[mask] if y is not None else None)
    return (*cleaned, mask) if return_mask else cleaned
