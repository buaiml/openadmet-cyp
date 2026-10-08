"""TabPFN regressor with train-fitted PCA for precomputed molecular features."""

from typing import ClassVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from models.base import CYPModel


class TabPFN(CYPModel):
    """Compress features to at most 200 dimensions before TabPFN inference.

    CheMeleon embeddings are supplied by the shared featurizer. PCA is fitted
    only on training features and reused unchanged for predictions. Small
    datasets or narrow representations use fewer components when necessary.
    """

    name: ClassVar[str] = "tabpfn"

    def __init__(self, n_components: int = 200, random_state: int = 42) -> None:
        from tabpfn import TabPFNRegressor

        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self._model = Pipeline(
            [
                ("pca", PCA(n_components=n_components, random_state=random_state)),
                ("regressor", TabPFNRegressor(random_state=random_state)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.set_params(pca__n_components=min(self.n_components, *X.shape))
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
