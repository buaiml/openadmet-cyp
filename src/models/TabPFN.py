"""
TabPFN CYB interface implementation with CheMeleon embeddings and PCA 


Input dimesions: TabPFN-3.5:  20,000 features and 1,000,000 rows limit.
"""

"""TabPFN regressor for the CYP challenge."""

from typing import ClassVar                                                                                               
import numpy as np
from sklearn.pipeline import Pipeline

from models.base import CYPModel
from tabpfn import TabPFNRegressor

class TabPFN(CYPModel):
    """
    TabPFN is a foundational model which develops guess based on previous unlabeled and labeled 
    pattern-related data.
    """

    name: ClassVar[str] = "TabPFN"

    def __init__(self) -> None:
        self._model = Pipeline(
            [
                ("regressor", TabPFNRegressor()),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    